"""K3/T1: strict observer isolation; copied from v2 sandbox with a trusted preload launcher.

Adapted from glomergence env/{sandbox,node_launcher}.py at
e602f63207fe4e67eeda4dff7b53d49e92cc318b. Requires root + Landlock, no weak
fallback. CUDA driver access remains part of the trusted computing base.
RSS, workdir size and VRAM caps are sampled, not cgroup hard limits; rlimits
additionally bound processes, address space, CPU and individual files.
"""
from __future__ import annotations
import ctypes
import fcntl
import glob
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UID = 64532

@dataclass(frozen=True)
class Limits:
    """K3: compiler process groups allowed; every descendant keeps our UID."""
    wall_seconds: float = 120
    cpu_seconds: int = 90
    address_bytes: int = 256 << 30
    rss_bytes: int = 8 << 30
    file_bytes: int = 512 << 20
    work_bytes: int = 2 << 30
    processes: int = 128
    gpu_bytes: int | None = 32 << 30

@dataclass
class SandboxResult:
    component: str
    returncode: int
    elapsed_seconds: float
    peak_rss_bytes: int
    peak_gpu_bytes: int
    workdir: str
    log: str

def landlock_abi():
    return ctypes.CDLL(None, use_errno=True).syscall(444, None, 0, 1)

def uid_processes():
    records = []
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            data = dict(line.split(':', 1) for line in path.read_text().splitlines())
            if int(data['Uid'].split()[0]) == UID:
                records.append((int(path.parent.name), int(data.get('VmRSS', '0').split()[0]) * 1024))
        except (OSError, ValueError, KeyError):
            continue
    return records

def read_paths():
    # No project, task, model, oracle, home or other rollout tree.
    paths = ['/usr', '/lib', '/lib64', '/bin', '/sbin', '/etc/ld.so.cache',
             '/etc/ld.so.conf', '/etc/localtime', '/etc/passwd', '/etc/group',
             '/proc/self', '/proc/cpuinfo', '/proc/meminfo', '/proc/driver/nvidia',
             '/proc/sys/vm/mmap_min_addr', '/proc/devices', '/etc/ssl/openssl.cnf',
             '/sys/devices', '/sys/bus/pci', '/sys/module/nvidia',
             str(ROOT / '.venv'), str(Path(sys.base_prefix).resolve())]
    # Resolve /proc/self in the child, not the supervisor (distinct PIDs).
    return [p for p in paths if Path(p).exists()]

def run(source: str, directory: Path, limits: Limits = Limits(), *, files=None,
        monitor_gpu=True, runner: str | None = None) -> SandboxResult:
    """Sandbox imports/compilation too. Input files must never include an oracle.

    Result files remain untrusted; callers must parse bounded non-pickle data.
    """
    if os.geteuid() != 0 or landlock_abi() < 1:
        raise RuntimeError('C1 requires root and Landlock ABI >= 1; no weak fallback')
    directory = directory.resolve()
    directory.mkdir(parents=True, mode=0o700)
    lockpath = ROOT / 'runs/sandbox.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if uid_processes():
            raise RuntimeError('Sandbox UID has processes; refuse overlapping rollouts')
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):
            raise RuntimeError('Cannot enable child subreaper')
        # Traverse, never list, through root-owned private ancestors such as /root.
        for ancestor in [directory, *directory.parents, *Path(sys.base_prefix).resolve().parents]:
            if ancestor.is_dir() and ancestor.stat().st_uid == 0:
                os.chmod(ancestor, ancestor.stat().st_mode | 0o001)
        work = directory / 'work'
        work.mkdir(mode=0o700)
        os.chown(work, UID, UID)
        (work / 'candidate.py').write_text(source)
        (work / 'candidate.py').chmod(0o444)
        for name, data in (files or {}).items():
            if Path(name).name != name or name == 'candidate.py':
                raise ValueError('Flat input names required')
            if isinstance(data, Path):
                import shutil
                # Only consume newly generated, private v2 staging files.
                # Moving within this filesystem avoids a second tensor copy.
                stage = directory.parent / 'staged-inputs'
                if (data.is_symlink() or data.parent.resolve() != stage.resolve()
                        or not data.is_file() or data.stat().st_uid != os.geteuid()
                        or data.stat().st_nlink != 1):
                    raise ValueError('v2 inputs must be private, owned staging files')
                if data.stat().st_dev != work.stat().st_dev:
                    raise ValueError('v2 staging must stay on the same filesystem')
                data.rename(work / name)
            else:
                (work / name).write_bytes(data)
            (work / name).chmod(0o444)
        cfg = {'work': str(work), 'limits': asdict(limits), 'uid': UID,
               'read_paths': read_paths(), 'devices': glob.glob('/dev/nvidia*'),
               'python': sys.executable, 'parent': os.getpid(), 'runner': runner}
        config = directory / 'sandbox.json'
        config.write_text(json.dumps(cfg))
        config.chmod(0o600)
        launcher = ROOT / 'research/reliability_v2/strict_launcher.py'
        launcher.chmod(0o600)
        gpu = None
        gpu_baseline = 0
        if monitor_gpu:
            import pynvml
            pynvml.nvmlInit()
            gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_baseline = pynvml.nvmlDeviceGetMemoryInfo(gpu).used
        log = directory / 'worker.log'
        t0, peak_rss, peak_gpu, reason = time.monotonic(), 0, 0, None
        with log.open('xb') as output:
            log.chmod(0o600)
            child = subprocess.Popen([sys.executable, '-I', '-S', str(launcher), str(config)],
                                     cwd=work, env={'PATH': '/usr/local/cuda/bin:/usr/bin:/bin'},
                                     stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True, close_fds=True)
            try:
                while child.poll() is None:
                    procs = uid_processes()
                    rss = sum(x[1] for x in procs)
                    peak_rss = max(peak_rss, rss)
                    if gpu is not None:
                        # NVML PIDs can be host PIDs outside this container's PID
                        # namespace. Serialized eval + sleeping policy permits a
                        # device-memory delta instead of a misleading PID join.
                        used = max(0, pynvml.nvmlDeviceGetMemoryInfo(gpu).used - gpu_baseline)
                        peak_gpu = max(peak_gpu, used)
                        if limits.gpu_bytes is not None and used > limits.gpu_bytes:
                            reason = 'gpu_memory_limit'
                    if rss > limits.rss_bytes:
                        reason = 'cpu_memory_limit'
                    if time.monotonic() - t0 > limits.wall_seconds:
                        reason = 'timeout'
                    size = 0
                    for p in work.rglob('*'):
                        try:
                            size += p.lstat().st_size
                        except FileNotFoundError:  # compiler removes temporary files
                            pass
                    if size > limits.work_bytes:
                        reason = 'disk_limit'
                    if reason:
                        break
                    time.sleep(0.05)
            finally:
                # Ninja creates child process groups. Freeze every dedicated-UID
                # descendant before killing so fork races cannot evade cleanup.
                for pid, _ in uid_processes():
                    try:
                        os.kill(pid, signal.SIGSTOP)
                    except ProcessLookupError:
                        pass
                for pid, _ in uid_processes():
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('GPU worker cannot be reaped; stop before another rollout') from None
                deadline = time.monotonic() + 10
                while uid_processes():
                    for pid, _ in uid_processes():
                        try:
                            os.kill(pid, signal.SIGKILL)
                            os.waitpid(pid, os.WNOHANG)
                        except (ChildProcessError, ProcessLookupError):
                            pass
                    if time.monotonic() > deadline:
                        raise RuntimeError('Sandbox descendants remain; stop before another rollout')
                    time.sleep(0.05)
        code = child.returncode
        if reason is None:
            reason = {-signal.SIGSYS: 'sandbox_blocked', -signal.SIGXCPU: 'cpu_limit',
                      -signal.SIGXFSZ: 'disk_limit', 0: 'ok', 77: 'gpu_oom',
                      78: 'sandbox_blocked', 79: 'process_limit'}.get(code, 'crash')
        result = SandboxResult(reason, code, time.monotonic()-t0, peak_rss, peak_gpu,
                               str(work), str(log))
        (directory / 'result.json').write_text(json.dumps(asdict(result), indent=2) + '\n')
        return result
