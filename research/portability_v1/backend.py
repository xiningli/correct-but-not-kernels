"""Separate short-lived GPU containers; no candidate sees host data or oracles."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
from .common import IMAGE, PYTHON, VENV, BASE_PYTHON, write, utc


def gpu_processes():
    r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                       check=True, capture_output=True, text=True, timeout=15)
    return r.stdout.strip()


def idle(seconds=600):
    deadline = time.monotonic() + seconds
    while gpu_processes():
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU is busy with another process; no process was interrupted')
        time.sleep(5)


def docker_command(name, script, inputs, output, plan, extra=None):
    command = ['docker', 'run', '--name', name, '--init', '--network', 'none',
        '--gpus', 'device=0', '--user', f'{os.getuid()}:{os.getgid()}', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--read-only',
        '--memory', str(plan['docker_memory_bytes']), '--memory-swap', str(plan['docker_memory_bytes']),
        '--cpus', str(plan['docker_cpus']), '--pids-limit', str(plan['docker_pids']),
        '--shm-size', '1g', '--tmpfs', '/tmp:rw,exec,nosuid,size=4g,mode=1777',
        '--workdir', '/out', '--ulimit', 'core=0', '--ulimit', 'nofile=4096:4096',
        '--ulimit', 'fsize=4294967296:4294967296']
    for key, val in {'HOME': '/tmp', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
        'PATH': VENV + '/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
        'MAX_JOBS': '4', 'TORCH_CUDA_ARCH_LIST': '12.0', 'TRITON_CACHE_DIR': '/tmp/triton',
        'TORCH_EXTENSIONS_DIR': '/tmp/extensions', 'XDG_CACHE_HOME': '/tmp/cache'}.items():
        command += ['--env', key + '=' + val]
    for src, dst, readonly in [
        (VENV, VENV, True), (BASE_PYTHON, BASE_PYTHON, True),
        (str(Path(script).resolve()), '/runner.py', True),
        (str(Path(inputs).resolve()), '/input', True),
        (str(Path(output).resolve()), '/out', False), *(extra or [])]:
        command += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if readonly else '')]
    return command + ['--entrypoint', PYTHON, IMAGE, '-u', '/runner.py']


def run(script, inputs, directory, plan, *, seconds, extra=None):
    idle(plan['busy_gpu_wait_seconds'])
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    output = directory / 'work'; output.mkdir()
    if shutil.disk_usage(directory).free < plan['minimum_free_disk_bytes']:
        raise RuntimeError('Insufficient disk headroom before GPU execution')
    name = 'cuda-port-v1-' + uuid.uuid4().hex[:16]
    command = docker_command(name, script, inputs, output, plan, extra)
    write(directory / 'launch.json', {'container': name, 'command': command, 'utc': utc()})
    reason = None; started = time.monotonic(); maximum_gpu = 0
    with (directory / 'worker.log').open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() - started > seconds:
                    reason = 'timeout'; break
                if log.tell() > 8 << 20:
                    reason = 'log_limit'; break
                used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], text=True, timeout=15)
                maximum_gpu = max(maximum_gpu, int(used.strip()) << 20)
                if maximum_gpu > plan['gpu_stop_bytes']:
                    reason = 'gpu_memory_limit'; break
                size = 0
                for p in output.rglob('*'):
                    try:
                        if not p.is_symlink():
                            size += p.lstat().st_size
                    except FileNotFoundError:
                        pass  # worker consumes its read-only input symlinks
                if size > plan['worker_disk_limit_bytes']:
                    reason = 'disk_limit'; break
                time.sleep(1)
        finally:
            if process.poll() is None:
                subprocess.run(['docker', 'kill', name], capture_output=True, timeout=30)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)
            inspected = subprocess.run(['docker', 'inspect', name, '--format', '{{json .State}}'],
                                       capture_output=True, text=True, timeout=30)
            state = json.loads(inspected.stdout) if inspected.returncode == 0 else {}
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=30)
    if state.get('OOMKilled'):
        reason = 'cpu_memory_limit'
    result = {'returncode': process.returncode, 'reason': reason, 'container_state': state,
              'seconds': time.monotonic() - started, 'peak_device_used_bytes': maximum_gpu,
              'work': str(output), 'log': str(directory / 'worker.log')}
    write(directory / 'execution.json', result)
    if not state:
        raise RuntimeError('Container did not start; inspect ' + result['log'])
    return result
