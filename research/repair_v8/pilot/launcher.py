"""K3/T1: trusted profiler preload; seccomp permits only thread naming within prctl.

Adapted from glomergence/env/node_launcher.py at
e602f63207fe4e67eeda4dff7b53d49e92cc318b. Never loads reference/expected values.
"""
import ctypes
import inspect
import json
import os
import resource
import signal
import struct
import sys

def filter_bytes():
    # Permit compiler subprocesses subject to RLIMIT_NPROC; deny escape/network.
    # Ninja's posix_spawn uses setpgid; the supervisor kills/reaps the dedicated
    # UID across process groups. setsid and namespace creation remain forbidden.
    deny = {41, 42, 43, 49, 50, 53, 101, 112, 129, 132, 133, 135,
            155, 161, 163, 165, 166, 169, 175, 176, 179, 200, 235,
            246, 248, 249, 250, 256, 259, 272, 279, 280, 288, 297, 298,
            304, 308, 310, 311, 313, 321, 323, 424, 425, 426, 427,
            428, 429, 430, 431, 432, 433, 434, 438, 440, 442, 448,
            # Landlock ABI 1 doesn't mediate truncate; deny explicitly.
            76, 85, 188, 189, 190, 197, 198, 199}
    ins = []
    def op(c, k, jt=0, jf=0):
        ins.append(struct.pack('HBBI', c, jt, jf, k & 0xffffffff))
    op(0x20, 4); op(0x15, 0xc000003e, 1); op(0x06, 0x80000000)
    op(0x20, 0); op(0x35, 0x40000000, 0, 1); op(0x06, 0x80000000)
    # Triton 3.4 LLVM workers call prctl(PR_SET_NAME, "llvm-worker-0").
    # PyTorch autograd also reads that name with PR_GET_NAME. Both affect only
    # the calling thread's name. All other prctl operations
    # (including dumpability, seccomp and process controls) remain fatal.
    op(0x20, 0); op(0x15, 157, 0, 5)
    op(0x20, 16); op(0x15, 15, 2, 0); op(0x15, 16, 1, 0)
    op(0x06, 0x80000000); op(0x06, 0x7fff0000)
    for nr in sorted(deny):
        op(0x20, 0); op(0x15, nr, 0, 1); op(0x06, 0x80000000)
    for nr in (435, 437):  # clone3/openat2 opaque args: libc fallback
        op(0x20, 0); op(0x15, nr, 0, 1); op(0x06, 0x00050000 | 38)
    op(0x20, 0); op(0x15, 56, 0, 3)  # clone: prohibit namespaces
    op(0x20, 16); op(0x45, 0x7e020000, 0, 1); op(0x06, 0x80000000)
    op(0x06, 0x7fff0000)
    return b''.join(ins)

def landlock(libc, cfg):
    class Attr(ctypes.Structure):
        _fields_ = [('access', ctypes.c_uint64)]
    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('access', ctypes.c_uint64), ('fd', ctypes.c_int)]
    handled = (1 << 13) - 1
    attr = Attr(handled)
    ruleset = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), 'Landlock create')
    paths = [(p, 1 | 4 | 8) for p in cfg['read_paths']]
    paths += [(cfg['work'], handled)]
    # CUDA 13 names its helper threads through task/<tid>/comm; this is
    # process-local metadata, not persistent filesystem output. UID + ptrace
    # restrictions keep other processes inaccessible. No create/delete rights.
    paths += [('/proc/self/task', 2 | 4 | 8)]
    paths += [(p, 2 | 4) for p in cfg['devices'] + ['/dev/null', '/dev/zero', '/dev/urandom']]
    for path, access in paths:
        if not os.path.exists(path):
            continue
        if not os.path.isdir(path):
            access &= 1 | 2 | 4
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        rule = Rule(access, fd)
        if libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0):
            raise OSError(ctypes.get_errno(), 'Landlock rule ' + path)
        os.close(fd)
    if libc.syscall(446, ruleset, 0):
        raise OSError(ctypes.get_errno(), 'Landlock restrict')
    os.close(ruleset)

def main():
    cfg = json.load(open(sys.argv[1]))
    # The launcher file is root-only; capture trusted bootstrap text before
    # dropping to the evaluation UID. No candidate source is read here.
    landlock_source = inspect.getsource(landlock) if cfg.get('sanitizer') else ''
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    if os.uname().machine != 'x86_64':
        raise RuntimeError('Unsupported syscall architecture')
    os.closerange(3, 1 << 16)
    caps = cfg['limits']
    for kind, value in [(resource.RLIMIT_CPU, caps['cpu_seconds']),
                        (resource.RLIMIT_AS, caps['address_bytes']),
                        (resource.RLIMIT_NPROC, caps['processes']),
                        (resource.RLIMIT_FSIZE, caps['file_bytes']),
                        (resource.RLIMIT_CORE, 0), (resource.RLIMIT_NOFILE, 256)]:
        resource.setrlimit(kind, (value, value))
    if libc.prctl(38, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'no_new_privs')
    if not cfg.get('sanitizer'):
        landlock(libc, cfg)
    os.setgroups([])
    os.setgid(cfg['uid'])
    os.setuid(cfg['uid'])
    libc.prctl(1, signal.SIGKILL, 0, 0, 0)
    if os.getppid() != cfg['parent']:
        os._exit(80)
    work = cfg['work']
    env = {'PATH': os.path.dirname(cfg['python']) + ':/usr/local/cuda/bin:/usr/bin:/bin', 'HOME': work, 'TMPDIR': work,
           'LANG': 'C.UTF-8', 'CUDA_HOME': '/usr/local/cuda', 'MAX_JOBS': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'TORCH_EXTENSIONS_DIR': work + '/extensions', 'TRITON_CACHE_DIR': work + '/triton',
           'TORCHINDUCTOR_CACHE_DIR': work + '/inductor', 'TORCHINDUCTOR_COMPILE_THREADS': '1',
           'CUDA_CACHE_PATH': work + '/cuda', 'PYTHONDONTWRITEBYTECODE': '1',
           'CUDA_VISIBLE_DEVICES': '0', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    # CUDA 13 needs a local socket during cuInit. Initialize the trusted runtime
    # after Landlock/UID/rlimits, then atomically apply seccomp to ALL threads.
    # No candidate is imported/read/executed before TSYNC succeeds. -I excludes
    # workdir/PYTHONPATH from trusted imports. No oracle exists in this process.
    bootstrap = '''import ctypes, os, runpy, tempfile
LANDLOCK_BOOTSTRAP
import torch
torch.set_num_threads(2)
torch.cuda.init()
# Profiler imports initialize an asyncio self-pipe. Complete trusted runtime
# initialization before TSYNC, just like cuInit; never relax candidate syscalls.
import torch._dynamo
# urllib3 probes IPv6 support by opening a local socket during import. Liger's
# Transformers dependency reaches it after worker entry. Initialize this pinned
# dependency before TSYNC; candidate network syscalls remain prohibited.
import requests
from torch.utils._python_dispatch import TorchDispatchMode
from contextlib import nullcontext
with (nullcontext() if SANITIZER_MODE else torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                       torch.profiler.ProfilerActivity.CUDA])) as _preload_profile:
    _preload_tensor = torch.ones(16, device='cuda') + 1
    torch.cuda.synchronize()
del _preload_tensor, _preload_profile
# Landlock ABI 1 denies cross-directory rename. Retain atomic publication by
# creating the temporary file in the destination directory (same filesystem).
from triton.runtime.cache import FileCacheManager
def cache_put(self, data, filename, binary=True):
    path = self._make_path(filename)
    fd, temp = tempfile.mkstemp(prefix='.tmp-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'wb' if isinstance(data, bytes) else 'w') as out:
            out.write(data if isinstance(data, bytes) else str(data))
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path
FileCacheManager.put = cache_put
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
class Prog(ctypes.Structure):
    _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.c_void_p)]
raw = RAW_FILTER
buf = ctypes.create_string_buffer(raw)
prog = Prog(len(raw)//8, ctypes.addressof(buf))
if libc.syscall(317, 1, 1, ctypes.byref(prog)) != 0:
    raise OSError(ctypes.get_errno(), 'seccomp TSYNC failed')
exec(compile(PRIVATE_RUNNER, '<private-harness>', 'exec'), {'__name__': '__main__'})
'''.replace('SANITIZER_MODE', repr(bool(cfg.get('sanitizer')))).replace('LANDLOCK_BOOTSTRAP', (
        landlock_source + '\nlibc=ctypes.CDLL(None,use_errno=True)\nlibc.syscall.restype=ctypes.c_long\nlandlock(libc, ' +
        repr({k: cfg[k] for k in ['read_paths', 'work', 'devices']}) + ')\n'
        if cfg.get('sanitizer') else '')).replace('RAW_FILTER', repr(filter_bytes())).replace('PRIVATE_RUNNER', repr(
        cfg['runner'] if cfg['runner'] is not None else
        'import runpy\nrunpy.run_path(' + repr(work + '/candidate.py') + ", run_name='__main__')"))
    command = [cfg['python'], '-I', '-B', '-c', bootstrap]
    if cfg.get('sanitizer'):
        # Start instrumentation after UID setup and environment reset,
        # so the tool's injection survives into the actual CUDA Python process.
        # The tool forks its target, so /proc/self Landlock rules must be bound
        # in the actual Python target (above), before torch or candidate imports.
        # The trusted tool remains outside Landlock, under the same UID/rlimits;
        # candidate Python installs both Landlock and TSYNC before execution.
        tool = '/usr/local/cuda/bin/compute-sanitizer'
        os.execve(tool, [tool, '--tool', 'memcheck', '--error-exitcode', '86',
                        '--target-processes', 'all', *command], env)
    os.execve(cfg['python'], command, env)

if __name__ == '__main__':
    main()
