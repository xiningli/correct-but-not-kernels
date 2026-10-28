"""Trace the trusted v8 Triton smoke after a startup failure; no model or candidate execution.

Run in the locked evaluation venv on AutoDL with strace installed. This diagnostic
does not relax the filter, create a PASS gate, or start the experiment controller.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.geteuid() != 0 or PROJECT.parent != Path('/root/autodl-tmp'):
        raise RuntimeError('Use the dedicated AutoDL deployment')
    tracer = shutil.which('strace')
    if not tracer:
        raise RuntimeError('strace is required to identify the blocked syscall')
    lock = Path('/root/autodl-tmp/repair-v8-pilot.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupied = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20, check=True)
    if occupied.stdout.strip():
        raise RuntimeError('GPU has a compute process; do not overlap the diagnostic')
    out = args.out.resolve()
    if not out.is_relative_to(PROJECT / 'runs'):
        raise ValueError('Keep diagnostic evidence under this deployment runs directory')
    out.mkdir(parents=True, exist_ok=False)
    code = '''import json,sys
from pathlib import Path
from dataclasses import asdict
from research.repair_v8.pilot.controls import SMOKE
from research.repair_v8.pilot.isolation import run
r=run(SMOKE,Path(sys.argv[1]))
print(json.dumps(asdict(r)),flush=True)
sys.exit(0 if r.component=='ok' else 1)
'''
    env = os.environ.copy()
    env.update(PYTHONPATH=str(PROJECT), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    with (out / 'trace-driver.log').open('x') as log:
        # SECCOMP_RET_KILL_PROCESS may omit a delivered SIGSYS siginfo record;
        # capture calls too, so the last syscall of the killed thread is visible.
        result = subprocess.run([tracer, '-f', '-o', str(out / 'syscalls.log'),
                                 sys.executable, '-c', code, str(out / 'smoke')],
                                cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
    signals = [line for line in (out / 'syscalls.log').read_text().splitlines()
               if 'SIGSYS' in line or 'si_syscall=' in line]
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'returncode': result.returncode,
               'sigsys_records': signals, 'model_outputs': 0, 'diagnostic_only': True,
               'existing_gates_unchanged': True, 'filter_relaxed': False}
    (out / 'diagnostic.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
