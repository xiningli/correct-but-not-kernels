"""Bounded AutoDL qualification, evidence preservation, and authorized shutdown."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time

from .prepare import sha, write_json

PROJECT = Path(__file__).resolve().parents[2]


def utc():
    return datetime.now(timezone.utc).isoformat()


def stop_tree(child):
    parents = {}
    for entry in Path('/proc').glob('[0-9]*/status'):
        try:
            fields = dict(line.split(':', 1) for line in entry.read_text().splitlines())
            parents[int(fields['Pid'])] = int(fields['PPid'])
        except (OSError, ValueError, KeyError):
            continue
    descendants = [child.pid]
    for parent in descendants:
        descendants.extend(pid for pid, ppid in parents.items() if ppid == parent and pid not in descendants)
    for pid in reversed(descendants):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def stage(name, command, env, control, timeout):
    write_json(control / 'progress.json', {'stage': name, 'utc': utc(), 'timeout_seconds': timeout})
    with (control / (name + '.log')).open('x') as stream:
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        write_json(control / 'child.json', {'stage': name, 'pid': child.pid, 'command': command, 'utc': utc()})
        try:
            result = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_tree(child)
            raise RuntimeError(name + ' exceeded wall-time budget')
    if result:
        raise RuntimeError(name + ' exited ' + str(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--power-off', action='store_true', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if os.geteuid() != 0 or PROJECT.parent != Path('/root/autodl-tmp') or not Path('/usr/bin/shutdown').is_file():
        raise RuntimeError('Restricted to the dedicated AutoDL data-disk deployment')
    if not root.is_relative_to(PROJECT):
        raise ValueError('Run directory must be within this deployment')
    lock = Path('/root/autodl-tmp/repair-v8-qualification.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    control = root / 'controller'
    control.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update({'PATH': '/root/autodl-tmp/cuda-gemma4-llm/.cache/bin:/usr/local/cuda/bin:/root/miniconda3/bin:' + env.get('PATH', ''),
                'QUALIFICATION_PYTHON': '/root/miniconda3/bin/python3.12',
                'PYTHONPATH': str(PROJECT), 'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1',
                'UV_CACHE_DIR': str(PROJECT / '.cache/uv'), 'PIP_CACHE_DIR': str(PROJECT / '.cache/pip'),
                'TRITON_CACHE_DIR': str(PROJECT / '.cache/triton'), 'CUDA_CACHE_PATH': str(PROJECT / '.cache/cuda'),
                'TMPDIR': str(PROJECT / '.cache/tmp'), 'UV_CONCURRENT_DOWNLOADS': '4',
                'UV_CONCURRENT_INSTALLS': '1', 'UV_HTTP_TIMEOUT': '300',
                'UV_DEFAULT_INDEX': 'https://pypi.tuna.tsinghua.edu.cn/simple', 'OMP_NUM_THREADS': '2',
                'MKL_NUM_THREADS': '2', 'CUDA_VISIBLE_DEVICES': '0'})
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    write_json(control / 'authorization.json', {'utc': utc(), 'pid': os.getpid(),
               'scope': 'Four development candidates: CPU preparation and original before/fixed GPU qualification only',
               'power_off': True, 'basis': 'User requested autonomous decisions and shutdown after completion or blockage',
               'prepare_wall_seconds': 2700, 'gpu_wall_seconds': 14400, 'backup_ack_wait_seconds': 300})
    error = None
    try:
        inventory = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30)
        write_json(control / 'inventory.json', {'utc': utc(), 'returncode': inventory.returncode,
                   'stdout': inventory.stdout, 'stderr': inventory.stderr})
        if inventory.returncode:
            raise RuntimeError('GPU inventory unavailable')
        entry = str(PROJECT / 'scripts/repair_v8_autodl.sh')
        stage('prepare', ['/bin/bash', entry, 'prepare', str(root)], env, control, 2700)
        stage('gpu-controls', ['/bin/bash', entry, 'run', str(root)], env, control, 14400)
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    summaries = []
    for path in sorted((root / 'gpu-controls').glob('*/summary.json')):
        summaries.extend(json.loads(path.read_text()))
    report = {'utc': utc(), 'status': 'blocked' if error else 'complete', 'error': error,
              'tasks': summaries, 'development_qualified': sum(x['status'] == 'development_controls_pass' for x in summaries),
              'formal_eligible': 0, 'model_evaluations': 0, 'training_runs': 0,
              'interpretation': 'Task qualification only; no new evidence about Gemma repair ability.'}
    write_json(control / 'report.json', report)
    archive_error = None
    info = None
    try:
        write_json(control / 'progress.json', {'stage': 'archiving', 'utc': utc()})
        archive = control / 'evidence.tgz'
        roots = [root, PROJECT / 'research/repair_v8', PROJECT / 'scripts', PROJECT / 'experiments']
        with tarfile.open(archive, 'w:gz') as tar:
            for base in roots:
                for path in sorted(base.rglob('*')):
                    if ({'venv', 'sources', 'execution', '__pycache__', '.pytest_cache'} & set(path.relative_to(base).parts)
                            or not path.is_file() or path.is_symlink() or path == archive):
                        continue
                    tar.add(path, arcname=str(path.relative_to(PROJECT)), recursive=False)
        info = {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive), 'utc': utc()}
        write_json(control / 'archive.json', info)
    except Exception as exc:
        archive_error = type(exc).__name__ + ': ' + str(exc)
    write_json(control / 'progress.json', {'stage': 'awaiting_verified_backup', 'utc': utc(), 'archive_error': archive_error})
    ack = control / 'backup-local-ack.json'
    verified = False
    deadline = time.monotonic() + 300
    while info and time.monotonic() < deadline:
        if ack.exists():
            try:
                verified = json.loads(ack.read_text()).get('sha256') == info['sha256']
            except (OSError, ValueError):
                pass
            if verified:
                break
        time.sleep(5)
    write_json(control / 'shutdown-request.json', {'utc': utc(), 'reason': error or 'qualification complete',
               'archive_error': archive_error, 'local_backup_verified': verified,
               'remote_data_disk_retains_evidence': True, 'command': '/bin/bash /usr/bin/shutdown'})
    os.sync()
    try:
        result = subprocess.run(['/bin/bash', '/usr/bin/shutdown'], capture_output=True, text=True, timeout=60)
        write_json(control / 'shutdown-return.json', {'utc': utc(), 'returncode': result.returncode,
                   'stdout': result.stdout, 'stderr': result.stderr})
    except Exception as exc:
        write_json(control / 'shutdown-return.json', {'utc': utc(), 'error': type(exc).__name__ + ': ' + str(exc)})
    os.sync()


if __name__ == '__main__':
    main()
