"""Bounded AutoDL-only operation, resumable artifacts, report and shutdown."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from research.repair_v8.prepare import sha
from .artifacts import write_json
from research.repair_v8.supervise import stage
from .prepare import load, PROJECT


def utc():
    return datetime.now(timezone.utc).isoformat()


def archive_evidence(out, control, prepared):
    archive = control / 'evidence.tgz'
    paths = []
    for base in [out, prepared, PROJECT / 'research/repair_v8/pilot']:
        for p in sorted(base.rglob('*')):
            parts = set(p.relative_to(base).parts)
            if parts & {'checkout', 'work', 'snapshots', 'tokenizer', '__pycache__', '.pytest_cache'}:
                continue
            if p.is_file() and not p.is_symlink() and p.name != 'evidence.tgz':
                paths.append(p)
    files = {str(p.relative_to(PROJECT)): p for p in paths}
    inventory = [{'path': name, 'sha256': sha(p), 'bytes': p.stat().st_size} for name, p in sorted(files.items())]
    write_json(control / 'archive-inventory.json', inventory)
    files[str((control / 'archive-inventory.json').relative_to(PROJECT))] = control / 'archive-inventory.json'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, p in sorted(files.items()):
            tar.add(p, arcname=name, recursive=False)
    import hashlib
    with tarfile.open(archive, 'r:gz') as tar:
        if {m.name for m in tar.getmembers()} != set(files):
            raise RuntimeError('Archive member mismatch')
        for row in inventory:
            if hashlib.sha256(tar.extractfile(row['path']).read()).hexdigest() != row['sha256']:
                raise RuntimeError('Archive verification failed')
    receipt = {'path': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
               'readback_verified': True, 'omitted': 'Reconstructible checkouts, immutable packaged input snapshots/tokenizer, sandbox caches and raw tensor scratch. Original logs, parsed measurements, prompts, outputs, sources and decisions retained.'}
    write_json(control / 'archive.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'out', 'inference-python', 'model']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--power-off', action='store_true', required=True)
    a = p.parse_args()
    prepared, out = a.prepared.resolve(), a.out.resolve()
    if os.geteuid() != 0 or PROJECT.parent != Path('/root/autodl-tmp') or not Path('/usr/bin/shutdown').is_file():
        raise RuntimeError('Power-off controller is restricted to a dedicated AutoDL data-disk deployment')
    if not out.is_relative_to(PROJECT) or not prepared.is_relative_to(PROJECT):
        raise RuntimeError('Preparation and output must be inside this deployment')
    _, cfg = load(prepared)
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 90 * (1 << 30):
        raise RuntimeError('Need the prepared 96 GiB-class GPU; no work or shutdown requested')
    lock = Path('/root/autodl-tmp/repair-v8-pilot.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'repairs/REPORT.json').exists():
        raise RuntimeError('This pilot is already complete; refuse a silent reroll')
    control = out / 'controllers' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    control.mkdir(parents=True, exist_ok=False)
    write_json(out / 'ACTIVE.json', {'controller': str(control), 'pid': os.getpid(), 'utc': utc()})
    write_json(control / 'authorization.json', {'basis': 'User authorized this development preparation and autonomous shutdown on completion/blockage',
               'power_off': True, 'scope': cfg['scope'], 'wall_seconds': cfg['total_wall_seconds'], 'training_enabled': False})
    env = os.environ.copy(); env.update(PYTHONPATH=str(PROJECT), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    started = time.monotonic(); error = None
    def execute(name, args, maximum):
        load(prepared)
        remaining = cfg['total_wall_seconds'] - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError('Pilot total wall budget exhausted')
        stage(name, [str(Path(os.sys.executable).absolute()), '-u', '-m', *args], env, control, min(maximum, remaining))
    try:
        inventory = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30)
        write_json(control / 'environment.json', {'utc': utc(), 'nvidia_smi': inventory.stdout,
                   'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
                   'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
        execute('runtime-preflight', ['research.repair_v8.pilot.runtime', '--prepared', str(prepared),
                '--inference-python', str(a.inference_python), '--model', str(a.model), '--out', str(control / 'runtime.json')], 1500)
        controls = out / 'controls'
        if not (controls / 'receipt.json').exists():
            if controls.exists():
                controls.rename(out / ('controls.interrupted-' + str(time.time_ns())))
            execute('controls', ['research.repair_v8.pilot.controls', '--prepared', str(prepared), '--out', str(controls)], 10800)
        else:
            # Re-run isolation after a reboot; saved model draws remain untouched.
            from .controls import isolation_controls
            isolation_controls(prepared, control / 'resume-isolation')
        execute('repairs', ['research.repair_v8.pilot.run', '--prepared', str(prepared), '--controls', str(controls / 'receipt.json'),
                '--out', str(out / 'repairs'), '--inference-python', str(a.inference_python), '--model', str(a.model)], cfg['total_wall_seconds'])
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    outcome = {'status': 'blocked' if error else 'complete', 'error': error, 'utc': utc(),
               'scope': cfg['scope'], 'seconds': time.monotonic() - started}
    write_json(control / 'outcome.json', outcome)
    write_json(out / 'outcome.json', outcome)
    (control / 'REPORT.md').write_text('# Three-task development pilot controller\n\n' +
        (error or 'Complete; see repairs/REPORT.md.') + '\n\nNo formal or training claim.\n')
    archive = None
    try:
        archive = archive_evidence(out, control, prepared)
    except Exception as exc:
        write_json(control / 'archive-error.json', {'error': str(exc), 'original_evidence_retained': True})
    write_json(control / 'progress.json', {'stage': 'awaiting_verified_backup', 'utc': utc()})
    acknowledged = False
    deadline = time.monotonic() + 300
    ack = control / 'backup-local-ack.json'
    while archive and time.monotonic() < deadline:
        if ack.exists():
            try:
                acknowledged = json.loads(ack.read_text()).get('sha256') == archive['sha256']
            except (OSError, ValueError):
                pass
            if acknowledged:
                break
        time.sleep(5)
    write_json(control / 'shutdown-request.json', {'utc': utc(), 'backup_ack': acknowledged,
               'reason': error or 'pilot complete', 'data_disk_evidence_retained': True})
    response = subprocess.run(['/bin/bash', '/usr/bin/shutdown'], capture_output=True, text=True, timeout=30)
    write_json(control / 'shutdown-return.json', {'returncode': response.returncode, 'stdout': response.stdout, 'stderr': response.stderr})


if __name__ == '__main__':
    main()
