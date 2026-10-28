"""D1/D2: serialized v4 handoff, controls, repairs, verified-backup shutdown."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time

from research.diagnostics_v1.common import PLAN, ROOT, digest, save, source_seal, utc

BASE = ROOT / 'runs/diagnostics-v5'


def alive(pid):
    path = Path('/proc') / str(pid) / 'stat'
    return path.exists() and path.read_text().split()[2] not in {'Z', 'X'}


def handoff_evidence(previous, pid):
    """No GPU acquisition while the predecessor lives or its backup is unverified."""
    previous = Path(previous)
    if alive(pid):
        return None
    if not (previous / 'shutdown-deferred.json').exists():
        raise RuntimeError('Predecessor disappeared without a handoff receipt')
    if any(not (previous / n).exists() for n in ['archive.json', 'backup-local-ack.json', 'outcome.json']):
        raise RuntimeError('Predecessor ended without verified backup')
    archived = json.loads((previous / 'archive.json').read_text())
    acknowledged = json.loads((previous / 'backup-local-ack.json').read_text())
    archive = Path(archived['path'])
    if (archive.resolve() != (previous / 'evidence.tgz').resolve()
            or archived['sha256'] != acknowledged['sha256']
            or archive.stat().st_size != archived['bytes'] or digest(archive) != archived['sha256']):
        raise ValueError('Predecessor backup identity mismatch')
    return {'utc': utc(), 'previous': str(previous), 'archive': archived, 'local_ack': acknowledged,
            'shutdown_deferred': json.loads((previous / 'shutdown-deferred.json').read_text()),
            'outcome': json.loads((previous / 'outcome.json').read_text())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--previous-pid', type=int, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    a = p.parse_args()
    previous = a.previous.resolve()
    expected = json.loads((ROOT / 'runs/reliability-v2/capability-active.json').read_text())
    if expected['path'] != str(previous) or expected['pid'] != a.previous_pid or not alive(a.previous_pid):
        raise ValueError('Cannot bind the exact live predecessor')
    directory = BASE / 'controllers' / utc().replace(':', '').replace('+', '_')
    directory.mkdir(parents=True, exist_ok=False)
    save(BASE / 'active.json', {'path': str(directory), 'pid': os.getpid(), 'utc': utc(),
                              'previous': str(previous)})
    save(directory / 'source-seal.json', source_seal())
    marker = previous / 'defer-shutdown'
    if marker.exists():
        raise ValueError('A predecessor shutdown handoff already exists')
    # Only the predecessor poweroff is deferred, never its evaluation/report/backup.
    marker.write_text(json.dumps({'successor': str(directory), 'pid': os.getpid(), 'utc': utc()}) + '\n')
    handoff, isolation, error = False, None, None

    def execute(name, args, timeout):
        command = ['/bin/bash', str(ROOT / 'scripts/gpu_env.sh'), str(ROOT / '.venv/bin/python'), *args]
        save(directory / 'progress.json', {'stage': name, 'utc': utc()})
        with (directory / (name + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(directory / 'child.json', {'pid': child.pid, 'stage': name, 'command': command, 'utc': utc()})
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise RuntimeError(name + ' exceeded wall budget')
            if code:
                raise RuntimeError(f'{name} exited {code}; see {name}.log')

    try:
        save(directory / 'progress.json', {'stage': 'waiting_for_previous_verified_backup', 'utc': utc()})
        deadline = time.monotonic() + 48 * 3600
        while time.monotonic() < deadline:
            evidence = handoff_evidence(previous, a.previous_pid)
            if evidence is not None:
                handoff = True
                save(directory / 'previous-handoff.json', evidence)
                if evidence['outcome']['status'] != 'complete':
                    raise RuntimeError('Predecessor stopped; diagnose before new GPU work')
                break
            time.sleep(15)
        else:
            raise RuntimeError('Predecessor handoff exceeded 48 hours')
        if json.loads((directory / 'source-seal.json').read_text()) != source_seal():
            raise ValueError('Queued source changed before execution')
        prior = set((ROOT / 'runs/reliability-v2/strict-isolation').glob('*/receipt.json'))
        execute('isolation', ['-m', 'research.reliability_v2.isolation_controls'], 900)
        created = set((ROOT / 'runs/reliability-v2/strict-isolation').glob('*/receipt.json')) - prior
        if len(created) != 1:
            raise RuntimeError('Cannot bind exactly one fresh isolation receipt')
        isolation, = created
        execute('controls', ['-m', 'research.diagnostics_v1.controls', '--prepared', str(a.prepared),
                '--isolation', str(isolation), '--out', str(directory / 'controls')], 3600)
        execute('repairs', ['-m', 'research.diagnostics_v1.run', '--prepared', str(a.prepared),
                '--controls', str(directory / 'controls'), '--out', str(directory / 'repairs')], 86400)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    save(directory / 'outcome.json', {'status': 'stopped' if error else 'complete', 'error': error,
                                     'handoff': handoff, 'utc': utc()})
    (directory / 'REPORT.md').write_text('# Diagnostic repair development controller\n\n'
        + (error if error else 'All development phases completed.')
        + '\n\nSee repairs/REPORT.md. Two reduced development defects do not establish generalization.\n')
    roots = [directory, a.prepared, *[ROOT / n for n in source_seal()]]
    if isolation:
        roots.append(isolation.parent)
    archive = directory / 'evidence.tgz'; added = set()
    with tarfile.open(archive, 'w:gz') as tar:
        for root in roots:
            for path in ([root] if root.is_file() else sorted(root.rglob('*'))):
                name = str(path.relative_to(ROOT))
                if (path.is_file() and not path.is_symlink() and path != archive and name not in added
                        and not {'work', 'staged-inputs', '__pycache__'} & set(path.relative_to(ROOT).parts)):
                    tar.add(path, arcname=name, recursive=False); added.add(name)
    archived = {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive)}
    save(directory / 'archive.json', archived)
    save(directory / 'progress.json', {'stage': 'awaiting_verified_backup', 'utc': utc()})
    deadline = time.monotonic() + 3600
    ack = directory / 'backup-local-ack.json'
    while time.monotonic() < deadline:
        if ack.exists() and json.loads(ack.read_text()).get('sha256') == archived['sha256']:
            break
        time.sleep(10)
    if alive(a.previous_pid):
        # Failed queue setup must not terminate the still-authorized prior run.
        if marker.exists() and json.loads(marker.read_text()).get('successor') == str(directory):
            marker.unlink()
        save(directory / 'shutdown-deferred.json', {'utc': utc(), 'reason': 'predecessor_still_running'})
        return
    save(directory / 'shutdown-request.json', {'utc': utc(), 'reason': error or 'diagnostic pilot complete',
         'backup_ack': ack.exists(), 'unverified_backup_timeout': not ack.exists()})
    result = subprocess.run(['/bin/bash', '/usr/bin/shutdown'], capture_output=True, text=True, timeout=60)
    save(directory / 'shutdown-return.json', {'returncode': result.returncode, 'stdout': result.stdout,
                                            'stderr': result.stderr, 'utc': utc()})


if __name__ == '__main__':
    main()
