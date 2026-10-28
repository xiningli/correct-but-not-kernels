"""D1/D2 operations: local archive verification, status, and poweroff receipt."""
import json
from pathlib import Path
import shlex
import subprocess
import time

from research.diagnostics_v1.common import ROOT, digest, save, utc

HOST = 'cuda-gemma4-westb-replication'
REMOTE = '/root/autodl-tmp/cuda-gemma4-llm'
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '-o', 'ControlPath=none',
           '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
OUT = ROOT / 'runs/diagnostics-v5/recovery'
PROBE = '''import json
from pathlib import Path
root=Path('/root/autodl-tmp/cuda-gemma4-llm')
active=json.loads((root/'runs/diagnostics-v5/active.json').read_text())
d=Path(active['path']); active['records']={}
for name in ['progress.json','child.json','outcome.json','archive.json','previous-handoff.json',
             'shutdown-request.json','shutdown-return.json','shutdown-deferred.json',
             'controls/checks.json','controls/receipt.json','repairs/progress.json',
             'repairs/outcome.json','repairs/REPORT.json']:
    if (d/name).exists(): active['records'][name]=json.loads((d/name).read_text())
p=Path('/proc')/str(active['pid'])/'stat'
active['controller_process_state']=p.read_text().split()[2] if p.exists() else None
print(json.dumps(active))
'''


def remote(code):
    result = subprocess.run(['ssh', *OPTIONS, HOST, REMOTE + '/.venv/bin/python -c ' + shlex.quote(code)],
                            capture_output=True, text=True, timeout=45, check=True)
    return json.loads(result.stdout)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    acknowledged, failures, retired = None, 0, False
    while True:
        try:
            state = remote(PROBE); state['observed_utc'] = utc()
            save(OUT / 'latest.json', state)
            records = state['records']
            stage = records.get('progress.json', {}).get('stage')
            status = {'stage': stage, 'remote_controller': state['path'], 'observed_utc': state['observed_utc'],
                      'training': False, 'formal_test': False, 'progress': records.get('repairs/progress.json'),
                      'outcome': records.get('outcome.json'), 'process_state': state['controller_process_state']}
            save(ROOT / 'runs/DIAGNOSTICS_RUN.json', status)
            if 'previous-handoff.json' in records:
                if not retired:
                    subprocess.run(['systemctl', '--user', 'stop', 'cuda-gemma4-capability-v4.service'],
                                   check=True, timeout=30)
                    retired = True
                save(ROOT / 'runs/ACTIVE_RUN.json', status)
            archived = records.get('archive.json')
            if archived:
                directory = OUT / Path(state['path']).name; directory.mkdir(exist_ok=True)
                final, part = directory / 'evidence.tgz', directory / 'evidence.tgz.part'
                if acknowledged != state['path']:
                    if not final.exists():
                        subprocess.run(['sftp', *OPTIONS, '-b', '-', HOST],
                            input='reget ' + archived['path'] + ' ' + str(part) + '\n', text=True,
                            timeout=1800, check=True)
                        if part.stat().st_size != archived['bytes'] or digest(part) != archived['sha256']:
                            part.rename(directory / ('mismatch-' + str(time.time_ns())))
                            raise ValueError('Archive hash/size mismatch; no acknowledgment')
                        part.replace(final)
                    if final.stat().st_size != archived['bytes'] or digest(final) != archived['sha256']:
                        raise ValueError('Local archive changed')
                    ack = {'sha256': archived['sha256'], 'utc': utc(), 'local': str(final)}
                    save(directory / 'verified.json', {**archived, 'verified_utc': utc()})
                    code = ('import json\nfrom pathlib import Path\np=Path('
                            + repr(state['path'] + '/backup-local-ack.json') + ')\np.write_text('
                            + repr(json.dumps(ack) + '\n') + ')\nprint("{}")')
                    remote(code); save(directory / 'ack.json', ack)
                    acknowledged = state['path']
                    print(json.dumps({'verified_backup': str(final), 'utc': utc()}), flush=True)
                save(directory / 'terminal-live-status.json', state)
                if 'shutdown-return.json' in records or 'shutdown-deferred.json' in records:
                    return
            failures = 0
        except Exception as error:
            save(OUT / 'last-error.json', {'utc': utc(), 'error': f'{type(error).__name__}: {error}'})
            if acknowledged:
                failures += 1
                if failures >= 20:
                    save(OUT / Path(acknowledged).name / 'shutdown-unverified.json', {
                        'utc': utc(), 'platform_power_and_billing_verified': False})
                    return
        time.sleep(15 if acknowledged else 45)


if __name__ == '__main__':
    main()
