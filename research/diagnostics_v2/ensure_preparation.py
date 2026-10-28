"""Resume only interrupted CPU staging; failed experiments are never rerolled."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from research.diagnostics_v2.common import BASE, ROOT, save, source_seal, utc


def running(module):
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().split(b'\0')
            if module.encode() in args and (p.parent/'stat').read_text().split()[2] not in {'Z','X'}:
                return True
        except (OSError,ValueError):pass
    return False


def needed(download,stage,download_alive,stage_alive):
    if download.get('status')=='stopped' or stage.get('stage')=='stopped':
        return []
    result=[]
    if download.get('status')!='complete' and not download_alive:result.append('download')
    if stage.get('stage')!='ready_for_GPU' and not stage_alive:result.append('stage')
    return result


def main():
    d=BASE/'preparation';d.mkdir(parents=True,exist_ok=True)
    with (d/'resume.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cpu=json.loads((d/'cpu-tests.json').read_text())
        if cpu['status']!='PASS' or cpu['source_sha256']!=source_seal():
            raise ValueError('Current-source CPU validation required before automatic preparation')
        if (BASE/'active.json').exists():
            print('{}');return
        def record(name):
            p=d/name;return json.loads(p.read_text()) if p.exists() else {}
        actions=needed(record('download-progress.json'),record('stage-progress.json'),
                       running('research.diagnostics_v2.download'),running('research.diagnostics_v2.stage'))
        launched=[]
        for name in actions:
            args=[sys.executable,'-m','research.diagnostics_v2.'+name]
            if name=='stage':args+=['--out',str(BASE/'prepared-20261016')]
            with (d/(name+'-resume.log')).open('a') as log:
                child=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                       stderr=subprocess.STDOUT,start_new_session=True)
            item={'utc':utc(),'module':name,'pid':child.pid};launched.append(item)
            save(d/(name+'-last-launch.json'),item)
        print(json.dumps({'launched':launched}))


if __name__=='__main__':main()
