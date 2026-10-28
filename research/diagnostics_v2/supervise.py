"""GPU-only controller: fresh gates, iterative experiment, verified backup, shutdown."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
from research.diagnostics_v2.common import BASE, ROOT, load_prepared, source_seal, save, digest, utc
from tools.replication import verify_frozen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    a=p.parse_args();a.prepared=a.prepared.resolve()
    # CPU-only preparation/boots must never enter GPU work or request shutdown.
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU is required; preparation remains ready')
    if torch.cuda.get_device_properties(0).total_memory < 90*(1<<30):
        raise RuntimeError('This BF16 31B plan requires the prepared 96 GiB-class GPU')
    verify_frozen(ROOT/'runs/replication-v1.1');plan,_,_=load_prepared(a.prepared)
    old=BASE/'active.json'
    if old.exists():
        previous=json.loads(old.read_text());d=Path(previous['path'])
        if not (d/'outcome.json').exists():raise RuntimeError('Unfinished v6 controller exists; inspect before restarting')
        raise RuntimeError('A v6 run already exists; do not silently reroll completed/stopped experiments')
    directory=BASE/'controllers'/utc().replace(':','').replace('+','_');directory.mkdir(parents=True)
    save(old,{'path':str(directory),'pid':os.getpid(),'utc':utc()})
    initial=source_seal();save(directory/'source-seal.json',initial)
    error=None;isolation=None
    def execute(stage,args,timeout):
        if source_seal()!=initial:raise ValueError('Source changed after launch')
        command=['/bin/bash',str(ROOT/'scripts/gpu_env.sh'),str(ROOT/'.venv/bin/python'),*args]
        save(directory/'progress.json',{'stage':stage,'utc':utc()})
        with (directory/(stage+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(directory/'child.json',{'stage':stage,'pid':child.pid,'command':command,'utc':utc()})
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError(stage+' exceeded wall budget')
            if code:raise RuntimeError(stage+' exited '+str(code))
    try:
        prior=set((ROOT/'runs/reliability-v2/strict-isolation').glob('*/receipt.json'))
        execute('isolation',['-m','research.reliability_v2.isolation_controls'],900)
        created=set((ROOT/'runs/reliability-v2/strict-isolation').glob('*/receipt.json'))-prior
        if len(created)!=1:raise RuntimeError('Need exactly one fresh isolation receipt')
        isolation,=created
        execute('controls',['-m','research.diagnostics_v1.controls','--prepared',str(ROOT/plan['controls']['prepared_v5']),
            '--isolation',str(isolation),'--out',str(directory/'controls')],3600)
        execute('feedback-controls',['-m','research.diagnostics_v2.controls','--prepared',str(a.prepared),
            '--controls',str(directory/'controls'),'--out',str(directory/'feedback-controls')],900)
        execute('repairs',['-m','research.diagnostics_v2.run','--prepared',str(a.prepared),
            '--controls',str(directory/'controls'),'--feedback-controls',str(directory/'feedback-controls'),
            '--out',str(directory/'repairs')],plan['limits']['run_wall_seconds']+1800)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    save(directory/'outcome.json',{'status':'stopped' if error else 'complete','error':error,'utc':utc()})
    (directory/'REPORT.md').write_text('# Iterative repair development controller\n\n'+(error or 'All phases complete.')
        +'\n\nSee repairs/REPORT.md. Two developer defects are not a generalization test.\n')
    save(directory/'progress.json',{'stage':'archiving','utc':utc()})
    roots=[directory,a.prepared,ROOT/plan['controls']['prepared_v5'],*[ROOT/n for n in initial]]
    if isolation:roots.append(isolation.parent)
    archive=directory/'evidence.tgz';added=set()
    with tarfile.open(archive,'w:gz') as tar:
        for root in roots:
            for path in ([root] if root.is_file() else sorted(root.rglob('*'))):
                name=str(path.relative_to(ROOT))
                if path.is_file() and not path.is_symlink() and path!=archive and name not in added and not {'work','staged-inputs','__pycache__'}&set(path.parts):
                    tar.add(path,arcname=name,recursive=False);added.add(name)
    info={'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive)};save(directory/'archive.json',info)
    save(directory/'progress.json',{'stage':'awaiting_verified_backup','utc':utc()})
    ack=directory/'backup-local-ack.json';verified=False;deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        if ack.exists() and json.loads(ack.read_text()).get('sha256')==info['sha256']:
            verified=True;break
        time.sleep(10)
    save(directory/'shutdown-request.json',{'utc':utc(),'reason':error or 'v6 development complete',
         'backup_ack':verified,'unverified_backup_timeout':not verified})
    response=subprocess.run(['/bin/bash','/usr/bin/shutdown'],capture_output=True,text=True,timeout=60)
    save(directory/'shutdown-return.json',{'utc':utc(),'returncode':response.returncode,'stdout':response.stdout,'stderr':response.stderr})


if __name__=='__main__':main()
