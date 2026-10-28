"""Local preparation/GPU watcher, one-time authorized launch, archive acknowledgment."""
import json
from pathlib import Path
import shlex
import subprocess
import time
from research.diagnostics_v2.common import BASE, ROOT, save, digest, utc

HOST='cuda-gemma4-westb-replication'
REMOTE='/root/autodl-tmp/cuda-gemma4-llm'
OPTIONS=['-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ControlPath=none',
         '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
PROBE='''import json,os,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/cuda-gemma4-llm');b=root/'runs/diagnostics-v6'
o={'preparation':{}}
for n in ['download-progress.json','download-receipt.json','stage-progress.json','cpu-tests.json']:
 p=b/'preparation'/n
 if p.exists():
  x=json.loads(p.read_text());o['preparation'][n]={k:v for k,v in x.items() if k not in ['files','structure','source_sha256']}
o['gpu']=False
try:
 p=subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=12)
 o['gpu']=p.returncode==0 and any(int(x.strip())>=90000 for x in p.stdout.splitlines() if x.strip().isdigit())
except OSError:pass
p=b/'active.json'
if p.exists():
 a=json.loads(p.read_text());d=Path(a['path']);a['records']={}
 for n in ['progress.json','outcome.json','archive.json','shutdown-request.json','shutdown-return.json','repairs/progress.json','repairs/outcome.json','repairs/REPORT.json']:
  if (d/n).exists():a['records'][n]=json.loads((d/n).read_text())
 stat=Path('/proc')/str(a['pid'])/'stat';a['process_state']=stat.read_text().split()[2] if stat.exists() else None
 o['active']=a
print(json.dumps(o))
'''


def remote(code,timeout=50):
    command='cd '+shlex.quote(REMOTE)+' && '+shlex.quote(REMOTE+'/.venv/bin/python')+' -c '+shlex.quote(code)
    r=subprocess.run(['ssh',*OPTIONS,HOST,command],
                     capture_output=True,text=True,timeout=timeout,check=True)
    return json.loads(r.stdout)


def launch(prepared):
    code='''import json,os,subprocess
from pathlib import Path
r=Path('/root/autodl-tmp/cuda-gemma4-llm');b=r/'runs/diagnostics-v6'
assert not (b/'active.json').exists()
p=b/'preparation/auto-launch.json'
with p.open('x') as f:
 with (b/'preparation/auto-launch.log').open('x') as log:
  child=subprocess.Popen(['/bin/bash',str(r/'scripts/gpu_env.sh'),str(r/'.venv/bin/python'),'-m','research.diagnostics_v2.supervise','--prepared',PREPARED],cwd=r,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 rec={'pid':child.pid,'prepared':PREPARED};json.dump(rec,f)
print(json.dumps(rec))
'''.replace('PREPARED',repr(prepared))
    return remote(code)


def main():
    out=BASE/'recovery';out.mkdir(parents=True,exist_ok=True)
    launched=(out/'auto-launch.json').exists();acknowledged=None;failures=0
    while True:
        try:
            s=remote(PROBE);s['observed_utc']=utc();save(out/'latest.json',s)
            prep=s['preparation'];stage=prep.get('stage-progress.json',{});active=s.get('active')
            status={'stage':stage.get('stage','preparing'),'observed_utc':s['observed_utc'],
                    'GPU_available':s['gpu'],'training':False,'formal_test':False,'preparation':prep}
            if active:
                records=active['records'];status.update(stage=records.get('progress.json',{}).get('stage'),
                    remote_controller=active['path'],progress=records.get('repairs/progress.json'),
                    outcome=records.get('outcome.json'),process_state=active['process_state'])
                archive=records.get('archive.json')
                if archive:
                    local=out/Path(active['path']).name;local.mkdir(exist_ok=True)
                    final=local/'evidence.tgz';part=local/'evidence.tgz.part'
                    if acknowledged!=active['path']:
                        if not final.exists():
                            subprocess.run(['sftp',*OPTIONS,'-b','-',HOST],input='reget '+archive['path']+' '+str(part)+'\n',text=True,timeout=1800,check=True)
                            if part.stat().st_size!=archive['bytes'] or digest(part)!=archive['sha256']:
                                part.rename(local/('mismatch-'+str(time.time_ns())));raise ValueError('Archive mismatch; no ack')
                            part.replace(final)
                        if final.stat().st_size!=archive['bytes'] or digest(final)!=archive['sha256']:
                            raise ValueError('Local archive identity changed')
                        ack={'sha256':archive['sha256'],'utc':utc(),'local':str(final)}
                        remote('from pathlib import Path\np=Path('+repr(active['path']+'/backup-local-ack.json')+')\np.write_text('+repr(json.dumps(ack)+'\n')+')\nprint("{}")')
                        save(local/'verified.json',{**archive,'verified_utc':utc()});save(local/'ack.json',ack)
                        acknowledged=active['path']
                    save(local/'terminal-live-status.json',s)
                    status.update(stage='complete_backup_verified' if records.get('outcome.json',{}).get('status')=='complete' else 'stopped_backup_verified',local_archive_verified=True)
                    if 'shutdown-return.json' in records:
                        status['shutdown_return']=records['shutdown-return.json'];save(ROOT/'runs/ACTIVE_RUN.json',status);save(ROOT/'runs/ITERATIVE_REPAIR_RUN.json',status);return
            elif prep.get('cpu-tests.json',{}).get('status')=='PASS':
                remote('from research.diagnostics_v2.ensure_preparation import main\nmain()')
                if not launched and s['gpu'] and stage.get('stage')=='ready_for_GPU':
                    saved=launch(stage['prepared']);save(out/'auto-launch.json',saved);launched=True
            save(ROOT/'runs/ITERATIVE_REPAIR_RUN.json',status)
            if active:save(ROOT/'runs/ACTIVE_RUN.json',status)
            failures=0
        except Exception as error:
            save(out/'last-error.json',{'utc':utc(),'error':f'{type(error).__name__}: {error}'})
            if acknowledged:
                failures+=1
                if failures>=20:
                    save(out/Path(acknowledged).name/'shutdown-unverified.json',{'utc':utc(),'platform_power_and_billing_verified':False});return
        time.sleep(15 if acknowledged else 45)


if __name__=='__main__':main()
