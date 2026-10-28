"""K1–K4: supervise C6, preserve evidence, then honor the operator's AutoDL power-off request."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from tools.gatelib import ROOT,finish_gate,init_run,require_gate,timestamp,write_json

def progress_stamp(directory):
    names={'metrics.jsonl','scores.jsonl','generations.jsonl','trainer_state.json'}
    stamps=[p.stat().st_mtime for p in directory.rglob('*') if p.name in names and p.is_file()]
    return max(stamps,default=0)

def stop_process(proc):
    import psutil
    try: children=psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess: children=[]
    for child in reversed(children):
        try: child.kill()
        except psutil.NoSuchProcess: pass
    try: os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError: pass
    try: proc.wait(timeout=15)
    except subprocess.TimeoutExpired: pass  # platform shutdown remains reachable if a driver is stuck

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',required=True,help='K1–K4: recorded authorization for autonomous pilots and power-off')
    p.add_argument('--power-off',action='store_true',required=True,help='K1–K4: invoke the AutoDL platform shutdown after preservation')
    a=p.parse_args()
    if os.geteuid()!=0 or ROOT.parent!=Path('/root/autodl-tmp') or not Path('/usr/bin/shutdown').exists():
        raise RuntimeError('This supervisor is restricted to the prepared AutoDL project')
    c5=require_gate('C5')
    c5_artifacts=json.loads((ROOT/c5).read_text())['artifacts']
    directory=ROOT/'runs/autonomous'/timestamp()
    init_run(directory,{'hypotheses':['K1','K2','K3','K4'],'operator':a.operator,
        'power_off':True,'no_progress_seconds':1800,'wall_seconds':72000},20261006)
    proc=None;reason='blocked';error=None;source_run=None
    try:
        with (directory/'pilot.log').open('x') as log:
            proc=subprocess.Popen([str(ROOT/'scripts/gpu_env.sh'),sys.executable,'-m','tools.gate_c6'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.time();last_progress=started
            while proc.poll() is None:
                runs=sorted((ROOT/'runs/c6').glob('*'))
                if runs:
                    source_run=runs[-1]
                    last_progress=max(last_progress,progress_stamp(source_run))
                now=time.time()
                if now-last_progress>1800 or now-started>72000:
                    error='no_progress_watchdog' if now-last_progress>1800 else 'wall_watchdog'
                    stop_process(proc)
                    finish_gate('C6',[{'name':'supervisor_watchdog','status':'FAIL','error':error}],
                        artifacts=str(source_run.relative_to(ROOT)) if source_run else None)
                    break
                time.sleep(10)
            if proc.returncode==0: reason='complete'
            elif error is None: error=f'C6 exit status {proc.returncode}'
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        if proc and proc.poll() is None: stop_process(proc)
    finally:
        write_json(directory/'outcome.json',{'status':reason,'error':error,
                   'pilot':str(source_run) if source_run else None})
        status=ROOT/'REPORT_STATUS.md'
        text=status.read_text()
        terminal=(f'## Latest autonomous pilot outcome\n\nStatus: **{reason}**. '
                  f'Detail: `{error}`.\nRun: `{directory.relative_to(ROOT)}`. '
                  'Checkpoint/evidence preservation and the authorized platform shutdown follow this record.\n\n')
        status.write_text(text.replace('# Report status\n\n','# Report status\n\n'+terminal,1))
        # A complete remote archive survives power-off on the data disk; live
        # tensor scratch files are reproducible and excluded, checkpoints stay.
        archive=directory/'evidence-and-checkpoints.tgz'
        members=['runs/gates','evidence/gates','REPORT.md','REPORT_STATUS.md','LEDGER.md']
        members.append(c5_artifacts)
        if source_run: members.append(str(source_run.relative_to(ROOT)))
        backup_error=None
        try:
            subprocess.run(['tar','--exclude=*/work/*','--exclude=*/private-inputs.pt',
                '-czf',str(archive),*members],cwd=ROOT,check=True,timeout=1200)
            digest=hashlib.sha256()
            with archive.open('rb') as f:
                for block in iter(lambda:f.read(8<<20),b''): digest.update(block)
            write_json(directory/'archive.json',{'path':str(archive),'sha256':digest.hexdigest(),
                                                'bytes':archive.stat().st_size})
            subprocess.run([sys.executable,'-m','eval.gates','--write-status'],cwd=ROOT,check=True)
            records=sorted((ROOT/'evidence/gates').glob('c6-*.json'))
            if records:
                subprocess.run(['git','add',str(records[-1]),'REPORT.md','REPORT_STATUS.md'],cwd=ROOT,check=True)
                subprocess.run(['git','commit','-m','Record autonomous C6 pilot gate and report'],cwd=ROOT,check=True)
        except Exception as exc:
            backup_error=f'{type(exc).__name__}: {exc}'
        write_json(directory/'shutdown-request.json',{'requested_at':timestamp(),
            'reason':reason,'error':error,'preservation_error':backup_error,
            'command':'/bin/bash /usr/bin/shutdown','operator':a.operator})
        os.sync()
        # The workstation watcher can copy and hash-check the finished archive.
        # The data-disk archive remains available even if that workstation goes
        # away; a missing acknowledgement must not keep the rental GPU running.
        backup_wait_start=time.monotonic()
        while not (directory/'backup-local-ack').exists():
            limit=1800 if (directory/'backup-local-started').exists() else 180
            if time.monotonic()-backup_wait_start>=limit:break
            time.sleep(5)
        with (directory/'shutdown.log').open('x') as log:
            # AutoDL's installed command is shell text without a shebang;
            # direct execve would fail with ENOEXEC instead of powering off.
            result=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdout=log,stderr=subprocess.STDOUT,timeout=60)
        write_json(directory/'shutdown-return.json',{'returncode':result.returncode})
        os.sync()

if __name__=='__main__': main()
