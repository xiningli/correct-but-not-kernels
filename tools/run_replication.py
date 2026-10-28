"""K1–K4: bounded AutoDL supervisor; preserve every phase, then honor authorized shutdown."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.autonomous_pilot import progress_stamp,stop_process
from tools.gatelib import ROOT,init_run,timestamp,write_json
from tools.replication import require_replication_gate,sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--stage',choices=['audit','reevaluate','confirm','all'],required=True)
    p.add_argument('--power-off',action='store_true',required=True)
    a=p.parse_args()
    if os.geteuid()!=0 or ROOT.parent!=Path('/root/autodl-tmp'):
        raise RuntimeError('Supervisor is restricted to the prepared AutoDL data disk')
    a.study=a.study.resolve();a.study.relative_to(ROOT/'runs')
    plan,_=require_replication_gate('R0',a.study)
    # Detect no-card boots without importing torch into their 2 GiB cgroup.
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],
                       capture_output=True,text=True,timeout=15)
    if gpu.returncode or not gpu.stdout.strip():raise RuntimeError('No GPU attached; preparation only')
    if int(gpu.stdout.strip().splitlines()[0])<80*1024:raise RuntimeError('At least 80 GiB VRAM required')
    controller=a.study/'controllers'/timestamp()
    init_run(controller,{'stage':a.stage,'automatic_shutdown':True,'protocol':sha256(a.study/'frozen.json')},plan['analysis']['seed'])
    proc=None;error=None;archive_error=None;report_error=None
    try:
        with (controller/'worker.log').open('x') as f:
            proc=subprocess.Popen([str(ROOT/'scripts/gpu_env.sh'),sys.executable,'-m','tools.replication_worker',
                '--study',str(a.study),'--stage',a.stage],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            started=last_progress=time.time()
            while proc.poll() is None:
                last_progress=max(last_progress,progress_stamp(a.study))
                if time.time()-last_progress>plan['no_progress_seconds']:
                    raise RuntimeError('No progress for the declared watchdog interval')
                if time.time()-started>plan['stage_wall_seconds']:
                    raise RuntimeError('Stage wall budget exceeded')
                time.sleep(10)
            if proc.returncode:raise RuntimeError(f'Worker exited {proc.returncode}')
    except Exception as e:
        error=f'{type(e).__name__}: {e}'
        if proc and proc.poll() is None:stop_process(proc)
    finally:
        write_json(controller/'outcome.json',{'status':'blocked' if error else 'complete','error':error,'stage':a.stage})
        try:
            from eval.terminal_report import generate
            generate(a.study,controller)
        except Exception as e:
            report_error=f'{type(e).__name__}: {e}'
            write_json(controller/'report-error.json',{'error':report_error})
            if not (controller/'TERMINAL_REPORT.md').exists():
                (controller/'TERMINAL_REPORT.md').write_text(
                    '# 阶段运行报告\n\n报告生成失败：'+report_error+'\n\n'
                    '请保留 outcome.json、worker.log 和各 gate 记录；不能声称实验报告已完成。\n')
        try:
            archive=controller/'evidence-and-checkpoints.tgz'
            partial=controller/'evidence-and-checkpoints.tgz.part'
            subprocess.run(['tar','--exclude=*.tgz','--exclude=*.part','--exclude=*/work/*',
                '--exclude=*/private-inputs.pt','-I','gzip -1','-cf',str(partial),
                str(a.study.relative_to(ROOT)),'runs/gates','evidence/gates','LEDGER.md','REPORT_STATUS.md'],
                cwd=ROOT,check=True,timeout=7200)
            partial.rename(archive)
            write_json(controller/'archive.json',{'path':str(archive),'sha256':sha256(archive),'bytes':archive.stat().st_size})
        except Exception as e:archive_error=f'{type(e).__name__}: {e}'
        write_json(controller/'shutdown-request.json',{'reason':error or report_error or 'complete',
            'report_error':report_error,'preservation_error':archive_error,
            'command':'/bin/bash /usr/bin/shutdown','requested_at':timestamp()})
        os.sync()
        start=time.monotonic()
        while not (controller/'backup-local-ack').exists():
            limit=1800 if (controller/'backup-local-started').exists() else 180
            if time.monotonic()-start>=limit:break
            time.sleep(5)
        with (controller/'shutdown.log').open('x') as f:
            result=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdout=f,stderr=subprocess.STDOUT,timeout=60)
        write_json(controller/'shutdown-return.json',{'returncode':result.returncode});os.sync()

if __name__=='__main__':main()
