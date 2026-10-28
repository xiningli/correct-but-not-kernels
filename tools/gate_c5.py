"""K1–K4: three GRPO steps, an explicit restart, and checkpoint re-evaluation."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import yaml
from env.reward import score_completion
from env.tasks import verified_catalog
from tools.gatelib import ROOT, enforce_failure_stop, finish_gate, init_run, require_gate, timestamp, write_json

def job(arguments, log, timeout):
    """K1–K3: a dead training/generation subprocess cannot leave a GPU server."""
    with log.open('x') as stream:
        proc=subprocess.Popen([str(ROOT/'scripts/gpu_env.sh'),sys.executable,*arguments],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            result=proc.wait(timeout=timeout)
            if result:
                raise RuntimeError(f'Process exited {result}; see {log.relative_to(ROOT)}')
        finally:
            import psutil
            try:
                descendants=psutil.Process(proc.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                descendants=[]
            for child in reversed(descendants):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            try:
                os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

def configuration(selection, run_id):
    """K1–K3: declare all thresholds before the first optimizer step."""
    return dict(hypotheses=['K1','K2','K3','K4'],enabled=True,stage='C5',
        model=selection['model'],tasks=selection['tasks'],seed=20261005,
        run_id=run_id,reward_arm='C',
        lora=dict(rank=16,alpha=32),
        grpo=dict(group_size=8,micro_batch=1,max_steps=3,save_steps=1,
                  max_completion_length=4096,learning_rate=1e-5,beta=.01,
                  importance_sampling_mode='token_truncate',importance_sampling_clip_max=3.),
        vllm=dict(gpu_memory_utilization=.4,max_model_len=8192),
        eval_gpu_memory_utilization=.85,
        kill=dict(saturation_pass_rate=.95,saturation_steps=5,
            reward_zero_std_fraction=.95,reward_mean_for_zero_std=.95,zero_std_steps=5,
            kl_max=.2,median_length_floor=32,truncation_fraction_max=.5,
            sandbox_violations_max=0,detector_fire_fraction_max=0.,max_wall_seconds=10800,
            importance_weight_mean_min=.1))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator-continue',help='K1–K4: recorded authorization to fix and retry after two failures')
    a=p.parse_args()
    try: enforce_failure_stop('C5')
    except RuntimeError:
        if not a.operator_continue:raise
        print(f'Operator-authorized continuation: {a.operator_continue}',flush=True)
    c4=require_gate('C4')
    previous=json.loads((ROOT/c4).read_text())
    selection=json.loads((ROOT/previous['artifacts']/'selection.json').read_text())
    if not selection['model']:
        raise RuntimeError('C4 did not select a model')
    stamp=timestamp(); directory=ROOT/'runs/c5'/stamp
    cfg=configuration(selection,f'c5/{stamp}/training')
    init_run(directory,cfg,cfg['seed'])
    config_path=directory/'config.yaml'
    checks=[]
    try:
        import pynvml
        pynvml.nvmlInit(); gpu=pynvml.nvmlDeviceGetHandleByIndex(0)
        baseline=pynvml.nvmlDeviceGetMemoryInfo(gpu).used
        job(['-m','train.launch',str(config_path),'--max-steps','1'],directory/'step-1.log',5400)
        train=ROOT/'runs'/cfg['run_id']
        first=json.loads((train/'completion-step-1.json').read_text())
        checkpoint=train/'checkpoints/checkpoint-1'
        if not (checkpoint/'optimizer.pt').is_file() or not (checkpoint/'trainer_state.json').is_file():
            raise RuntimeError('Optimizer/trainer state missing from checkpoint')
        job(['-m','train.launch',str(config_path),'--resume','--max-steps','3'],directory/'resume.log',5400)
        resumed=json.loads((train/'resume-checkpoint-1.json').read_text())
        last=json.loads((train/'completion-step-3.json').read_text())
        metrics=[json.loads(x) for x in (train/'metrics.jsonl').read_text().splitlines()]
        steps={x['step'] for x in metrics if 'loss' in x}
        stable=all(x.get('env/gpu_delta_bytes',0)<=64<<20 for x in metrics)
        after=pynvml.nvmlDeviceGetMemoryInfo(gpu).used
        checks.append(dict(name='three_resumable_steps',status='PASS' if
            first['lora_hash']==resumed['lora_hash'] and {1,2,3}<=steps and stable
            and after-baseline<=64<<20 and not (train/'kill.json').exists() else 'FAIL',
            steps=sorted(steps),first=first,resumed=resumed,final=last,
            gpu_baseline_bytes=baseline,gpu_after_bytes=after))
        request={**selection['config'],'model':cfg['model'],'adapter':last['adapter'],
                 'samples_per_task':4,'seed':cfg['seed']+50000}
        write_json(directory/'request.json',request)
        job(['-m','tools.sample_policy','--request',str(directory/'request.json'),
             '--out',str(directory/'generations.jsonl')],directory/'reeval.log',3600)
        tasks={t.task_id:t for t in verified_catalog()}
        import torch
        torch.set_num_threads(8)
        rows=[json.loads(x) for x in (directory/'generations.jsonl').read_text().splitlines()]
        outcomes=[]
        for i,row in enumerate(rows):
            result=score_completion(row['text'],tasks[row['task']],directory/f'reeval-{i:03d}',
                                    seed=cfg['seed']+90000+i,trials=4)
            outcomes.append(asdict(result))
            with (directory/'scores.jsonl').open('a') as f:
                f.write(json.dumps({**row,'result':asdict(result)})+'\n')
        checks.append(dict(name='checkpoint_reload_reevaluation',status='PASS' if
            len(rows)==4*len(cfg['tasks']) and all(x['component']!='harness_error' for x in outcomes)
            else 'FAIL',outcomes=outcomes))
    except Exception as error:
        checks.append(dict(name='blocking_failure',status='FAIL',error=f'{type(error).__name__}: {error}'))
        print(checks[-1]['error'],flush=True)
    raise SystemExit(finish_gate('C5',checks,prerequisite=c4,artifacts=str(directory.relative_to(ROOT)),
                                operator_continue=a.operator_continue))

if __name__ == "__main__":
    main()
