"""K1–K4: bounded C/S pilots, evaluation every 10 steps and final-policy bootstrap tables."""
import hashlib
import json
from pathlib import Path
import sys
import yaml
from env.tasks import split_ids
from tools.gate_c5 import configuration,job
from tools.gatelib import ROOT,enforce_failure_stop,finish_gate,init_run,require_gate,timestamp,write_json

def final_checkpoint(train):
    """K1–K4: recover the latest complete optimizer boundary, even after a crash."""
    policies=[]
    for path in train.glob('completion-step-*.json'):
        policies.append(json.loads(path.read_text()))
    for path in (train/'checkpoints').glob('checkpoint-*'):
        if (path/'trainer_state.json').is_file() and (path/'optimizer.pt').is_file():
            policies.append({'step':int(path.name.split('-')[-1]),'adapter':str(path)})
    if not policies:raise RuntimeError('No completed optimizer checkpoint')
    return max(policies,key=lambda x:x['step'])

def main():
    enforce_failure_stop('C6');c5=require_gate('C5');c4=require_gate('C4')
    gate=json.loads((ROOT/c4).read_text())
    selection=json.loads((ROOT/gate['artifacts']/'selection.json').read_text())
    stamp=timestamp();directory=ROOT/'runs/c6'/stamp
    _,heldout=split_ids()
    split_hash=hashlib.sha256((ROOT/'env/heldout.json').read_bytes()).hexdigest()
    plans={}
    for arm in ('C','S'):
        cfg=configuration(selection,f'c6/{stamp}/training-{arm}')
        cfg.update(stage='C6',seed=20261006,reward_arm=arm)
        cfg['checkpoint_keep']=2
        cfg['grpo'].update(max_steps=40,save_steps=1)
        # Reserve time for full final evaluation inside the per-arm 8h envelope.
        cfg['kill']['max_wall_seconds']=14400
        cfg.update(pilot_wall_seconds=28800,final_eval_reserve_seconds=10800)
        plans[arm]=cfg
    evaluation={**selection['config'],'model':selection['model'],
        'tasks':sorted(heldout),'samples_per_task':1,'fresh_checks':4,
        'seed':20261101,'split_sha256':split_hash}
    # Both policies' hyperparameters and the full evaluation grid are frozen
    # before any held-out outcome exists. No per-item result drives a decision.
    init_run(directory,{'hypotheses':['K1','K2','K3','K4'],
        'training':plans,'evaluation':evaluation,'max_arm_seconds':28800},20261006)
    for arm,cfg in plans.items():
        (directory/f'arm-{arm}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    checks=[];manifest={}
    def evaluate(label,adapter=None,timeout=10800):
        request={**evaluation,'adapter':adapter}
        request_path=directory/f'eval-{label}.json';write_json(request_path,request)
        out=directory/'heldout'/label
        job(['-m','eval.heldout','--request',str(request_path),'--out',str(out)],
            directory/f'eval-{label}.log',timeout)
        manifest[label]=str(out/'scores.jsonl')
    try:
        import time
        evaluate('base')
        for arm in ('C','S'):
            started=time.monotonic();error=None;reason=None;final=None;last_label=None
            train=ROOT/'runs'/plans[arm]['run_id']
            for boundary in range(10,plans[arm]['grpo']['max_steps']+1,10):
                remaining=28800-(time.monotonic()-started)
                if final and remaining<=plans[arm]['final_eval_reserve_seconds']:
                    reason={'reason':'wall_budget_reserve','step':final['step']}
                    break
                arguments=['-m','train.launch',str(directory/f'arm-{arm}.yaml'),'--max-steps',str(boundary)]
                if final:arguments.append('--resume')
                try:
                    job(arguments,directory/f'train-{arm}-to-{boundary}.log',
                        max(1,min(18000,remaining-plans[arm]['final_eval_reserve_seconds'])))
                except Exception as exc:
                    error=f'{type(exc).__name__}: {exc}'
                final=final_checkpoint(train)
                remaining=28800-(time.monotonic()-started)
                label=f'{arm}-step-{final["step"]}'
                if label!=last_label:
                    evaluate(label,final['adapter'],timeout=max(1,min(10800,remaining)))
                    last_label=label
                reason=json.loads((train/'kill.json').read_text()) if (train/'kill.json').exists() else None
                if reason or error:break
            write_json(directory/f'final-{arm}.json',final)
            # Final is already scored, including an early declared stop or the
            # last complete optimizer boundary after an infrastructure failure.
            manifest[arm]=manifest.pop(last_label)
            elapsed=time.monotonic()-started
            checks.append({'name':f'arm_{arm}','status':'PASS' if error is None and elapsed<=28800
                and (final['step']==plans[arm]['grpo']['max_steps'] or reason) else 'FAIL','final':final,
                'declared_stop':reason,'elapsed_seconds':elapsed,'error':error})
            if error: raise RuntimeError(f'Arm {arm} failed; final saved policy was evaluated: {error}')
        write_json(directory/'report-manifest.json',manifest)
        job(['-m','eval.report','--manifest',str(directory/'report-manifest.json'),
             '--out',str(directory/'REPORT.md')],directory/'report.log',300)
        (ROOT/'REPORT.md').write_bytes((directory/'REPORT.md').read_bytes())
        checks.append({'name':'complete_bootstrap_report','status':'PASS',
                       'manifest':str((directory/'report-manifest.json').relative_to(ROOT))})
    except Exception as error:
        checks.append({'name':'blocking_failure','status':'FAIL','error':f'{type(error).__name__}: {error}'})
        print(checks[-1]['error'],flush=True)
    write_json(directory/'available-evaluations.json',manifest)
    code=finish_gate('C6',checks,prerequisite=c5,artifacts=str(directory.relative_to(ROOT)))
    from eval.gates import render
    for path in (directory/'REPORT.md',ROOT/'REPORT.md'):
        if path.exists():
            old=path.read_text();begin='<!-- BEGIN GENERATED GATES -->';end='<!-- END GENERATED GATES -->'
            if begin in old and end in old:
                left,rest=old.split(begin,1);_,right=rest.split(end,1)
                path.write_text(left+render()+right.lstrip('\n'))
    raise SystemExit(code)

if __name__ == "__main__":
    main()
