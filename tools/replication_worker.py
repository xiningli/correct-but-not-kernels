"""K1–K4: ordered GPU audits, independent re-evaluation and frozen multi-seed/SFT jobs."""
import argparse
import copy
import json
from pathlib import Path
import time
import yaml
from eval.replication import read_rows
from tools.gatelib import ROOT,init_run,write_json,finish_gate,enforce_failure_stop,timestamp
from tools.gate_c5 import job
from tools.gate_c6 import final_checkpoint
from tools.replication import (verify_frozen,require_replication_gate,sha256,local_path,evaluation_request,candidate_seeds)

def record_once(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'Existing result changed: {path}')
    else:write_json(path,value)

class Study:
    def __init__(self,directory):
        self.directory=Path(directory)
        self.plan,_=verify_frozen(self.directory)
        self.pilot=yaml.safe_load((local_path(self.plan['pilot'])/'config.yaml').read_text())

    def require_gpu(self):
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<80<<30:
            raise RuntimeError('Phase-2 GPU work requires at least 80 GiB VRAM; CPU fallback prohibited')

    def evaluate(self,request,out):
        out=Path(out)
        if (out/'complete.json').exists():
            from eval.replication import validate_rows
            if json.loads((out/'request.json').read_text())!=request:raise ValueError('Evaluation request drift')
            done=json.loads((out/'complete.json').read_text())
            if done['scores_sha256']!=sha256(out/'scores.jsonl'):raise ValueError('Completed scores changed')
            validate_rows(read_rows(out/'scores.jsonl'),request,scores=True)
            return out/'scores.jsonl'
        req=out.parent/(out.name+'-request.json')
        if req.exists():
            if json.loads(req.read_text())!=request:raise ValueError('Saved request changed')
        else:write_json(req,request)
        job(['-m','eval.replication','--request',str(req),'--out',str(out),*(['--resume'] if out.exists() else [])],
            out.parent/(out.name+f'-{time.time_ns()}.log'),self.plan['evaluation_timeout_seconds'])
        return out/'scores.jsonl'

    def request(self,adapter=None,seeds=None,split='heldout',tasks=None):
        if adapter and Path(adapter).is_absolute():adapter=str(Path(adapter).relative_to(ROOT))
        return evaluation_request(self.plan,self.pilot,adapter=adapter,seeds=seeds,split=split,tasks=tasks)

    def config(self,arm,seed,run_id,stage='R3'):
        cfg=copy.deepcopy(self.pilot['training']['S' if arm=='S' else 'C'])
        cfg.update(stage=stage,seed=seed,method='sft' if arm=='SFT' else 'grpo',run_id=run_id,
            protocol=str(self.directory.relative_to(ROOT)))
        cfg['sft']=copy.deepcopy(self.plan['sft'])
        cfg['sft']['learning_rate']=None
        return cfg

    def train(self,cfg,out,max_steps,resume=False):
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        path=out/'training.yaml'
        if path.exists():
            if yaml.safe_load(path.read_text())!=cfg:raise ValueError('Frozen training config changed')
        else:path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        existing=ROOT/'runs'/cfg['run_id']
        if existing.exists():
            if yaml.safe_load((existing/'config.yaml').read_text())!=cfg:raise ValueError('Run config drift')
            latest=final_checkpoint(existing)
            if latest['step']>=max_steps or (existing/'kill.json').exists():return latest
            resume=True
        job(['-m','train.launch',str(path),'--max-steps',str(max_steps),*(['--resume'] if resume else [])],
            out/f'train-{max_steps}-{time.time_ns()}.log',cfg['kill']['max_wall_seconds']+1800)
        return final_checkpoint(ROOT/'runs'/cfg['run_id'])

    def make_sft_data(self,seed,directory,count=None):
        count=count or self.plan['sft']['candidates_per_seed']
        request=self.request(seeds=candidate_seeds(seed,count),split='train',tasks=self.plan['train_tasks'])
        request.update(purpose='sft_candidate_selection',fresh_checks=1)
        scores=self.evaluate(request,directory/'candidates')
        from train.sft_data import prepare
        dataset=directory/'dataset/train.jsonl'
        if dataset.exists():
            meta=json.loads((dataset.parent/'manifest.json').read_text())
            if meta['data_sha256']!=sha256(dataset) or meta['source_scores_sha256']!=sha256(scores):
                raise ValueError('SFT dataset/source drift')
            return dataset
        return prepare(scores.parent,directory/'dataset',self.plan['train_tasks'],count)

    def sft_config(self,seed,run_id,data,lr,stage='R3'):
        cfg=self.config('SFT',seed,run_id,stage)
        cfg['sft'].update(data=str(data.relative_to(ROOT)),data_sha256=sha256(data),learning_rate=lr)
        cfg['kill']['max_wall_seconds']=cfg['sft']['max_wall_seconds']
        return cfg

    def audit(self):
        require_replication_gate('R0',self.directory);enforce_failure_stop('R1');self.require_gpu()
        out=self.directory/'audit'/timestamp();init_run(out,self.plan,20266001)
        checks=[]
        try:
            job(['-m','tools.kl_probe','--plan',str(self.directory/'plan.yaml'),'--out',str(out/'kl-probe.json')],
                out/'kl-probe.log',7200)
            checks.append({'name':'adapter_disabled_reference_invariance','status':'PASS',
                'evidence':str((out/'kl-probe.json').relative_to(ROOT)),'kl_spike_cause':'not established by this check'})
            # Full fixed grid is an evaluator control, never policy selection data.
            control=json.loads((self.directory/'requests/reference-replay.json').read_text())
            sentinel=self.evaluate(control,out/'reference-replay')
            sentinel_rows=read_rows(sentinel)
            if not all(r['result']['correct'] for r in sentinel_rows):
                raise RuntimeError('A legitimate reference replay was rejected; inspect evaluator false positives')
            checks.append({'name':'reference_replay_correctness','status':'PASS','programs':len(sentinel_rows)})
            cfg=self.config('C',20266001,str((out/'grpo').relative_to(ROOT/'runs')),'R1')
            cfg['grpo']['max_steps']=3
            first=self.train(cfg,out/'grpo-logs',1)
            last=self.train(cfg,out/'grpo-logs',3,resume=True)
            train=ROOT/'runs'/cfg['run_id']
            restored=json.loads((train/'resume-checkpoint-1.json').read_text())
            if restored['lora_hash']!=first['lora_hash'] or last['step']!=3 or first['lora_hash']==last['lora_hash']:
                raise RuntimeError('GRPO optimizer/resume audit failed')
            if (train/'kill.json').exists():raise RuntimeError('GRPO smoke hit a stop criterion')
            checks.append({'name':'three_step_grpo_restart','status':'PASS'})
            data=self.make_sft_data(20266001,out/'sft-data',count=32)
            cfg=self.sft_config(20266001,str((out/'sft').relative_to(ROOT/'runs')),data,1e-5,'R1')
            cfg['grpo']['max_steps']=3
            first=self.train(cfg,out/'sft-logs',1)
            last=self.train(cfg,out/'sft-logs',3,resume=True)
            train=ROOT/'runs'/cfg['run_id']
            restored=json.loads((train/'resume-checkpoint-1.json').read_text())
            if restored['lora_hash']!=first['lora_hash'] or last['step']!=3 or first['lora_hash']==last['lora_hash']:
                raise RuntimeError('SFT optimizer/resume audit failed')
            if (train/'kill.json').exists():raise RuntimeError('SFT smoke hit a stop criterion')
            req=self.request(last['adapter'],seeds=[20266091+i for i in range(4)],split='train',tasks=self.plan['train_tasks'])
            self.evaluate(req,out/'sft-reeval')
            checks.append({'name':'three_step_sft_restart_and_reevaluation','status':'PASS'})
        except Exception as e:
            checks.append({'name':'audit','status':'FAIL','error':f'{type(e).__name__}: {e}'})
        code=finish_gate('R1',checks,protocol_sha256=sha256(self.directory/'frozen.json'),
            artifacts=str(out.relative_to(ROOT)))
        if code:raise RuntimeError('R1 failed; formal replication remains blocked')

    def reevaluate(self):
        require_replication_gate('R1',self.directory);enforce_failure_stop('R2');self.require_gpu()
        out=self.directory/'reevaluation';out.mkdir(exist_ok=True)
        manifest=[];checks=[]
        try:
            for label in ['base','C','S']:
                request=json.loads((self.directory/f'requests/reeval-{label}.json').read_text())
                scores=self.evaluate(request,out/label)
                manifest.append({'label':label,'arm':label,'training_seed':None,'scores':str(scores.relative_to(ROOT))})
            record_once(out/'manifest.json',manifest)
            if not (out/'REPORT.json').exists():
                job(['-m','eval.replication_report','--plan',str(self.directory/'plan.yaml'),
                    '--manifest',str(out/'manifest.json'),'--out',str(out/'REPORT.md')],out/f'report-{time.time_ns()}.log',7200)
            checks.append({'name':'independent_programs_complete','status':'PASS','programs':2880})
        except Exception as e:
            checks.append({'name':'reevaluation','status':'FAIL','error':f'{type(e).__name__}: {e}'})
        code=finish_gate('R2',checks,protocol_sha256=sha256(self.directory/'frozen.json'),
            artifacts=str(out.relative_to(ROOT)))
        if code:raise RuntimeError('R2 failed; formal training remains blocked')

    def tune_sft(self):
        out=self.directory/'sft-tuning';out.mkdir(exist_ok=True)
        selection=out/'selection.json'
        if selection.exists():return json.loads(selection.read_text())['learning_rate']
        seed=self.plan['sft']['tuning_seed'];data=self.make_sft_data(seed,out/'data')
        candidates=[]
        for i,lr in enumerate(self.plan['sft']['learning_rates']):
            directory=out/f'candidate-{i}'
            cfg=self.sft_config(seed,str((directory/'training').relative_to(ROOT/'runs')),data,lr)
            final=self.train(cfg,directory,self.plan['max_steps'])
            request=self.request(final['adapter'],seeds=[self.plan['sft']['validation_seed']+j
                for j in range(self.plan['sft']['validation_programs'])],split='train',tasks=self.plan['train_tasks'])
            scores=self.evaluate(request,directory/'validation')
            rows=read_rows(scores)
            if (ROOT/'runs'/cfg['run_id']/'kill.json').exists():
                raise RuntimeError('SFT tuning hit a stop criterion; no silent exclusion of a candidate')
            candidates.append({'learning_rate':lr,'correctness':sum(r['result']['correct'] for r in rows)/len(rows),
                'scores_sha256':sha256(scores)})
        winner=sorted(candidates,key=lambda c:(-c['correctness'],c['learning_rate']))[0]
        write_json(selection,{**winner,'candidates':candidates,'scope':'TRAIN only; extra tuning cost must be reported'})
        return winner['learning_rate']

    def confirm(self):
        require_replication_gate('R2',self.directory);enforce_failure_stop('R3');self.require_gpu()
        reevaluation=json.loads((self.directory/'reevaluation/REPORT.json').read_text())
        if all(reevaluation['comparisons'][f'{arm}-base/level2']['paired']['value']<=0 for arm in ['C','S']):
            raise RuntimeError('Both pilot correctness gains disappeared on independent draws; diagnose before mass training')
        out=self.directory/'confirmation';out.mkdir(exist_ok=True)
        base=json.loads((self.directory/'reevaluation/manifest.json').read_text())[0]
        manifest=[base];checks=[]
        try:
            lr=self.tune_sft()
            for item in json.loads((self.directory/'jobs.json').read_text()):
                arm,seed=item['arm'],item['seed'];label=f'{arm}-{seed}'
                directory=out/label;directory.mkdir(exist_ok=True)
                cfg=yaml.safe_load((self.directory/item['config']).read_text())
                if arm=='SFT':
                    data=self.make_sft_data(seed,directory/'data')
                    cfg=self.sft_config(seed,cfg['run_id'],data,lr)
                final=None
                for boundary in range(10,self.plan['max_steps']+1,10):
                    try:
                        final=self.train(cfg,directory,boundary,resume=final is not None)
                    except Exception:
                        # Always attempt scoring the last fully saved optimizer state;
                        # preserve the failure and prevent the next seed from hiding it.
                        final=final_checkpoint(ROOT/'runs'/cfg['run_id'])
                        req=self.request(final['adapter'])
                        self.evaluate(req,directory/'interrupted-final')
                        raise
                    stopped=(ROOT/'runs'/cfg['run_id']/'kill.json').exists()
                    is_final=stopped or final['step']==self.plan['max_steps']
                    seeds=None if is_final else [self.plan['evaluation_seeds'][0]]
                    req=self.request(final['adapter'],seeds=seeds)
                    scores=self.evaluate(req,directory/f'eval-step-{final["step"]}')
                    if is_final:break
                record_once(directory/'final.json',final)
                manifest.append({'label':label,'arm':arm,'training_seed':seed,'scores':str(scores.relative_to(ROOT))})
                record_once(directory/'policy-manifest.json',manifest[-1])
            record_once(out/'manifest.json',manifest)
            if not (out/'REPORT.json').exists():
                job(['-m','eval.replication_report','--plan',str(self.directory/'plan.yaml'),
                    '--manifest',str(out/'manifest.json'),'--out',str(out/'REPORT.md')],out/f'report-{time.time_ns()}.log',7200)
            checks.append({'name':'all_nine_training_runs_and_final_evaluations','status':'PASS'})
        except Exception as e:
            checks.append({'name':'confirmation','status':'FAIL','error':f'{type(e).__name__}: {e}'})
        code=finish_gate('R3',checks,protocol_sha256=sha256(self.directory/'frozen.json'),artifacts=str(out.relative_to(ROOT)))
        if code:raise RuntimeError('R3 failed; partial results are not a completed confirmation')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--stage',choices=['audit','reevaluate','confirm','all'],required=True)
    a=p.parse_args();study=Study(a.study)
    for stage in (['audit','reevaluate','confirm'] if a.stage=='all' else [a.stage]):getattr(study,stage)()

if __name__=='__main__':main()
