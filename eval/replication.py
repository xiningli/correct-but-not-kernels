"""K1–K4: independent seeded programs, resumable scores and unchanged C1 execution."""
import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import time
from env.reward import score_completion
from env.tasks import split_ids, verified_catalog
from tools.gatelib import ROOT, init_run, require_gate, write_json
from tools.gate_c5 import job
from tools.replication import sha256, stable_seed, local_path

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def validate_rows(rows, request, *, complete=True, scores=False):
    expected={(t,i) for t in request['tasks'] for i in range(request['samples_per_task'])}
    keys=[(r['task'],r['sample']) for r in rows]
    if len(keys)!=len(set(keys)) or not set(keys)<=expected or (complete and set(keys)!=expected):
        raise ValueError('Duplicate, foreign or missing task/sample IDs')
    if scores and any(r['result']['component']=='harness_error' for r in rows):
        raise RuntimeError('Infrastructure failure is not a model score; preserve and diagnose this attempt')
    if scores and any(r.get('evaluation_trials',0)<request['fresh_checks'] for r in rows):
        raise ValueError('Insufficient fresh input checks')
    for row in rows:
        if 'sample_seeds' in request and not request.get('reference_replay'):
            seed=stable_seed(request['sample_seeds'][row['sample']],row['task'])
            if row.get('sampling_seed')!=seed:raise ValueError('Undeclared generation seed')
        if scores and 'input_seed' in request:
            seed=stable_seed(request['input_seed'],row['task'],row['sample'])
            if row.get('input_seed')!=seed:raise ValueError('Undeclared correctness input seed')

def run(request, out, resume=False):
    require_gate('C6')
    cfg=json.loads(Path(request).read_text())
    if cfg.get('adapter'):
        adapter=local_path(cfg['adapter'])
        if sha256(adapter/'adapter_model.safetensors')!=cfg['adapter_sha256'] or \
           sha256(adapter/'adapter_config.json')!=cfg['adapter_config_sha256']:
            raise ValueError('Evaluation adapter/config hash changed')
    train,heldout=split_ids();split=cfg.get('split','heldout')
    if cfg['split_sha256']!=sha256(ROOT/'env/heldout.json'):
        raise ValueError('Split drift')
    if len(cfg['tasks'])!=len(set(cfg['tasks'])):
        raise ValueError('Repeated task in request')
    if split=='heldout' and set(cfg['tasks'])!=heldout:
        raise ValueError('Held-out grid must remain complete')
    if split not in ['train','heldout'] or (split=='train' and not set(cfg['tasks'])<=train):
        raise ValueError('TRAIN/held-out leakage')
    selection=split=='train' and cfg.get('purpose')=='sft_candidate_selection'
    if cfg['fresh_checks']<(1 if selection else 4):
        raise ValueError('Four fresh checks required in phase 2')
    out=Path(out).resolve()
    out.relative_to(ROOT/'runs')
    if out.exists():
        if not resume:raise FileExistsError('Use explicit --resume for an interrupted scoring run')
        if json.loads((out/'request.json').read_text())!=cfg:raise ValueError('Resume request changed')
        if (out/'complete.json').exists():
            validate_rows(read_rows(out/'scores.jsonl'),cfg,scores=True)
            return
    else:
        init_run(out,cfg,cfg['seed']);write_json(out/'request.json',cfg)
    tasks={t.task_id:t for t in verified_catalog()}
    generations=out/'generations.jsonl'
    if not generations.exists():
        if cfg.get('reference_replay'):
            with generations.open('x') as f:
                for task in cfg['tasks']:
                    tree=ast.parse(tasks[task].source())
                    # Input factories may contain seeded RNG used by the trusted
                    # reference. They are not part of the candidate's forward API.
                    tree.body=[node for node in tree.body if not (isinstance(node,ast.FunctionDef)
                        and node.name in ('get_inputs','get_init_inputs'))]
                    source=ast.unparse(tree)+'\nclass ModelNew(Model):\n    pass\n'
                    for i in range(cfg['samples_per_task']):
                        f.write(json.dumps({'task':task,'sample':i,'text':'```python\n'+source+'\n```',
                            'source_sha256':tasks[task].sha256,'control':'reference_replay'})+'\n')
        else:
            launch=dict(cfg)
            if launch.get('adapter'):launch['adapter']=str(local_path(launch['adapter']))
            # This request contains only machine-local path expansion, kept as provenance.
            launch_path=out/'sampling-request.json'
            if not launch_path.exists():write_json(launch_path,launch)
            job(['-m','tools.sample_policy','--request',str(launch_path),'--out',str(generations)],
                out/'generation.log',cfg['generation_timeout_seconds'])
    rows=read_rows(generations);validate_rows(rows,cfg)
    for row in rows:
        if row['source_sha256']!=tasks[row['task']].sha256:raise ValueError('Task source drift')
    scores=out/'scores.jsonl'
    prior=read_rows(scores) if scores.exists() else []
    validate_rows(prior,cfg,complete=False,scores=True)
    done={(r['task'],r['sample']) for r in prior}
    import torch
    torch.set_num_threads(8)
    with scores.open('a') as f:
        for i,row in enumerate(rows):
            if (row['task'],row['sample']) in done:continue
            seed=stable_seed(cfg['input_seed'],row['task'],row['sample'])
            # Distinct attempt directories retain any interrupted execution evidence.
            work=out/'rollouts'/f'{i:05d}-{time.time_ns()}'
            result=score_completion(row['text'],tasks[row['task']],work,seed=seed,
                trials=cfg['fresh_checks'],split=split)
            record={**row,'result':asdict(result),'evaluation_trials':cfg['fresh_checks'],
                    'input_seed':seed,'program_directory':str(work.relative_to(ROOT))}
            f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
            print(f'evaluated {len(done)+1}/{len(rows)}',flush=True)
            if result.component=='harness_error':
                raise RuntimeError('Evaluator infrastructure failed; retained failing row')
            done.add((row['task'],row['sample']))
    final=read_rows(scores);validate_rows(final,cfg,scores=True)
    write_json(out/'complete.json',{'programs':len(final),'request_sha256':sha256(out/'request.json'),
        'scores_sha256':sha256(scores),'generations_sha256':sha256(generations)})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.request,a.out,a.resume)

if __name__=='__main__':main()
