"""K1/K2/K4: frozen held-out grid, four fresh checks and full timing per program."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from env.reward import score_completion
from env.tasks import verified_catalog, validate_split
from tools.gate_c5 import job
from tools.gatelib import ROOT, init_run, require_gate, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True,help='K1–K4: frozen pilot evaluation request')
    p.add_argument('--out',type=Path,required=True,help='K1–K4: new evaluation directory')
    a=p.parse_args()
    require_gate('C5')
    cfg=json.loads(a.request.read_text())
    tasks={t.task_id:t for t in verified_catalog()}
    _,heldout=validate_split(list(tasks.values()))
    digest=hashlib.sha256((ROOT/'env/heldout.json').read_bytes()).hexdigest()
    if digest!=cfg['split_sha256'] or set(cfg['tasks'])!=heldout:
        raise RuntimeError('Held-out split drift or incomplete evaluation request')
    if cfg['fresh_checks']<4 or cfg['samples_per_task']<1:
        raise ValueError('Four fresh checks and at least one program per task required')
    init_run(a.out,cfg,cfg['seed'])
    write_json(a.out/'request.json',cfg)
    job(['-m','tools.sample_policy','--request',str(a.out/'request.json'),
         '--out',str(a.out/'generations.jsonl')],a.out/'generation.log',7200)
    rows=[json.loads(x) for x in (a.out/'generations.jsonl').read_text().splitlines()]
    if len(rows)!=len(cfg['tasks'])*cfg['samples_per_task']:
        raise RuntimeError('Incomplete held-out generations')
    import torch
    torch.set_num_threads(8)
    with (a.out/'scores.jsonl').open('x') as f:
        for i,row in enumerate(rows):
            result=score_completion(row['text'],tasks[row['task']],a.out/f'rollout-{i:04d}',
                seed=cfg['seed']+i,trials=cfg['fresh_checks'],split='heldout')
            record={**row,'result':asdict(result),'evaluation_trials':cfg['fresh_checks']}
            f.write(json.dumps(record)+'\n'); f.flush()
            # Avoid showing held-out per-item outcomes while the frozen pilots
            # are running. They remain in the artifact for final analysis.
            print(f'heldout evaluated {i+1}/{len(rows)}',flush=True)
            if result.component=='harness_error':
                raise RuntimeError(f'Held-out infrastructure failure; see rollout-{i:04d}')
    write_json(a.out/'complete.json',{'programs':len(rows),'split_sha256':digest})

if __name__ == "__main__":
    main()
