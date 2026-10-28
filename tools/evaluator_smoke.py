"""K1/K3: positive controls on every C4 TRAIN task before sampling policy text."""
from dataclasses import asdict
import json
import yaml
from env.tasks import verified_catalog
from env.run_kernel import evaluate_task
from tools.gatelib import ROOT, timestamp, init_run, write_json

def main():
    import torch
    torch.set_num_threads(8)
    cfg = yaml.safe_load((ROOT/'train/configs/c4.yaml').read_text())
    tasks = {t.task_id:t for t in verified_catalog()}
    directory = ROOT/'runs/evaluator-smoke'/timestamp()
    init_run(directory, cfg, cfg['seed'])
    records = []
    for i, name in enumerate(cfg['tasks']):
        task = tasks[name]
        source = task.source().replace('class Model(', 'class ModelNew(').replace('super(Model,', 'super(ModelNew,')
        result = evaluate_task(source, task, directory/str(i), seed=cfg['seed']+i)
        record = {'task':name,**asdict(result)}
        records.append(record)
        print(json.dumps({'task':name,'correct':result.correct,'component':result.component}), flush=True)
        if not result.correct:
            print(json.dumps(record), flush=True)
            break
    write_json(directory/'controls.json', records)
    raise SystemExit(0 if len(records)==len(cfg['tasks']) and all(x['correct'] for x in records) else 1)

if __name__ == '__main__':
    main()
