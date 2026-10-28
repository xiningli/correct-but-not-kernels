"""K1–K4: Gemma base-policy signal on the frozen training distribution only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import yaml
from env.reward import extract_program
from env.run_kernel import RunResult, evaluate_task
from env.tasks import verified_catalog, validate_split
from eval.signal import summarize
from tools.gatelib import ROOT, enforce_failure_stop, finish_gate, require_gate, timestamp, write_json, init_run

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'train/configs/c4.yaml', help='K1–K4: fixed TRAIN request')
    p.add_argument('--operator-continue', help='K1: recorded operator authorization after two failed attempts')
    a = p.parse_args()
    try:
        enforce_failure_stop('C4')
    except RuntimeError:
        if not a.operator_continue:
            raise
        print(f'Operator-authorized continuation: {a.operator_continue}',flush=True)
    c3 = require_gate('C3')
    cfg = yaml.safe_load(a.config.read_text())
    tasks = {t.task_id:t for t in verified_catalog()}
    train, _ = validate_split(list(tasks.values()))
    if not set(cfg['tasks']) <= train or cfg['samples_per_task'] != 32:
        raise ValueError('C4 must use TRAIN tasks, 32 samples each')
    directory = ROOT/'runs/c4'/timestamp()
    init_run(directory, cfg, cfg['seed'])
    checks, summaries = [], {}
    try:
        for model in cfg['models']:
            sub = directory/model
            sub.mkdir()
            write_json(sub/'request.json', {**cfg, 'model':model})
            with (sub/'generation.log').open('x') as log:
                subprocess.run([str(ROOT/'scripts/gpu_env.sh'), sys.executable,
                    '-m','tools.sample_policy','--request',str(sub/'request.json'),
                    '--out',str(sub/'generations.jsonl')], cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT, check=True, timeout=7200)
            import torch
            torch.set_num_threads(8)
            rows = []
            generations = [json.loads(x) for x in (sub/'generations.jsonl').read_text().splitlines()]
            if len(generations) != 32*len(cfg['tasks']):
                raise RuntimeError('Incomplete generation file')
            with (sub/'scores.jsonl').open('x') as scores:
                for i, row in enumerate(generations):
                    source = extract_program(row['text'])
                    result = evaluate_task(source, tasks[row['task']], sub/f'rollout-{i:04d}',
                                           seed=cfg['seed']+i) if source else RunResult('format')
                    scored = {**row,'result':asdict(result)}
                    scores.write(json.dumps(scored)+'\n'); scores.flush()
                    rows.append(scored)
                    print(json.dumps({'model':model,'rollout':i,'component':result.component,
                                      'correct':result.correct,'hacks':result.hacks}), flush=True)
                    if result.component == 'harness_error':
                        raise RuntimeError(f'Infrastructure error: {result.details}')
            summaries[model] = summarize(rows, cfg['group_size'])
            write_json(sub/'signal.json', summaries[model])
            with (directory/'metrics.jsonl').open('a') as f:
                f.write(json.dumps({'model':model,**summaries[model]})+'\n')
        eligible = [m for m in cfg['models'] if summaries[m]['mean_p_signal'] >= cfg['minimum_signal']
                    and summaries[m]['signal_tasks'] >= cfg['minimum_signal_tasks']]
        selected = eligible[0] if eligible else None
        write_json(directory/'selection.json', {'model':selected,'tasks':cfg['tasks'],
                   'summaries':summaries,'config':cfg})
        checks.append({'name':'base_policy_signal','status':'PASS' if selected else 'FAIL',
                       'selected_model':selected,'summaries':summaries})
    except Exception as error:
        checks.append({'name':'blocking_failure','status':'FAIL','error':f'{type(error).__name__}: {error}'})
        print(checks[-1]['error'], flush=True)
    raise SystemExit(finish_gate('C4', checks, prerequisite=c3, artifacts=str(directory.relative_to(ROOT)),
                                operator_continue=a.operator_continue))

if __name__ == '__main__':
    main()
