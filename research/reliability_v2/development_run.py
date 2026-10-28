"""K1/K4, T1/T3: small isolated development pilot, no confirmatory test access.

Runs 6 development graphs x 2 prompts x base/C21/SFT21, one sample each.
Records all failures and costs. Stops on any harness error. New training, test
task sampling, and autonomous promotion of development results are prohibited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from env.reward import extract_program
from research.reliability_v2.new_tasks import catalog, make_inputs, trusted_reference
from research.reliability_v2.prepare_transfer import messages
from research.reliability_v2.repair_protocol import sampling_seed
from research.reliability_v2.strict_evaluator import evaluate
from tools.gatelib import ROOT
from tools.replication import sha256, verify_frozen


def write(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policies', type=Path, required=True)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--isolation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    verify_frozen(ROOT / 'runs/replication-v1.1')
    controls = json.loads(a.controls.read_text())
    isolation = json.loads(a.isolation.read_text())
    if controls['status'] != 'PASS' or isolation['status'] != 'PASS':
        raise ValueError('Development requires passing runtime and isolation controls')
    for path, digest in controls['source_sha256'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Strict evaluator changed after GPU controls')
    if len(isolation['checks']) != 6 or any(c['status'] != 'PASS' for c in isolation['checks']):
        raise ValueError('Incomplete strict C1 regression coverage')
    plan_path = ROOT / 'experiments/cuda-transfer-v3-draft.yaml'
    plan = yaml.safe_load(plan_path.read_text())
    if plan['generation_enabled'] or plan['training_enabled'] or not plan['development_generation']['enabled']:
        raise ValueError('Only explicitly scoped development generation is supported')
    policies = {x['label']: x for x in json.loads(a.policies.read_text())}
    selected = plan['development_generation']['policies']
    if selected != ['base', 'C-20261021', 'SFT-20261021']:
        raise ValueError('Undeclared development policy selection')
    tasks = [t for t in catalog() if t['split'] == 'development']
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'config.yaml').write_bytes(plan_path.read_bytes())
    (a.out / 'git_sha').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True))
    (a.out / 'seed').write_text(str(plan['seeds']['generation_namespace']) + '\n')
    source_hashes = {str(x.relative_to(ROOT)): sha256(x) for x in (ROOT / 'research/reliability_v2').glob('*.py')}
    write(a.out / 'source-seal.json', source_hashes)
    import torch
    torch.set_num_threads(2)
    write(a.out / 'versions.json', {'python': sys.version, 'torch': torch.__version__,
          'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
          'controls_sha256': sha256(a.controls), 'isolation_sha256': sha256(a.isolation),
          'policies_sha256': sha256(a.policies)})
    entries = []
    for task in tasks:
        for condition in plan['strict_generation']['conditions']:
            prompt = messages(task, condition)
            entries.append({'task': task['id'], 'condition': condition, 'messages': prompt,
                'prompt_sha256': hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()})
    start, scored = time.monotonic(), 0
    outcome = {'status': 'running', 'scope': 'development_only_not_confirmatory'}
    metrics = a.out / 'metrics.jsonl'
    metrics.touch()
    try:
        for label in selected:
            directory = a.out / label; directory.mkdir()
            request = {'scope': 'development_only', 'policy': policies[label],
                       'sampling': plan['sampling'], 'entries': entries,
                       'namespace': plan['seeds']['generation_namespace']}
            write(directory / 'request.json', request)
            write(a.out / 'progress.json', {'phase': 'generation', 'policy': label, 'scored': scored, 'total': 36})
            generation_start = time.monotonic()
            with (directory / 'generation.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'research.reliability_v2.sample_transfer',
                    '--request', str(directory / 'request.json'), '--out', str(directory / 'generations.jsonl')],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800, cwd=ROOT)
            write(directory / 'generation-cost.json', {'seconds': time.monotonic() - generation_start})
            rows = [json.loads(s) for s in (directory / 'generations.jsonl').read_text().splitlines()]
            expected_ids = {(e['task'], e['condition']) for e in entries}
            if len(rows) != len(entries) or {(r['task'], r['condition']) for r in rows} != expected_ids:
                raise ValueError('Incomplete or duplicate development generations')
            for index, row in enumerate(rows):
                if row['policy'] != label:
                    raise ValueError('Policy mismatch')
                task = next(t for t in tasks if t['id'] == row['task'])
                model = trusted_reference(task)
                seed = sampling_seed(plan['seeds']['hidden_inputs'], task['id'], 0, 0)
                inputs = [make_inputs(task, seed=seed + c, case=c) for c in range(4)]
                expected = [model(*values) for values in inputs]
                source = extract_program(row['text'])
                before = time.monotonic()
                result = ({'component': 'format', 'correct': False, 'numerically_correct': False,
                           'contract_compliant': False} if source is None else
                          evaluate(source, inputs, expected, directory / f'candidate-{index:03}', seed=seed))
                scored += 1
                saved = {**row, 'result': result, 'evaluation_seconds': time.monotonic() - before}
                with metrics.open('a') as stream:
                    stream.write(json.dumps(saved, allow_nan=False) + '\n')
                write(a.out / 'progress.json', {'phase': 'evaluation', 'policy': label, 'scored': scored, 'total': 36})
                print(json.dumps({'policy': label, 'task': row['task'], 'condition': row['condition'],
                                  'component': result['component'], 'scored': scored}), flush=True)
                if result['component'] == 'harness_error':
                    raise RuntimeError('Infrastructure error; do not continue candidate scoring')
        outcome['status'] = 'complete'
    except Exception as error:
        outcome.update(status='stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        outcome.update(scored=scored, elapsed_seconds=time.monotonic() - start,
                       utc=datetime.now(timezone.utc).isoformat(), metrics_sha256=sha256(metrics))
        write(a.out / 'outcome.json', outcome)


if __name__ == '__main__':
    main()
