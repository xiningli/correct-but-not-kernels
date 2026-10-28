"""K1/K4, T1-T3: verify old policy identities and prepare reviewable new inputs.

This command generates NO model outputs and is NOT a protocol freeze. It binds
the existing final policies to archived evaluation receipts, includes one base,
and emits symmetric prompts. Test task/reference review and repair curation
remain explicit blockers to a confirmatory run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from research.reliability_v2.new_tasks import catalog
from tools.replication import sha256, verify_frozen

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = '''Write a complete Python program defining ModelNew(torch.nn.Module).
Implement the mathematical specification with custom CUDA C++ or Triton kernels.
Use PyTorch only for empty tensor allocation, shape/stride metadata and views.
All arithmetic, reductions, indexing computation and dtype conversion must run
in your custom GPU kernels. Do not use PyTorch arithmetic, torch.compile, vendor
BLAS/DNN libraries, reference replay, CPU computation or runtime inspection.
Preserve inputs, produce fresh output storage, and handle the declared dtype
and layout domain. Return one Python code block; no explanation is required.
'''
SCAFFOLD = '''Interface reminders (identical for every policy):
- import torch; import torch.nn as nn; class ModelNew(nn.Module).
- forward receives the same positional arguments as Model.forward; no init args.
- torch.empty(shape, device=x.device, dtype=torch.float32) allocates the output.
- Inputs may be non-contiguous: shape, stride() and storage offset define layout.
- For Triton use import triton and import triton.language as tl; tl.constexpr
  arguments are compile-time parameters. Launch kernel[grid](arguments).
- For CUDA use torch.utils.cpp_extension.load_inline and declare __global__
  kernel functions explicitly. Implement the math in those kernel functions.
'''


def policies_from_receipt(path):
    receipt = json.loads(Path(path).read_text())
    expected = {'base'} | {f'{arm}-{seed}' for arm in ['C', 'S', 'SFT']
                         for seed in [20261021, 20261022, 20261023]}
    if set(receipt['policies']) != expected:
        raise ValueError('Need shared base and every existing training seed')
    policies = []
    for label, record in sorted(receipt['policies'].items()):
        directory = ROOT / record['path']
        if not directory.resolve().is_relative_to(ROOT):
            raise ValueError('Archive receipt escapes project')
        request_path = directory / 'request.json'
        complete = json.loads((directory / 'complete.json').read_text())
        if (sha256(request_path) != complete['request_sha256']
                or complete['scores_sha256'] != record['scores_sha256']
                or complete['generations_sha256'] != record['generations_sha256']):
            raise ValueError('Archived policy evidence changed: ' + label)
        request = json.loads(request_path.read_text())
        adapter = request.get('adapter')
        if label == 'base':
            if adapter is not None:
                raise ValueError('Shared base unexpectedly has an adapter')
        else:
            step = 15 if label == 'S-20261021' else 40
            suffix = f'{label}/checkpoints/policy-step-{step}'
            if not adapter or not adapter.endswith(suffix):
                raise ValueError('Final-checkpoint rule changed: ' + label)
            if any(len(request.get(k, '')) != 64 for k in ['adapter_sha256', 'adapter_config_sha256']):
                raise ValueError('Missing adapter hashes')
        policies.append({'label': label, 'arm': label.split('-')[0],
                         'training_seed': None if label == 'base' else int(label.split('-')[1]),
                         **{k: request.get(k) for k in ['model', 'adapter', 'adapter_sha256', 'adapter_config_sha256']},
                         'prior_request_sha256': sha256(request_path)})
    return policies


def messages(task, condition):
    if condition not in {'minimal_contract', 'api_scaffold'}:
        raise ValueError('Undeclared prompt intervention')
    prompt = SYSTEM + (('\n' + SCAFFOLD) if condition == 'api_scaffold' else '')
    content = ('Implement this trusted mathematical reference. Output dtype is float32.\n'
               + 'Valid domain: ' + json.dumps(task['domain'], sort_keys=True)
               + '\n```python\n' + task['reference_source'] + '```')
    return [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--receipt', type=Path, default=ROOT /
                   'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/local-analysis-verification.json')
    a = p.parse_args()
    verify_frozen(ROOT / 'runs/replication-v1.1')
    plan_path = ROOT / 'experiments/cuda-transfer-v3-draft.yaml'
    plan = yaml.safe_load(plan_path.read_text())
    if plan['status'] != 'development_not_frozen' or plan['generation_enabled'] or plan['training_enabled']:
        raise ValueError('This preparer is only for the non-executing development draft')
    policies = policies_from_receipt(a.receipt)
    tasks = catalog()
    a.out.mkdir(parents=True, exist_ok=False)
    prompts = [{'task': t['id'], 'split': t['split'], 'condition': c, 'messages': messages(t, c)}
               for t in tasks for c in plan['strict_generation']['conditions']]
    for row in prompts:
        row['sha256'] = hashlib.sha256(json.dumps(row['messages'], sort_keys=True).encode()).hexdigest()
    records = {'policies.json': policies, 'tasks.json': tasks, 'prompts.json': prompts}
    for name, content in records.items():
        (a.out / name).write_text(json.dumps(content, indent=2) + '\n')
    report = {'status': 'prepared_for_review_not_execution', 'ready_for_policy_evaluation': False,
              'prior_source_seal': 'PASS', 'policies': len(policies),
              'base_policies': sum(p['arm'] == 'base' for p in policies),
              'test_graph_candidates': sum(t['split'] == 'test_candidate' for t in tasks),
              'development_graphs': sum(t['split'] == 'development' for t in tasks),
              'prompt_conditions': plan['strict_generation']['conditions'],
              'plan_sha256': sha256(plan_path), 'receipt_sha256': sha256(a.receipt),
              'files': {n: sha256(a.out / n) for n in records},
              'remaining_before_freeze': [
                  'independent_reference_and_prior_task_overlap_review',
                  'sealed_real_repair_test_set_distinct_from_the_two_development_bugs',
                  'runtime_controls_bound_to_final_source_and_boot',
                  'development_generation_cost_and_statistical_precision_estimate',
                  'analysis_and_end_to_end_controller_validation']}
    (a.out / 'prepared.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
