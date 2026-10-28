"""Sealed v4 development intervention and retrospective decomposition runner.

No training or new confirmatory test outputs. Unknown observations are reported
explicitly. A harness failure stops evaluation and preserves the partial report.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from env.reward import extract_program
from env.tasks import load_reference, verified_catalog
from research.reliability_v2 import decomposition
from research.reliability_v2.capability_controls import boot_identity, source_hashes
from research.reliability_v2.fixed_interface import CONDITIONS, assemble, extract_code, messages
from research.reliability_v2.new_tasks import catalog, make_inputs, trusted_reference
from research.reliability_v2.repair_protocol import sampling_seed
from research.reliability_v2.strict_evaluator import evaluate
from tools.gatelib import ROOT, git_state
from tools.replication import sha256, verify_frozen


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def report(directory):
    directory = Path(directory)
    path = directory / 'metrics.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    groups, seen = defaultdict(list), set()
    for r in rows:
        key = (r['policy'], r.get('condition', 'saved_program'), r['task'], r.get('sample', 0))
        if key in seen:
            raise ValueError('Duplicate attempt in report')
        seen.add(key); groups[key[:2]].append(r)
    cells = []
    for (policy, condition), values in sorted(groups.items()):
        cell = {'policy': policy, 'condition': condition, 'attempts': len(values),
                'components': dict(Counter(v['result']['component'] for v in values))}
        for key in ['initialization_completed', 'forward_completed', 'numerically_correct', 'contract_compliant', 'correct']:
            counts = Counter(str(v['result'].get(key)) for v in values)
            cell[key] = {label: counts[value] for label, value in [('true', 'True'), ('false', 'False'), ('unknown', 'None')]}
        cell['generated_tokens'] = sum(v.get('new_tokens', 0) for v in values)
        cell['prompt_tokens'] = sum(v.get('prompt_tokens', 0) for v in values)
        cell['evaluation_seconds'] = sum(v.get('evaluation_seconds', 0) for v in values)
        if all('original' in v for v in values):
            cell['original_correct'] = sum(v['original']['correct'] for v in values)
        cells.append(cell)
    outcome = json.loads((directory / 'outcome.json').read_text()) if (directory / 'outcome.json').exists() else {'status': 'partial'}
    data = {'outcome': outcome, 'attempts': len(rows), 'cells': cells,
            'limits': ['Retrospective known-task decomposition and development interface checks, not confirmatory new-task gains.',
                      'All attempt denominators are retained; unobserved runtime/numerical outcomes are unknown, not negative observations.',
                      'Initialization operations are separate from the custom-forward contract.',
                      'PyTorch compute was legal in the old study; this diagnostic does not establish cheating.',
                      'Profiling cost is not a kernel speed benchmark. No training occurred.']}
    write(directory / 'REPORT.json', data)
    lines = ['# CUDA capability decomposition / interface development', '',
             '| Policy | Condition | Attempts | Numerical true / unknown | Custom true / unknown | Strict successes |',
             '|---|---|---:|---:|---:|---:|']
    for c in cells:
        n, k = c['numerically_correct'], c['contract_compliant']
        lines.append(f"| {c['policy']} | {c['condition']} | {c['attempts']} | {n['true']} / {n['unknown']} | {k['true']} / {k['unknown']} | {c['correct']['true']} |")
    lines += ['', 'Outcome: ' + outcome['status'], '', *data['limits']]
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return data


def validate_controls(path, isolation):
    receipt = json.loads(Path(path).read_text())
    if receipt['status'] != 'PASS' or receipt['controls'] != 14 or receipt['source_sha256'] != source_hashes() or receipt['boot'] != boot_identity():
        raise ValueError('Current source/current boot controls required')
    isolated = json.loads(Path(isolation).read_text())
    if isolated['status'] != 'PASS' or len(isolated['checks']) != 6 or any(c['status'] != 'PASS' for c in isolated['checks']):
        raise ValueError('All six isolation controls required')


def execute(args):
    import torch
    verify_frozen(ROOT / 'runs/replication-v1.1')
    validate_controls(args.controls, args.isolation)
    torch.set_num_threads(4)
    plan_path = ROOT / 'experiments/cuda-capability-v4.yaml'
    plan = yaml.safe_load(plan_path.read_text())
    if plan['training_enabled'] or plan['formal_new_task_generation_enabled']:
        raise ValueError('Unsupported execution scope')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'config.yaml').write_bytes(plan_path.read_bytes())
    write(args.out / 'git_sha', git_state())
    (args.out / 'seed').write_text(str(plan['seeds']['generation']) + '\n')
    write(args.out / 'versions.json', {p: importlib.metadata.version(p) for p in ['torch', 'triton', 'vllm', 'transformers', 'safetensors']})
    write(args.out / 'source-seal.json', source_hashes())
    write(args.out / 'execution.json', {'utc': utc(), 'boot': boot_identity(),
          'controls_sha256': sha256(args.controls), 'isolation_sha256': sha256(args.isolation),
          'manifest_sha256': sha256(args.manifest) if args.manifest else None,
          'policies_sha256': sha256(args.policies) if args.policies else None})
    metrics = args.out / 'metrics.jsonl'; metrics.touch()
    scored, start = 0, time.monotonic()
    outcome = {'status': 'running', 'action': args.action, 'utc': utc()}

    def save_row(row, result, elapsed, total):
        nonlocal scored
        saved = {**row, 'result': result, 'evaluation_seconds': elapsed}
        with metrics.open('a') as f:
            f.write(json.dumps(saved, allow_nan=False) + '\n'); f.flush()
        scored += 1
        write(args.out / 'progress.json', {'phase': 'evaluation', 'policy': row['policy'],
              'task': row['task'], 'scored': scored, 'total': total, 'utc': utc()})
        if scored % 28 == 0:
            report(args.out)
        if result['component'] == 'harness_error':
            raise RuntimeError('Harness error: ' + result.get('error', 'inspect verdict'))

    try:
        if args.action == 'interface':
            policies = {p['label']: p for p in json.loads(args.policies.read_text())}
            tasks = {t['id']: t for t in catalog() if t['split'] == 'development'}
            entries = []
            for task in tasks.values():
                for condition in CONDITIONS:
                    prompt = messages(task, condition)
                    entries.append({'task': task['id'], 'condition': condition, 'messages': prompt,
                        'prompt_sha256': hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()})
            total = len(plan['policies']) * len(entries)
            for label in plan['policies']:
                d = args.out / label; d.mkdir()
                request = {'scope': 'development_only', 'policy': policies[label], 'entries': entries,
                           'sampling': plan['sampling'], 'namespace': plan['seeds']['generation']}
                write(d / 'request.json', request)
                write(args.out / 'progress.json', {'phase': 'generation', 'policy': label, 'scored': scored, 'total': total, 'utc': utc()})
                generation_start = time.monotonic()
                with (d / 'generation.log').open('x') as log:
                    subprocess.run([sys.executable, '-m', 'research.reliability_v2.sample_interface',
                                    '--request', str(d / 'request.json'), '--out', str(d / 'generations.jsonl')],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
                write(d / 'generation-cost.json', {'seconds': time.monotonic() - generation_start})
                rows = [json.loads(s) for s in (d / 'generations.jsonl').read_text().splitlines()]
                if len(rows) != len(entries) or {(r['task'], r['condition']) for r in rows} != {(e['task'], e['condition']) for e in entries}:
                    raise ValueError('Incomplete/duplicate generated attempt grid')
                for index, row in enumerate(rows):
                    if row['policy'] != label:
                        raise ValueError('Policy identity changed')
                    task = tasks[row['task']]
                    seed = sampling_seed(plan['seeds']['inputs'], task['id'], 0, 0)
                    cases = [make_inputs(task, seed=seed + c, case=c) for c in range(4)]
                    expected = [trusted_reference(task)(*values) for values in cases]
                    begin = time.monotonic()
                    source = extract_code(row['text'])
                    result = decomposition.empty_result('format')
                    if source is not None:
                        try:
                            assembled = assemble(source, task, row['condition'])
                        except (SyntaxError, ValueError) as error:
                            result['format_reason'] = str(error)
                        else:
                            result = evaluate(assembled, cases, expected, d / f'candidate-{index:03d}', seed=seed)
                            # v3 returned False for unobserved fields. For this new
                            # report, incomplete observations are explicitly null.
                            if 'checks' not in result and result['component'] != 'incorrect':
                                result['numerically_correct'] = None
                            if 'contract_checks' not in result:
                                result['contract_compliant'] = None
                    save_row(row, result, time.monotonic() - begin, total)
        else:
            manifest = json.loads(args.manifest.read_text())
            if manifest['attempts'] != 5600 or manifest['policies'] != decomposition.LABELS:
                raise ValueError('Expected all 5600 saved attempts of the seven primary policies')
            if manifest['frozen_sha256'] != sha256(ROOT / 'runs/replication-v1.1/frozen.json'):
                raise ValueError('Original source seal changed')
            tasks = {t.task_id: t for t in verified_catalog() if t.level == 2}
            if (len(manifest['rows']) != 5600 or
                {(r['policy'], r['task'], r['sample']) for r in manifest['rows']} !=
                {(p, t, s) for p in decomposition.LABELS for t in tasks for s in range(8)}):
                raise ValueError('Incomplete/duplicate saved attempt grid')
            for index, row in enumerate(manifest['rows']):
                if tasks[row['task']].sha256 != row['reference_sha256']:
                    raise ValueError('Reference changed')
                source = row['source']
                if source is not None and hashlib.sha256(source.encode()).hexdigest() != row['candidate_sha256']:
                    raise ValueError('Saved program changed')
                begin = time.monotonic()
                result = (decomposition.empty_result('format') if source is None else
                          decomposition.evaluate(source, load_reference(tasks[row['task']]),
                          args.out / 'candidates' / f'{index:05d}', seed=row['input_seed']))
                save_row({k: v for k, v in row.items() if k != 'source'}, result,
                         time.monotonic() - begin, manifest['attempts'])
        outcome['status'] = 'complete'
    except Exception as error:
        outcome.update(status='stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        outcome.update(scored=scored, seconds=time.monotonic() - start, finished_utc=utc())
        write(args.out / 'outcome.json', outcome)
        report(args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['interface', 'decomposition', 'report', 'prepare'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--controls', type=Path)
    p.add_argument('--isolation', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--policies', type=Path)
    a = p.parse_args()
    if a.action == 'report':
        print(json.dumps(report(a.out), indent=2))
    elif a.action == 'prepare':
        print(json.dumps(decomposition.prepare_manifest(a.out)))
    else:
        execute(a)


if __name__ == '__main__':
    main()
