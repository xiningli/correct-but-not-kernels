"""K1/K3: sealed all-policy replay of one task affected by evaluator disk errors.

No generation, training, best-checkpoint selection or old-score rewriting.
The prospective strict-kernel/new-task experiment remains separate and disabled.
"""
import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
from pathlib import Path
import time

from env.reward import extract_program
from env.run_kernel import RunResult
from env.tasks import verified_catalog
from research.reliability_v2.audit import DEFAULT_RECEIPT
from research.reliability_v2.controls import run_controls
from research.reliability_v2.evaluator import evaluate_task
from tools.gatelib import ROOT, init_run
from tools.replication import sha256, verify_frozen

CODE = ['research/reliability_v2/' + f for f in
        ['storage.py', 'sandbox.py', 'tensor_checks.py', 'kernel_worker.py', 'evaluator.py', 'controls.py', 'replay.py']]
TASK = 'level1/38_L1Norm_'


def code_hashes():
    return {f: sha256(ROOT / f) for f in CODE}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def seal(path):
    verify_frozen(ROOT / 'runs/replication-v1.1')
    receipt = json.loads(DEFAULT_RECEIPT.read_text())
    rows = []; inputs = []
    for label, saved in receipt['policies'].items():
        scores = ROOT / saved['path'] / 'scores.jsonl'
        if sha256(scores) != saved['scores_sha256']:
            raise ValueError('Old score grid changed')
        selected = [json.loads(line) for line in scores.open() if json.loads(line)['task'] == TASK]
        if len(selected) != 8 or {x['sample'] for x in selected} != set(range(8)):
            raise ValueError('Every policy must include all eight original attempts')
        inputs.append({'label': label, 'scores_sha256': saved['scores_sha256']})
        for row in selected:
            source = extract_program(row['text'])
            rows.append({'label': label, 'sample': row['sample'], 'task': TASK,
                         'input_seed': row['input_seed'], 'source_sha256': row['source_sha256'],
                         'source': source, 'candidate_sha256': hashlib.sha256(source.encode()).hexdigest() if source else None,
                         'original': {'component': row['result']['component'], 'correct': row['result']['correct'],
                                      'hacks': row['result']['hacks'], 'speedup': row['result']['speedup']}})
    labels = {'base'} | {f'{a}-{s}' for a in ['C', 'S', 'SFT'] for s in [20261021, 20261022, 20261023]}
    if {r['label'] for r in rows} != labels or len(rows) != 80:
        raise ValueError('Need shared base and all nine final policies')
    record = {'kind': 'measurement_repair_replay_not_new_generalization', 'created_utc': utc(),
              'code_hashes': code_hashes(), 'old_frozen_sha256': sha256(ROOT / 'runs/replication-v1.1/frozen.json'),
              'inputs': inputs, 'task': TASK, 'trials': 4, 'timing_samples': 31,
              'regime': 'original_permissive', 'candidate_selection': 'all_80_saved_attempts_no_regeneration',
              'changes': ['single-copy streamed input storage', 'explicit free-space preflight',
                          'shape/dtype checks before output storage', 'NaN-safe persistence checks'],
              'rows': sorted(rows, key=lambda x: (x['label'], x['sample']))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    print(json.dumps({'manifest': str(path), 'sha256': sha256(path), 'attempts': len(rows)}))


def verified_manifest(path):
    manifest = json.loads(path.read_text())
    verify_frozen(ROOT / 'runs/replication-v1.1')
    if manifest['code_hashes'] != code_hashes() or manifest['old_frozen_sha256'] != sha256(ROOT / 'runs/replication-v1.1/frozen.json'):
        raise ValueError('Replay or old study source changed after seal')
    return manifest


def require_gpu():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 80 << 30:
        raise RuntimeError('At least 80 GiB VRAM required; no CPU fallback')
    torch.set_num_threads(8)


def controls(manifest_path, out):
    manifest = verified_manifest(manifest_path); require_gpu()
    init_run(out, {k: v for k, v in manifest.items() if k != 'rows'}, 2026101201)
    try:
        checks = run_controls(out / 'cases')
        record = {'status': 'PASS', 'manifest_sha256': sha256(manifest_path), 'code_hashes': code_hashes(),
                  'utc': utc(), 'checks': [dict(name=x['name'], passed=x['passed']) for x in checks]}
        (out / 'complete.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)
    except Exception as error:
        (out / 'failure.json').write_text(json.dumps({'utc': utc(), 'error': f'{type(error).__name__}: {error}'}, indent=2) + '\n')
        raise


def replay(manifest_path, controls_path, out):
    manifest = verified_manifest(manifest_path); require_gpu()
    passed = json.loads((controls_path / 'complete.json').read_text())
    if passed['status'] != 'PASS' or passed['manifest_sha256'] != sha256(manifest_path) or passed['code_hashes'] != code_hashes():
        raise ValueError('Matching controls PASS required before replay')
    task = next(t for t in verified_catalog() if t.task_id == TASK)
    init_run(out, {k: v for k, v in manifest.items() if k != 'rows'}, 2026101201)
    rows = []
    with (out / 'scores.jsonl').open('x') as stream:
        for i, row in enumerate(manifest['rows']):
            if row['source_sha256'] != task.sha256:
                raise ValueError('Reference source drift')
            if row['source'] is None:
                result = RunResult('format')
            else:
                if hashlib.sha256(row['source'].encode()).hexdigest() != row['candidate_sha256']:
                    raise ValueError('Candidate source drift')
                result = evaluate_task(row['source'], task, out / 'rollouts' / f'{i:03d}-{time.time_ns()}', seed=row['input_seed'])
            record = {k: v for k, v in row.items() if k != 'source'}
            record['replay'] = asdict(result)
            stream.write(json.dumps(record, allow_nan=False) + '\n'); stream.flush(); rows.append(record)
            print(json.dumps({'completed': i + 1, 'total': len(manifest['rows']), 'label': row['label'],
                              'sample': row['sample'], 'component': result.component}), flush=True)
            if result.component == 'harness_error':
                raise RuntimeError('Replay measurement failed; preserve row and stop without model-success claim')
    report = ['# Measurement-repair replay', '', 'Original scores remain immutable; this is a separately versioned replay of saved programs.',
              'Original model/input seeds and four checks retained. Storage handling and NaN persistence semantics changed.',
              'This is not a new-task evaluation or a strict custom-kernel result.', '',
              '| Policy | Original correct / 8 | Replay correct / 8 |', '|---|---:|---:|']
    for label in sorted({r['label'] for r in rows}):
        part = [r for r in rows if r['label'] == label]
        report.append(f"| {label} | {sum(r['original']['correct'] for r in part)} | {sum(r['replay']['correct'] for r in part)} |")
    (out / 'REPORT.md').write_text('\n'.join(report) + '\n')
    (out / 'complete.json').write_text(json.dumps({'status': 'complete', 'utc': utc(), 'programs': len(rows),
        'manifest_sha256': sha256(manifest_path), 'scores_sha256': sha256(out / 'scores.jsonl'),
        'report_sha256': sha256(out / 'REPORT.md')}, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['seal', 'controls', 'replay'])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--controls', type=Path)
    args = p.parse_args()
    if args.action == 'seal':
        seal(args.manifest)
    elif args.action == 'controls':
        controls(args.manifest, args.out)
    else:
        replay(args.manifest, args.controls, args.out)


if __name__ == '__main__':
    main()
