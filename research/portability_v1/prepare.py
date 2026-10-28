"""Freeze all seven policies, including failures, before second-device outcomes."""
import argparse
import json
from pathlib import Path
from env.reward import extract_program
from env.tasks import verified_catalog
from .common import ROOT, PLAN, RECEIPT, REMOTE, sha, text_sha, write, read, code_files, utc


def prepare(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    plan = read(PLAN)
    frozen = read(ROOT / 'runs/replication-v1.1/frozen.json')
    # The legacy guard also rejects newly added scripts from later studies.
    # Verify every originally sealed file; do not rewrite that seal or allow
    # modifications to any of its members.
    for name, digest in frozen['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Historical source changed: ' + name)
    if sha(ROOT / 'runs/replication-v1.1/plan.yaml') != frozen['plan_sha256']:
        raise ValueError('Historical plan changed')
    for name, digest in frozen['generated_sha256'].items():
        if sha(ROOT / 'runs/replication-v1.1' / name) != digest:
            raise ValueError('Historical job identity changed: ' + name)
    tasks = {t.task_id: t for t in verified_catalog() if t.level == 2}
    if len(tasks) != 100:
        raise ValueError('Expected 100 Level-2 tasks')
    remote = {}
    for line in REMOTE.open():
        r = json.loads(line); key = (r['policy'], r['task'], r['sample'])
        if key in remote:
            raise ValueError('Duplicate remote comparison')
        remote[key] = r
    expected_keys = {(p, t, s) for p in plan['policies'] for t in tasks for s in range(8)}
    if set(remote) != expected_keys:
        raise ValueError('Incomplete remote comparator')
    receipt = read(RECEIPT)
    rows, seeds, inputs = [], {}, {}
    (out / 'programs').mkdir(); (out / 'references').mkdir()
    for policy in plan['policies']:
        meta = receipt['policies'][policy]
        path = ROOT / meta['path'] / 'scores.jsonl'
        if sha(path) != meta['scores_sha256']:
            raise ValueError('Historical score hash mismatch')
        inputs[policy] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
        for line in path.open():
            r = json.loads(line)
            if r['task'] not in tasks:
                continue
            key = (policy, r['task'], r['sample'])
            paired = remote[key]
            source = extract_program(r['text'])
            digest = text_sha(source) if source is not None else None
            if (digest != paired['candidate_sha256'] or
                    r['source_sha256'] != tasks[r['task']].sha256 or
                    r['source_sha256'] != paired['reference_sha256'] or
                    r['input_seed'] != paired['input_seed'] or r['evaluation_trials'] != 4):
                raise ValueError('Program, reference or seed pairing mismatch')
            seed_key = (r['task'], r['sample'])
            if seed_key in seeds and seeds[seed_key] != r['input_seed']:
                raise ValueError('Policies have different evaluation inputs')
            seeds[seed_key] = r['input_seed']
            if source is not None:
                (out / 'programs' / (digest + '.py')).write_text(source)
            result = paired['result']
            rows.append({'policy': policy, 'task': r['task'], 'sample': r['sample'],
                'input_seed': r['input_seed'], 'candidate_sha256': digest,
                'reference_sha256': tasks[r['task']].sha256,
                'historical': {'correct': r['result']['correct'], 'component': r['result']['component']},
                'remote': {k: result.get(k) for k in [
                    'component', 'numerically_correct', 'contract_compliant', 'correct']}})
    if len(rows) != len(expected_keys) or {(r['policy'], r['task'], r['sample']) for r in rows} != expected_keys:
        raise ValueError('Incomplete or duplicate program grid')
    task_rows = []
    for task, t in tasks.items():
        name = text_sha(task) + '.py'
        (out / 'references' / name).write_text(t.source())
        task_rows.append({'task': task, 'file': 'references/' + name, 'sha256': t.sha256,
                          'seeds': [seeds[task, s] for s in range(8)]})
    # Deterministic ordering independent of any success/failure outcome.
    task_rows.sort(key=lambda t: text_sha('portability-v1:' + t['task']))
    order = {t['task']: i for i, t in enumerate(task_rows)}
    rows.sort(key=lambda r: (order[r['task']], r['sample'], plan['policies'].index(r['policy'])))
    (out / 'programs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write(out / 'tasks.json', task_rows); write(out / 'plan.json', plan)
    files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    manifest = {'utc': utc(), 'scope': plan['scope'], 'attempts': len(rows), 'tasks': len(tasks),
                'model_outputs': 0, 'code': code_files(), 'files': files, 'historical_scores': inputs,
                'historical_seal_sha256': sha(ROOT / 'runs/replication-v1.1/frozen.json'),
                'remote_comparator_path': str(REMOTE.relative_to(ROOT)), 'remote_comparator_sha256': sha(REMOTE)}
    write(out / 'manifest.json', manifest)
    return {'prepared': str(out), 'sha256': sha(out / 'manifest.json'),
            'programs': len(rows), 'unique_extractable_sources': len(list((out / 'programs').glob('*.py')))}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().out.resolve()), indent=2))
