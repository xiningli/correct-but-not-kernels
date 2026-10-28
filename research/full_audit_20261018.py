"""Read-only project audit. Never import generated programs or rewrite old results.

Writes inventories and freshly reconstructed evidence only beneath --out.
This module is outside every completed experiment's frozen source namespace.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import statistics
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
MAIN_ARCHIVES = {
    'runs/autodl-pilot-backup/20261005T035943.378043Z-reconnected-20261005T162218Z/evidence-and-checkpoints.tgz':
        'd58e5119237fe80eeb3553d3d9fe800f30ce75f6ad6d0509bba7d28bff1e96fc',
    'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/evidence-and-checkpoints.tgz':
        '8cc3de5ecadc11e6ce5339dacba4bafc702e0f79d4892f57f927cb07b9f01e74',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha_records(value):
    if isinstance(value, dict):
        if isinstance(value.get('sha256'), str):
            yield {k: value[k] for k in ('sha256', 'bytes', 'path') if k in value}
        for v in value.values():
            yield from sha_records(v)
    elif isinstance(value, list):
        for v in value:
            yield from sha_records(v)


def assets(out):
    rows = []
    by_root = defaultdict(lambda: {'files': 0, 'bytes': 0})
    archives = []
    for dirname, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '.venv', '.cache', '__pycache__', '.pytest_cache'}
                         and not (Path(dirname) / d).is_symlink()
                         and not (Path(dirname) / d).resolve().is_relative_to(out.resolve()))
        for name in sorted(files):
            p = Path(dirname) / name
            if p.is_symlink():
                continue
            s = p.stat(); rel = str(p.relative_to(ROOT))
            rows.append({'path': rel, 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns})
            group = '/'.join(p.relative_to(ROOT).parts[:2]) if rel.startswith('runs/') else p.relative_to(ROOT).parts[0]
            by_root[group]['files'] += 1; by_root[group]['bytes'] += s.st_size
            if name.endswith(('.tgz', '.tar.gz')):
                archives.append(p)
    with (out / 'file-inventory.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    save(out / 'inventory-summary.json', {'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
         'by_root': dict(sorted(by_root.items())), 'excluded': ['.git', '.venv', '.cache', '__pycache__', '.pytest_cache', 'symlinks', str(out)]})
    results = []
    for i, p in enumerate(sorted(archives)):
        rel = str(p.relative_to(ROOT)); sha = digest(p)
        candidates = []
        for receipt in sorted(p.parent.glob('*.json')):
            if not any(x in receipt.name for x in ('archive', 'verified', 'verification', 'receipt')):
                continue
            try:
                for record in sha_records(json.loads(receipt.read_text())):
                    candidates.append({'receipt': str(receipt.relative_to(ROOT)), **record})
            except (ValueError, OSError):
                continue
        matches = [r for r in candidates if r['sha256'] == sha and r.get('bytes', p.stat().st_size) == p.stat().st_size]
        row = {'path': rel, 'bytes': p.stat().st_size, 'sha256': sha, 'matching_receipts': matches,
               'expected_main_archive_sha256': MAIN_ARCHIVES.get(rel),
               'main_archive_hash_match': sha == MAIN_ARCHIVES[rel] if rel in MAIN_ARCHIVES else None}
        if rel in MAIN_ARCHIVES and not row['main_archive_hash_match']:
            raise ValueError('Main archive checksum mismatch: ' + rel)
        results.append(row)
        save(out / 'archive-hashes.json', results)
        print(json.dumps({'phase': 'archive_hash', 'done': i + 1, 'total': len(archives), 'path': rel}), flush=True)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    save(out / 'tracked-files.json', {p: digest(ROOT / p) for p in tracked if p and (ROOT / p).is_file()})
    return {'files': len(rows), 'archives': len(results), 'archive_bytes': sum(r['bytes'] for r in results),
            'archives_matching_historical_receipts': sum(bool(r['matching_receipts']) for r in results)}


def archive_members(out):
    results = []
    for index, (rel, expected) in enumerate(MAIN_ARCHIVES.items()):
        target = out / ('pilot-archive' if index == 0 else 'replication-archive')
        target.mkdir(exist_ok=True)
        counts = Counter(); checkpoints = []; duplicates = []; seen = set(); total = 0
        last = time.monotonic()
        with tarfile.open(ROOT / rel, 'r|gz') as tar, (target / 'members.jsonl').open('w') as stream:
            for member in tar:
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe archived member')
                if member.name in seen:
                    duplicates.append(member.name)
                seen.add(member.name)
                counts['files' if member.isfile() else 'links' if member.issym() or member.islnk() else 'other'] += 1
                record = {'path': member.name, 'bytes': member.size, 'type': member.type.decode(errors='replace')}
                # Read every regular member to verify decompression and record a digest.
                if member.isfile():
                    total += member.size
                    source = tar.extractfile(member)
                    h = hashlib.sha256()
                    capture = (member.size < 200_000_000 and path.name in {
                        'REPORT.md', 'REPORT.json', 'outcome.json', 'manifest.json', 'training.json', 'kill.json',
                        'final-policy.json', 'metrics.jsonl', 'trainer_state.json', 'config.yaml', 'frozen.json',
                        'plan.yaml', 'selection.json', 'source-seal.json', 'complete.json', 'request.json'})
                    dest = target / 'selected' / member.name
                    if capture:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                    output = dest.open('wb') if capture else None
                    try:
                        for block in iter(lambda: source.read(8 << 20), b''):
                            h.update(block)
                            if output: output.write(block)
                    finally:
                        if output: output.close()
                    record['sha256'] = h.hexdigest()
                    if path.name == 'adapter_model.safetensors':
                        checkpoints.append(record)
                stream.write(json.dumps(record) + '\n')
                if time.monotonic() - last > 20:
                    print(json.dumps({'phase': 'archive_members', 'archive': rel, 'members': sum(counts.values()), 'bytes': total}), flush=True)
                    last = time.monotonic()
        result = {'path': rel, 'member_counts': dict(counts), 'uncompressed_regular_bytes': total,
                  'duplicate_member_names': duplicates, 'adapter_checkpoints': checkpoints}
        results.append(result); save(out / 'main-archive-members.json', results)
        print(json.dumps({'phase': 'archive_members_complete', 'archive': rel, 'members': sum(counts.values()),
                          'adapter_checkpoints': len(checkpoints), 'duplicate_names': len(duplicates)}), flush=True)
    return results


def studies(out):
    from tools.replication import verify_frozen
    from env.tasks import verified_catalog, split_ids
    from eval.replication import validate_rows
    from eval.replication_report import load_complete, task_rates, paired_interval
    from research.reliability_v2.task_inference import centered_bootstrap_test
    from research.reliability_v2.development_report import report as transfer_report
    from research.reliability_v2.capability_run import report as capability_report
    from research.diagnostics_v1.report import report as diagnostic_report
    from research.diagnostics_v2.report import report as iterative_report

    plan, frozen = verify_frozen(ROOT / 'runs/replication-v1.1')
    catalog = {t.task_id: t for t in verified_catalog()}
    training, heldout = split_ids()
    receipt_path = ROOT / 'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/local-analysis-verification.json'
    receipt = json.loads(receipt_path.read_text())
    policies = {}; grids = defaultdict(list); saved_rows = {}
    for label, record in sorted(receipt['policies'].items()):
        p = ROOT / record['path']
        assert digest(p / 'scores.jsonl') == record['scores_sha256']
        assert digest(p / 'generations.jsonl') == record['generations_sha256']
        rows = load_complete(p / 'scores.jsonl')
        request = json.loads((p / 'request.json').read_text())
        generations = [json.loads(line) for line in (p / 'generations.jsonl').read_text().splitlines()]
        validate_rows(generations, request)
        index = {(r['task'], r['sample']): r for r in generations}
        for r in rows:
            assert r['source_sha256'] == catalog[r['task']].sha256
            g = index[(r['task'], r['sample'])]
            assert all(r[k] == v for k, v in g.items()), (label, r['task'], r['sample'])
        saved_rows[label] = rows
        levels = {}
        for level in ['level1', 'level2']:
            subset = [r for r in rows if r['task'].startswith(level + '/')]
            levels[level] = {'programs': len(subset), 'tasks': len({r['task'] for r in subset}),
                'correct': sum(r['result']['correct'] for r in subset),
                'fast1': sum(r['result']['correct'] and r['result']['speedup'] > 1 for r in subset),
                'fast1_1': sum(r['result']['correct'] and r['result']['speedup'] > 1.1 for r in subset),
                'fast2': sum(r['result']['correct'] and r['result']['speedup'] > 2 for r in subset)}
        grid = task_rates([r for r in rows if r['task'].startswith('level2/')])
        grids[label.split('-')[0]].append(grid)
        policies[label] = {'levels': levels, 'correct': sum(r['result']['correct'] for r in rows),
                           'components': dict(Counter(r['result']['component'] for r in rows)),
                           'adapter_sha256': request.get('adapter_sha256'),
                           'adapter': request.get('adapter'),
                           'scores_sha256': digest(p / 'scores.jsonl'),
                           'generation_score_rows_identical': True}
    arms = {arm: {t: statistics.fmean(g[t] for g in values) for t in values[0]} for arm, values in grids.items()}
    comparisons = {}; saved_test = json.loads((ROOT / 'runs/reliability-v2/task-inference-20261014/analysis.json').read_text())
    for left, right in plan['analysis']['primary_comparisons']:
        key = left + '-' + right + '/level2'
        result = paired_interval(arms[left], arms[right], seed=plan['analysis']['seed'],
              repeats=plan['analysis']['bootstrap_repeats'], alpha=plan['analysis']['familywise_alpha'] / 3)
        assert result == receipt['primary_comparisons'][key], key
        test = centered_bootstrap_test([arms[left][t] - arms[right][t] for t in sorted(arms[left])], repeats=100000, seed=20261014)
        assert all(saved_test['comparisons'][key][k] == v for k, v in test.items()), key
        comparisons[key] = {'interval': result, 'post_hoc_test': test}
        print(json.dumps({'phase': 'reconstruct_primary', 'comparison': key, 'difference': result['value'], 'p': test['p_raw']}), flush=True)
    result = {'frozen_files': len(frozen['source_sha256']), 'train_split_tasks': len(training),
              'heldout_tasks': len(heldout), 'actual_training_tasks': plan['train_tasks'],
              'final_programs': sum(len(x) for x in saved_rows.values()), 'policies': policies,
              'arm_level2_rates': {a: statistics.fmean(g.values()) for a, g in arms.items()},
              'primary_comparisons': comparisons, 'studies': {}}
    save(out / 'study-reconstruction.json', result)
    specs = [
        ('transfer_v3', 'runs/reliability-v2/transfer-development-recovery/20261013T200847.197395Z/evidence.tgz', '/pilot/', transfer_report, 'metrics.jsonl', 36),
        ('interface_v4', 'runs/reliability-v2/capability-recovery/20261015T185011.215170Z/interface-snapshot/interface-complete.tgz', '/interface/', capability_report, 'metrics.jsonl', 84),
        ('strict_replay_v4', 'runs/reliability-v2/capability-recovery/20261015T185011.215170Z/evidence.tgz', '/decomposition/', capability_report, 'metrics.jsonl', 5600),
        ('diagnostics_v5', 'runs/diagnostics-v5/recovery/2026-10-16T051325.446441_0000/evidence.tgz', '/repairs/', diagnostic_report, 'metrics.jsonl', 84),
        ('iterative_12b_v6', 'runs/diagnostics-v6/recovery/2026-10-16T222119.856430_0000/evidence.tgz', '/repairs/', iterative_report, 'nodes.jsonl', 37),
        ('iterative_31b_v6', 'runs/diagnostics-v6-resume31b/recovery/2026-10-17T204539.738896_0000/evidence.tgz', '/repairs/', iterative_report, 'nodes.jsonl', 22),
    ]
    for name, archive, suffix, regenerate, metric, expected_n in specs:
        with tarfile.open(ROOT / archive) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            prefix, = [n.removesuffix(metric) for n in members if n.endswith(suffix + metric)]
            def read(n): return tar.extractfile(members[n]).read()
            target = out / 'reconstructed' / name; target.mkdir(parents=True, exist_ok=True)
            for key in members:
                if key.startswith(prefix) and PurePosixPath(key).name in {'metrics.jsonl', 'nodes.jsonl', 'episodes.jsonl', 'outcome.json', 'REPORT.md', 'REPORT.json', 'cost.json'}:
                    rel = PurePosixPath(key[len(prefix):])
                    assert not rel.is_absolute() and '..' not in rel.parts
                    dest = target / str(rel); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(read(key))
            originals = {n: read(prefix + n) for n in ['REPORT.md', 'REPORT.json']}
            seals = [key for key in members if key.endswith('source-seal.json') and (key.startswith(prefix) or prefix.startswith(str(PurePosixPath(key).parent) + '/'))]
            verified_seals = {}
            for key in seals:
                seal = json.loads(read(key))
                for file, sha in seal.items():
                    assert hashlib.sha256(read(file)).hexdigest() == sha, (name, file)
                    assert digest(ROOT / file) == sha, (name, file)
                verified_seals[key] = len(seal)
            data = regenerate(target)
            for n in originals:
                assert (target / n).read_bytes() == originals[n], (name, n)
            rows = [json.loads(line) for line in (target / metric).read_text().splitlines()]
            assert len(rows) == expected_n, name
            record = {'records': len(rows), 'archive': archive, 'archive_sha256': digest(ROOT / archive),
                      'reports_reproduced_byte_identically': True, 'source_seals': verified_seals,
                      'metrics_sha256': digest(target / metric), 'report': data}
            if name == 'strict_replay_v4':
                original = {(label, r['task'], r['sample']): r for label, values in saved_rows.items() for r in values}
                for r in rows:
                    old = original[(r['policy'], r['task'], r['sample'])]
                    from env.reward import extract_program
                    source = extract_program(old['text'])
                    assert r['candidate_sha256'] == (hashlib.sha256(source.encode()).hexdigest() if source else None)
                    assert r['input_seed'] == old['input_seed']
                    assert r['original'] == {'correct': old['result']['correct'], 'component': old['result']['component']}
                record['all_original_sources_seeds_scores_match'] = True
                record['components'] = dict(Counter(r['result']['component'] for r in rows))
                record['strict_successes'] = sum(r['result']['correct'] for r in rows)
                record['numerical_observations'] = dict(Counter(str(r['result']['numerically_correct']) for r in rows))
                record['contract_observations'] = dict(Counter(str(r['result']['contract_compliant']) for r in rows))
            if metric == 'nodes.jsonl':
                episodes = [json.loads(line) for line in (target / 'episodes.jsonl').read_text().splitlines()]
                index = {r['id']: r for r in rows}; assert len(index) == len(rows)
                first = defaultdict(set)
                for e in episodes:
                    assert len(e['trace']) == e['attempts_used'] <= 4
                    assert e['selected'] == e['trace'][-1]
                    assert not any(index[n]['visible_pass'] for n in e['trace'][:-1])
                    assert e['visible_pass'] == index[e['selected']]['visible_pass']
                    assert e['visible_pass'] or len(e['trace']) == 4
                    assert e['resolved'] == bool(e['visible_pass'] and e['withheld'] and e['withheld']['correct'])
                    first[(e['task'], e['replicate'])].add(e['trace'][0])
                assert len(episodes) == 12 and all(len(v) == 1 for v in first.values())
                record.update(episodes=len(episodes), logical_successes=sum(e['resolved'] for e in episodes),
                    distinct_successful_programs=len({e['selected'] for e in episodes if e['resolved']}),
                    physical_components=dict(Counter(n['component'] for n in rows)),
                    physical_prompt_tokens=sum(n['prompt_tokens'] for n in rows),
                    physical_generated_tokens=sum(n['new_tokens'] for n in rows),
                    first_candidate_shared_correctly=True, stopping_and_hidden_selection_verified=True)
            result['studies'][name] = record
            save(out / 'study-reconstruction.json', result)
            print(json.dumps({'phase': 'study_reconstructed', 'study': name, 'records': len(rows), 'report_equal': True}), flush=True)
    return result


def legacy(out):
    from eval.report import bootstrap, paired_change, measurements
    from eval.replication_report import load_complete, task_rates, paired_interval, common_correct_speed
    from tools.replication import verify_frozen
    from env.tasks import split_ids
    from env.reward import extract_program
    import yaml

    plan, _ = verify_frozen(ROOT / 'runs/replication-v1.1')
    _, expected_tasks = split_ids()
    pilot_root = ROOT / 'runs/c6/20261005T035943.498204Z'
    pilot_old = json.loads((pilot_root / 'REPORT.json').read_text())
    pilot_rows = {}; result = {'pilot': {}, 'reevaluation': {}, 'training': {}, 'storage_replay': {}}
    for label in pilot_old:
        actual = {'C': 'C-step-35', 'S': 'S-step-40'}.get(label, label)
        p = pilot_root / 'heldout' / actual / 'scores.jsonl'
        rows = [json.loads(s) for s in p.read_text().splitlines()]
        assert len(rows) == 120 and {r['task'] for r in rows} == expected_tasks
        pilot_rows[label] = rows
        levels = {}
        for level in ['level1', 'level2']:
            values = [r for r in rows if r['task'].startswith(level + '/')]
            calculated = bootstrap(values)
            assert calculated == pilot_old[label]['levels'][level]
            levels[level] = measurements(values)
        result['pilot'][label] = {'programs': len(rows), 'levels': levels, 'all_metric_intervals_reproduced': True}
    for label in ['C', 'S']:
        for level in ['level1', 'level2']:
            change = paired_change(*[[r for r in pilot_rows[l] if r['task'].startswith(level + '/')] for l in ['base', label]])
            assert change == pilot_old[label]['paired_correctness_change'][level]
    r2_root = ROOT / 'runs/autodl-westb-reboot-20261006/r2-complete/extracted/runs/replication-v1.1/reevaluation'
    r2_old = json.loads((r2_root / 'REPORT.json').read_text()); r2_rows = {}
    for label in ['base', 'C', 'S']:
        rows = load_complete(r2_root / label / 'scores.jsonl'); r2_rows[label] = rows
        levels = {}
        for level in ['level1', 'level2']:
            part = [r for r in rows if r['task'].startswith(level + '/')]
            point = measurements(part)
            for key, value in point.items(): assert value == r2_old['policies'][label]['levels'][level][key]['value']
            levels[level] = point
        result['reevaluation'][label] = {'programs': len(rows), 'levels': levels}
    for label in ['C', 'S']:
        for level in ['level1', 'level2']:
            left, right = [[r for r in r2_rows[l] if r['task'].startswith(level + '/')] for l in [label, 'base']]
            interval = paired_interval(task_rates(left), task_rates(right), seed=plan['analysis']['seed'],
                        repeats=plan['analysis']['bootstrap_repeats'], alpha=.05 / 3 if level == 'level2' else .05)
            assert interval == r2_old['comparisons'][label + '-base/' + level]['paired']
            assert common_correct_speed(left, right) == r2_old['common_correct_speed'][label + '-base/' + level]
    result['reevaluation_primary_intervals_and_common_correct_speed_reproduced'] = True
    print(json.dumps({'phase': 'legacy', 'pilot_programs': 1080, 'r2_programs': 2880, 'r2_base_reused_in_r3': 960}), flush=True)

    member_index = {}
    for label in ['pilot-archive', 'replication-archive']:
        member_index[label] = {r['path']: r for line in (out / label / 'members.jsonl').read_text().splitlines()
                               if (r := json.loads(line)).get('sha256')}
    index = member_index['replication-archive']
    selected = out / 'replication-archive/selected'
    saved = json.loads((out / 'study-reconstruction.json').read_text())['policies']
    for label, record in saved.items():
        if label == 'base': continue
        adapter_path = record['adapter'] + '/adapter_model.safetensors'
        assert index[adapter_path]['sha256'] == record['adapter_sha256']
        train = selected / 'runs/replication-v1.1/training' / label
        cfg = yaml.safe_load((train / 'config.yaml').read_text())
        assert cfg['tasks'] == ['level1/40_LayerNorm'] and cfg['model'] == 'gemma-4-12B-it'
        rows = [json.loads(s) for s in (train / 'metrics.jsonl').read_text().splitlines()]
        final_step = max(r.get('step', 0) for r in rows)
        assert final_step == (15 if label == 'S-20261021' else 40)
        data = {'method': cfg['method'], 'tasks': cfg['tasks'], 'final_step': final_step,
                'adapter_hash_matches_final_evaluation': True, 'adapter_sha256': record['adapter_sha256'],
                'elapsed_training_seconds': max(r.get('elapsed_seconds', 0) for r in rows),
                'stop': json.loads((train / 'kill.json').read_text())['reason'] if (train / 'kill.json').exists() else None}
        if label.startswith('SFT-'):
            manifest_path = Path('runs/replication-v1.1/confirmation') / label / 'data/dataset/manifest.json'
            m = json.loads((selected / manifest_path).read_text())
            assert index[str(manifest_path.parent / 'train.jsonl')]['sha256'] == m['data_sha256']
            request_path = manifest_path.parent.parent / 'candidates/request.json'
            request = json.loads((selected / request_path).read_text())
            assert request['adapter'] is None and request['split'] == 'train' and request['tasks'] == cfg['tasks']
            assert request['model'] == cfg['model']
            data['sft_dataset'] = m
            data['sft_generator'] = 'same Gemma 12B base; no teacher adapter'
        result['training'][label] = data
    result['sft_tuning'] = json.loads((selected / 'runs/replication-v1.1/sft-tuning/selection.json').read_text())

    receipt = json.loads((ROOT / 'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/local-analysis-verification.json').read_text())
    assert digest(r2_root / 'base/scores.jsonl') == digest(ROOT / receipt['policies']['base']['path'] / 'scores.jsonl')
    result['r2_base_rows_reused_in_r3'] = 960
    for label, info in receipt['policies'].items():
        local = ROOT / info['path']
        canonical = info['path'].split('/extracted/', 1)[1]
        for filename in ['scores.jsonl', 'generations.jsonl', 'request.json', 'complete.json']:
            assert digest(local / filename) == index[canonical + '/' + filename]['sha256'], (label, filename)
    for label in pilot_old:
        actual = {'C': 'C-step-35', 'S': 'S-step-40'}.get(label, label)
        canonical = 'runs/c6/20261005T035943.498204Z/heldout/' + actual + '/scores.jsonl'
        assert digest(ROOT / canonical) == member_index['pilot-archive'][canonical]['sha256']
    result['local_final_grids_match_complete_archives'] = True

    replay_root = ROOT / 'runs/reliability-v2/remote-recovery-20261012/20261012T222455.458848Z/extracted'
    p, = list(replay_root.glob('runs/reliability-v2/controllers/*/replay/scores.jsonl'))
    rows = [json.loads(s) for s in p.read_text().splitlines()]
    done = json.loads(p.with_name('complete.json').read_text())
    assert digest(p) == done['scores_sha256'] and len(rows) == 80
    assert digest(p.with_name('REPORT.md')) == done['report_sha256']
    keys = {(r['label'], r['sample']) for r in rows}
    assert len(keys) == 80 and keys == {(label, sample) for label in receipt['policies'] for sample in range(8)}
    originals = {}
    for label, info in receipt['policies'].items():
        for line in (ROOT / info['path'] / 'scores.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r['task'] == 'level1/38_L1Norm_': originals[(label, r['sample'])] = r
    changed = []
    for r in rows:
        old = originals[(r['label'], r['sample'])]; source = extract_program(old['text'])
        assert r['candidate_sha256'] == (hashlib.sha256(source.encode()).hexdigest() if source else None)
        assert r['input_seed'] == old['input_seed'] and r['original']['correct'] == old['result']['correct']
        if r['original']['component'] != r['replay']['component']:
            changed.append({'label': r['label'], 'sample': r['sample'], 'from': r['original']['component'], 'to': r['replay']['component']})
    assert len(changed) == 21 and all(r['from'] == 'crash' and r['to'] == 'correct' for r in changed)
    result['storage_replay'] = {'programs': 80, 'original_correct': sum(r['original']['correct'] for r in rows),
              'replay_correct': sum(r['replay']['correct'] for r in rows), 'changed': changed,
              'same_candidate_and_seed_verified': True, 'original_scores_unchanged': True}
    save(out / 'legacy-and-training.json', result)
    return result


def probes(out):
    """Parse retained sources; run only a manually transcribed CPU expression."""
    import torch
    from env.reward import _FENCE, _THINK
    from research.reliability_v2.fixed_interface import extract_code
    from research.diagnostics_v1.tasks import public_task
    from research.scoped_repair.edits import build_scope, validate_candidate, EditError

    path = out / 'reconstructed/interface_v4/metrics.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    failures = []
    for r in rows:
        if r['condition'] != 'full_program' or extract_code(r['text']) is not None:
            continue
        blocks = _FENCE.findall(_THINK.sub('', r['text']))
        reason = 'no_fence'
        line = None
        if blocks:
            try:
                ast.parse(blocks[-1])
                reason = 'unclosed_thought_or_other_extraction_failure'
            except SyntaxError as error:
                reason = error.msg
                line = error.text.strip() if error.text else None
        failures.append({k: r[k] for k in ['policy', 'task', 'condition', 'finish_reason']} |
                        {'reason': reason, 'syntax_error_line': line})
    capped = [{k: r[k] for k in ['policy', 'task', 'condition', 'new_tokens', 'finish_reason']} |
              {'component': r['result']['component'], 'correct': r['result']['correct']}
              for r in rows if r['finish_reason'] != 'stop']
    formats = {'counts': dict(Counter(r['reason'] for r in failures)), 'rows': failures,
               'full_program_format_contract_rejections': [
                   {k: r[k] for k in ['policy', 'task']} |
                   {'reason': r['result']['format_reason'],
                    'ModelNew_classes': sum(isinstance(n, ast.ClassDef) and n.name == 'ModelNew'
                                           for n in ast.parse(extract_code(r['text'])).body)}
                   for r in rows if r['condition'] == 'full_program' and r['result'].get('format_reason')],
               'length_capped_rows': capped,
               'success_if_non_stop_outputs_counted_as_failure': {
                   condition: sum(r['condition'] == condition and r['result']['correct'] and
                                  r['finish_reason'] == 'stop' for r in rows)
                   for condition in ['full_program', 'fixed_host']},
               'scope': 'static parse and descriptive sensitivity only; no execution or historical score changes'}
    save(out / 'interface-format-diagnosis.json', formats)

    nodes = [json.loads(s) for s in (out / 'reconstructed/iterative_31b_v6/nodes.jsonl').read_text().splitlines()]
    node, = [r for r in nodes if r['task'] == 'development/case-001' and r['component'] == 'correct']
    source = node['source']
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    assert source_sha == '35c80af2bdc1d47c0a750b010f167e396928b974656383f7940387ef7b02cae6'
    assert 'return (tl.exp(x) - tl.exp(-x)) / (tl.exp(x) + tl.exp(-x))' in source
    x = torch.tensor([-100., -90., -80., 0., 80., 90., 100.], dtype=torch.float32, device='cpu')
    y = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    scope = build_scope(public_task('development/case-001')['source'])
    error = None
    try:
        validate_candidate(scope, source)
    except EditError as exception:
        error = str(exception)
    numeric = {'candidate_sha256': source_sha,
        'scope': 'trusted manually transcribed arithmetic on CPU; candidate source was not imported/executed; no GPU verdict or retrospective score change',
        'torch': torch.__version__, 'inputs': x.tolist(),
        'formula_outputs': [float(v) if torch.isfinite(v) else 'nonfinite' for v in y],
        'torch_tanh': torch.tanh(x).tolist(), 'existing_v7_accepts_saved_success': error is None,
        'v7_rejection': error}
    save(out / 'numerical-and-scope-probe.json', numeric)
    return {'interface': formats, 'numerical_and_scope': numeric}


def final_report(out):
    from eval.replication_report import report
    from tools.replication import verify_frozen
    plan, _ = verify_frozen(ROOT / 'runs/replication-v1.1')
    receipt = json.loads((ROOT / 'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/local-analysis-verification.json').read_text())
    original = out / 'replication-archive/selected/runs/replication-v1.1/confirmation/REPORT.json'
    old = json.loads(original.read_text())
    manifest = [{'label': label, 'arm': cell['arm'], 'training_seed': cell['training_seed'],
                 'scores': receipt['policies'][label]['path'] + '/scores.jsonl'}
                for label, cell in old['policies'].items()]
    rebuilt = report(plan, manifest)
    assert rebuilt == old, 'Final confirmation report differs from raw-row reconstruction'
    save(out / 'confirmation-report.json', rebuilt)
    return {'all_fields_match_archived_report': True, 'policies': len(manifest),
            'archived_report_sha256': digest(original), 'reconstructed_report_sha256': digest(out / 'confirmation-report.json'),
            'scope': 'Exact parsed JSON equality, including all per-policy intervals, taxonomy, comparisons and common-correct speed ratios'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['assets', 'archive-members', 'studies', 'legacy', 'probes', 'final-report'])
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    result = {'assets': assets, 'archive-members': archive_members, 'studies': studies, 'legacy': legacy, 'probes': probes, 'final-report': final_report}[args.phase](out)
    save(out / (args.phase + '-completion.json'), {'utc': datetime.now(timezone.utc).isoformat(), 'result': result})
    print(json.dumps({'phase': args.phase, 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
