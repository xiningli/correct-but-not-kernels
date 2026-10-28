"""K1/K4: transparent diagnostic counts and costs; no confirmatory claims."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def report(directory):
    directory = Path(directory)
    path = directory / 'metrics.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row['policy'], row['condition'], row['task'])
        if key in seen:
            raise ValueError('Duplicate development attempt')
        seen.add(key)
        groups[key[:2]].append(row)
    cells = []
    for (policy, condition), values in sorted(groups.items()):
        cells.append({'policy': policy, 'condition': condition, 'attempts': len(values),
            'strict_correct': sum(v['result']['correct'] for v in values),
            'numerically_correct': sum(v['result']['numerically_correct'] for v in values),
            'components': dict(Counter(v['result']['component'] for v in values)),
            'total_generated_tokens': sum(v['new_tokens'] for v in values),
            'mean_evaluation_seconds': statistics.fmean(v['evaluation_seconds'] for v in values)})
    outcome_path = directory / 'outcome.json'
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {'status': 'not_started_or_preflight_failed'}
    data = {'scope': 'six_development_graphs_one_draw_first_training_seed_only',
            'not_confirmatory': True, 'outcome': outcome, 'cells': cells,
            'limitations': ['No new test-graph policy outputs.', 'No real-repair agent result.',
                'First training seed is for integration/cost, never a substitute for the three-seed comparison.',
                'One draw on each of six developer graphs is too small for an effect or mechanism claim.',
                'Profiling wall time is not a kernel-speed benchmark.']}
    (directory / 'REPORT.json').write_text(json.dumps(data, indent=2) + '\n')
    lines = ['# CUDA transfer development pilot', '',
             'Integration and cost measurements only. These are not confirmatory gains or repair-agent results.', '',
             '| Policy | Prompt | Attempts | Strict correct | Numerically correct |',
             '|---|---|---:|---:|---:|']
    lines += [f"| {c['policy']} | {c['condition']} | {c['attempts']} | {c['strict_correct']} | {c['numerically_correct']} |" for c in cells]
    lines += ['', 'Outcome: `' + outcome['status'] + '`.', '', *data['limitations']]
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.directory), indent=2))
