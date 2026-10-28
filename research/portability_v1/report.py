"""Paired task-cluster intervals; unknowns and partial tasks are explicit."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .common import read, write


def aggregate(rows, policies, samples=8):
    by_task = {}
    for r in rows:
        task = by_task.setdefault(r['task'], {})
        key = (r['policy'], r['sample'])
        if key in task:
            raise ValueError('Duplicate scored attempt')
        task[key] = r
    expected = {(p, s) for p in policies for s in range(samples)}
    complete = {t: grid for t, grid in by_task.items() if set(grid) == expected}
    counts = {}
    for p in policies:
        subset = [grid[p, s] for grid in complete.values() for s in range(samples)]
        local = [r['result'] for r in subset]; remote = [r['remote'] for r in subset]
        counts[p] = {'attempts': len(subset),
            'local_numeric_pass': sum(r['numerically_correct'] is True for r in local),
            'local_numeric_unknown': sum(r['numerically_correct'] is None for r in local),
            'remote_numeric_pass': sum(r['numerically_correct'] is True for r in remote),
            'remote_numeric_unknown': sum(r['numerically_correct'] is None for r in remote),
            'both_numeric_pass': sum(a['numerically_correct'] is True and b['numerically_correct'] is True
                                     for a, b in zip(local, remote)),
            'remote_pass_local_not_pass': sum(a['numerically_correct'] is not True and b['numerically_correct'] is True
                                             for a, b in zip(local, remote)),
            'local_strict_pass': sum(r['correct'] is True for r in local),
            'components': dict(Counter(r['component'] for r in local))}
    return complete, counts


def effects(complete, plan):
    if not complete:
        return {}
    policies = plan['policies']; samples = plan['samples']
    families = {'base': ['base'], 'C': [p for p in policies if p.startswith('C-')],
                'SFT': [p for p in policies if p.startswith('SFT-')]}
    task_values = {}
    for location in ['remote', 'result']:
        for arm, members in families.items():
            task_values[location, arm] = np.array([
                np.mean([grid[p, s][location]['numerically_correct'] is True
                         for p in members for s in range(samples)])
                for _, grid in sorted(complete.items())])
    rng = np.random.default_rng(plan['bootstrap_seed'])
    indices = rng.integers(len(complete), size=(plan['bootstrap_repeats'], len(complete)))
    alpha = .05 / len(plan['comparisons'])
    result = {}
    for a, b in plan['comparisons']:
        local = task_values['result', a] - task_values['result', b]
        remote = task_values['remote', a] - task_values['remote', b]
        for name, diff in [('local', local), ('remote_same_tasks', remote), ('local_minus_remote_effect', local - remote)]:
            interval = np.quantile(diff[indices].mean(1), [alpha / 2, 1 - alpha / 2])
            result[f'{a}-{b}/{name}'] = {'estimate': float(diff.mean()),
                'interval': interval.tolist(), 'tasks': len(diff), 'confidence': 1 - alpha}
    return result


def report(out, plan):
    out = Path(out); path = out / 'metrics.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    complete, counts = aggregate(rows, plan['policies'], plan['samples'])
    eligibility = read(out / 'eligibility.json') if (out / 'eligibility.json').exists() else {}
    outcome = read(out / 'outcome.json') if (out / 'outcome.json').exists() else {'status': 'running'}
    result = {'outcome': outcome, 'attempts_recorded': len(rows), 'complete_tasks': len(complete),
              'partial_attempts_excluded_from_rates': len(rows) - len(complete) * len(plan['policies']) * plan['samples'],
              'tasks_qualified': sum(r['eligible'] for r in eligibility.values()),
              'tasks_excluded': {t: r for t, r in eligibility.items() if not r['eligible']},
              'policies': counts, 'effects': effects(complete, plan),
              'interval_status': 'final_descriptive' if outcome.get('status') == 'complete' else 'interim_descriptive_no_stopping_decision',
              'limits': plan['claims'], 'model_outputs': 0}
    write(out / 'REPORT.json', result)
    lines = ['# Consumer GPU frozen-program replication', '',
        'Status: ' + outcome['status'] + '. No new generation or training.',
        f"Completed tasks: {len(complete)} / 100; recorded attempts: {len(rows)} / 5600 before reference exclusions.",
        'Rates use complete task grids only. Unknown numerical outcomes remain in operational denominators.',
        '', '| Policy | Attempts | Local numeric pass / unknown | Remote numeric pass / unknown | Pass both | Local strict |',
        '|---|---:|---:|---:|---:|---:|']
    for p, r in counts.items():
        lines.append(f"| {p} | {r['attempts']} | {r['local_numeric_pass']} / {r['local_numeric_unknown']} | "
                     f"{r['remote_numeric_pass']} / {r['remote_numeric_unknown']} | {r['both_numeric_pass']} | {r['local_strict_pass']} |")
    lines += ['', 'Task-cluster intervals and complete reference coverage are in REPORT.json.', '',
              *plan['claims'], '', 'Local workstation remains running; only experiment containers are stopped.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return result
