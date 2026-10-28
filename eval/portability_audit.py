"""K1/K4: verify completed frozen-program replication and summarize its scope."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

from research.portability_v1.common import load, read, sha, write
from research.portability_v1.report import aggregate, effects


def audit(prepared, run, out):
    manifest, plan = load(prepared)
    saved = read(run / 'REPORT.json')
    outcome = read(run / 'outcome.json')
    archive = read(run / 'archive.json')
    assert outcome['status'] == 'complete' and saved['outcome'] == outcome
    assert sha(archive['path']) == archive['sha256']
    assert read(run / 'binding.json')['preparation_sha256'] == sha(prepared / 'manifest.json')
    rows = [json.loads(s) for s in (run / 'metrics.jsonl').read_text().splitlines()]
    frozen = [json.loads(s) for s in (prepared / 'programs.jsonl').read_text().splitlines()]
    key = lambda r: (r['task'], r['policy'], r['sample'])
    source = {key(r): r for r in frozen}
    for row in rows:
        assert all(row[k] == v for k, v in source[key(row)].items())
    eligibility = read(run / 'eligibility.json')
    assert set(eligibility) == {r['task'] for r in frozen}
    eligible = {t for t, r in eligibility.items() if r['eligible']}
    complete, counts = aggregate(rows, plan['policies'], plan['samples'])
    assert set(complete) == eligible
    assert len(rows) == len(eligible) * len(plan['policies']) * plan['samples'] == outcome['attempts']
    assert counts == saved['policies'] and effects(complete, plan) == saved['effects']
    families = {}
    for family in ['base', 'C', 'SFT']:
        members = [r for r in rows if (r['policy'] == family or r['policy'].startswith(family + '-'))]
        values = {'attempts': len(members)}
        for location, label in [('result', 'local'), ('remote', 'remote')]:
            values[label + '_pass'] = sum(r[location]['numerically_correct'] is True for r in members)
            values[label + '_unknown'] = sum(r[location]['numerically_correct'] is None for r in members)
            values[label + '_pass_rate'] = values[label + '_pass'] / len(members)
        families[family] = values
    discordant = []
    for r in rows:
        if r['remote']['numerically_correct'] != r['result']['numerically_correct']:
            discordant.append({k: r[k] for k in ['policy', 'task', 'sample', 'candidate_sha256', 'remote']} | {
                'local': {k: r['result'].get(k) for k in ['component', 'numerically_correct', 'checks']},
                'candidate_error': r['result'].get('runtime', {}).get('candidate_error')})
    receipt = {
        'verification': 'PASS', 'source': str(run), 'finished_utc': outcome['utc'],
        'hours': (datetime.fromisoformat(outcome['utc']) - datetime.fromisoformat(read(run / 'process.json')['utc'])).total_seconds() / 3600,
        'preparation_sha256': sha(prepared / 'manifest.json'), 'sealed_files_verified': len(manifest['files']),
        'metrics_sha256': sha(run / 'metrics.jsonl'), 'report_sha256': sha(run / 'REPORT.json'),
        'archive': archive, 'attempts': len(rows), 'complete_tasks': len(complete),
        'excluded_tasks': len(eligibility) - len(complete),
        'exclusion_counts': dict(Counter(r['error'] for r in eligibility.values() if not r['eligible'])),
        'families': families, 'effects': saved['effects'],
        'strict_pass': sum(r['result']['correct'] is True for r in rows),
        'numeric_transitions': dict(Counter(f"{r['remote']['numerically_correct']}->{r['result']['numerically_correct']}" for r in rows)),
        'discordant': discordant,
        'limits': plan['claims'],
        'diagnosis': 'Discordant logs are observations; no additional GPU reproducer or causal mechanism established.',
    }
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'audit.json', receipt)
    lines = ['# Consumer GPU replay: verified completion', '',
             f"Completed {len(rows):,} attempts on {len(complete)} common tasks in {receipt['hours']:.2f} hours.",
             f"Excluded {receipt['excluded_tasks']} tasks by frozen reference/storage limits: {receipt['exclusion_counts']}.",
             '', '| Policy family | Attempts | RTX PRO 6000 pass | RTX 5070 Ti pass | Local unknown |',
             '|---|---:|---:|---:|---:|']
    for family, r in families.items():
        lines.append(f"| {family} | {r['attempts']} | {r['remote_pass_rate']:.2%} | {r['local_pass_rate']:.2%} | {r['local_unknown']} |")
    lines += ['', f'Same {len(complete)} task grids on both platforms. Unknowns remain in denominators.',
              'C and SFT each average the three equally sized training-seed evaluations.', '',
              '| Local contrast | Percentage-point change | 98.333% task-bootstrap interval |',
              '|---|---:|---:|']
    for a, b in plan['comparisons']:
        e = saved['effects'][f'{a}-{b}/local']
        lines.append(f"| {a} − {b} | {100*e['estimate']:+.2f} | [{100*e['interval'][0]:+.2f}, {100*e['interval'][1]:+.2f}] |")
    lines += ['', f"Strict custom-kernel successes: {receipt['strict_pass']} / {len(rows)}.",
              f"Numerical three-state disagreements: {len(discordant)}; details are in audit.json.",
              'The pass gains persist on this common resource-qualified subset. This is same-architecture platform replication.',
              'It does not establish new-task transfer, acceleration, improved custom-kernel skill or RL superiority to SFT.',
              f"The {receipt['excluded_tasks']} exclusions reflect temporary/output-storage policy; they do not establish that these tasks cannot run on a 16GB GPU.",
              'No new generation, training or GPU rerun was used in this audit.', '',
              'Metrics, input bindings, summary counts, bootstrap effects and terminal archive hash verified.',
              'Reproduce with `python3 -m eval.portability_audit --prepared PREPARED --run RUN --out NEW_DIRECTORY`.']
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    write(out / 'checksums.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k: receipt[k] for k in ['verification', 'attempts', 'complete_tasks', 'excluded_tasks', 'families', 'strict_pass']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'run', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.prepared, args.run, args.out)
