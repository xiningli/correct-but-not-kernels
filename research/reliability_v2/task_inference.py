"""K1/K4: post-hoc, conditional task-cluster inference for sealed v1.1 scores.

Null: mean task-level correctness difference is zero, conditional on saved
checkpoints. A null-centered studentized bootstrap treats paired task vectors
as independent draws from a representative task population. No normal task
distribution or treatment-label exchangeability is assumed. This is an
approximate model-based inference, NOT an exact randomization test, a test of
algorithm superiority across training seeds, or a pre-registered analysis.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys

from eval.replication_report import load_complete, task_rates
from tools.replication import local_path, sha256, verify_frozen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / 'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/local-analysis-verification.json'


def mean_se(values):
    n = len(values)
    mean = statistics.fmean(values)
    se = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / (n * (n - 1)))
    return mean, se


def centered_bootstrap_test(differences, *, repeats=100_000, seed=20261014):
    """Resample tasks, compare |(mean* - mean)/SE*| to |mean/SE|.

    Centering the bootstrap numerator imposes the mean-zero null; ordinary
    uncentered bootstrap draws are not a null distribution. The plus-one tail
    estimate prevents reported zero p-values but does not confer finite-sample
    exactness. Degenerate original samples cannot estimate task uncertainty.
    """
    values = list(map(float, differences))
    if len(values) < 2 or any(not math.isfinite(x) for x in values):
        raise ValueError('Need at least two finite paired task differences')
    if repeats < 1:
        raise ValueError('Need positive bootstrap repeats')
    mean, se = mean_se(values)
    result = {'tasks': len(values), 'mean_difference': mean, 'standard_error': se,
              'repeats': repeats, 'seed': seed, 'alternative': 'two-sided'}
    if se == 0:
        return {**result, 'status': 'degenerate_task_sample_not_testable',
                'p_raw': None, 'observed_t': None}
    observed = mean / se
    rng = random.Random(seed)
    extreme = 0
    degenerate = 0
    for _ in range(repeats):
        sampled = rng.choices(values, k=len(values))
        mean_star, se_star = mean_se(sampled)
        numerator = mean_star - mean
        if se_star == 0:
            # Do not discard resamples or invent variance at a boundary.
            degenerate += 1
            statistic = 0.0 if numerator == 0 else math.inf
        else:
            statistic = abs(numerator / se_star)
        extreme += statistic >= abs(observed) - 1e-12
    p = (extreme + 1) / (repeats + 1)
    return {**result, 'status': 'approximate_conditional_test',
            'observed_t': observed, 'tail_exceedances': extreme,
            'degenerate_resamples': degenerate, 'p_raw': p,
            'p_estimator': '(1 + exceedances) / (1 + repeats)',
            'monte_carlo_resolution': 1 / (repeats + 1),
            'tail_count_zero_warning': extreme == 0}


def aggregate_grids(grids):
    if not grids or not grids[0] or any(set(g) != set(grids[0]) for g in grids):
        raise ValueError('Every saved policy needs the same nonempty task grid')
    return {task: statistics.fmean(g[task] for g in grids) for task in sorted(grids[0])}


def analyze(receipt_path, out, *, repeats, seed):
    plan, frozen = verify_frozen(ROOT / 'runs/replication-v1.1')
    receipt = json.loads(Path(receipt_path).read_text())
    expected = {'base'} | {f'{arm}-{s}' for arm in plan['arms'] for s in plan['train_seeds']}
    if set(receipt['policies']) != expected:
        raise ValueError('Exactly one base and all declared training seeds are required')
    level = plan['analysis']['primary_level']
    by_arm = defaultdict(list)
    inputs = {}
    seed_rates = {}
    for label, record in sorted(receipt['policies'].items()):
        path = local_path(record['path']) / 'scores.jsonl'
        if sha256(path) != record['scores_sha256']:
            raise ValueError('Receipt score hash mismatch: ' + label)
        rows = load_complete(path)
        grid = task_rates([r for r in rows if r['task'].startswith(level + '/')])
        by_arm[label.split('-')[0]].append(grid)
        seed_rates[label] = statistics.fmean(grid.values())
        inputs[label] = {'scores': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
    arms = {arm: aggregate_grids(grids) for arm, grids in by_arm.items()}
    family = plan['analysis']['primary_comparisons']
    comparisons = {}
    task_differences = {}
    for left, right in family:
        if set(arms[left]) != set(arms[right]):
            raise ValueError('Comparison task sets differ')
        key = f'{left}-{right}/{level}'
        differences = {t: arms[left][t] - arms[right][t] for t in sorted(arms[left])}
        measured = statistics.fmean(differences.values())
        original = receipt['primary_comparisons'][key]
        if not math.isclose(measured, original['value'], abs_tol=1e-12):
            raise ValueError('Original estimand changed')
        test = centered_bootstrap_test(differences.values(), repeats=repeats, seed=seed)
        test['p_bonferroni'] = min(1.0, test['p_raw'] * len(family)) if test['p_raw'] is not None else None
        test['original_preregistered_percentile_interval'] = original
        comparisons[key] = test
        task_differences[key] = differences
    result = {
        'status': 'post_hoc_supplement_not_new_confirmatory_evidence',
        'utc': datetime.now(timezone.utc).isoformat(),
        'method': 'null-centered studentized paired-task bootstrap, two-sided',
        'null': 'E_task[mean_saved_policy_correctness_left - mean_saved_policy_correctness_right] = 0',
        'assumptions': [
            'Paired task vectors are modeled as independent, representative task-population draws.',
            'All saved checkpoints are fixed; within-task policy/seed/sample outcomes stay together.',
            'Task differences have finite, nonzero variance; bootstrap validity is approximate.',
        ],
        'limitations': [
            'Tasks were curated and previously observed, not randomly sampled unseen tasks.',
            'Related tasks may be dependent; task-IID assumption is not established by this analysis.',
            'No population inference over training seeds; three seeds are not hundreds of training replicates.',
            'No exact label-exchangeability or normality claim; no equivalence or mechanism conclusion.',
            'Method was chosen after results were observed; original protocol and intervals remain unchanged.',
            'Bonferroni uses the three existing Level-2 contrasts, not every historical exploratory analysis.',
            'Original percentile intervals and studentized tests are not exact inverses of one another.',
            'Monte Carlo resolution is numerical, not a guarantee about the unknown true tail probability.',
        ],
        'family_size': len(family), 'familywise_alpha': plan['analysis']['familywise_alpha'],
        'comparisons': comparisons, 'policy_rates': seed_rates,
        'task_differences': task_differences, 'inputs': inputs,
        'receipt_sha256': sha256(receipt_path), 'frozen_plan_sha256': frozen['plan_sha256'],
        'analysis_source_sha256': sha256(Path(__file__)),
        'python': sys.version, 'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    result = analyze(args.receipt, args.out, repeats=args.repeats, seed=args.seed)
    print(json.dumps(result['comparisons'], indent=2))


if __name__ == '__main__':
    main()
