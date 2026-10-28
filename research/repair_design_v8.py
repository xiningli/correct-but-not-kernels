"""Planning arithmetic only; no models, observed outcomes, GPU calls or p-values.

For one paired replica let Z = success_D - success_A in {-1, 0, 1}.
Given mean delta and discordance q, Var(Z) = q - delta**2. For R
exchangeable replicas with within-defect correlation rho, the variance of
their mean is Var(Z) * (1 + (R-1)*rho) / R. The power below uses a normal
approximation and Bonferroni alpha/3, not a simulation or a guarantee about
the future stratified bootstrap test. Repository heterogeneity is omitted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist


def scenario(n, delta, discordance, rho, replicas=3, family=3):
    if not (n > 1 and abs(delta) <= discordance <= 1 and 0 <= rho <= 1):
        raise ValueError('Invalid paired-outcome planning assumptions')
    variance = (discordance - delta ** 2) * (1 + (replicas - 1) * rho) / replicas
    if variance <= 0:
        raise ValueError('Positive planning variance required')
    normal = NormalDist()
    threshold = normal.inv_cdf(1 - .05 / (2 * family))
    shift = delta * math.sqrt(n / variance)
    power = normal.cdf(-threshold - shift) + 1 - normal.cdf(threshold - shift)
    n80 = math.ceil((threshold + normal.inv_cdf(.8)) ** 2 * variance / delta ** 2)
    return {'defects': n, 'replicas': replicas, 'difference': delta,
            'discordance': discordance, 'within_defect_replica_correlation': rho,
            'task_mean_variance': variance, 'family_size': family,
            'approximate_power': power, 'approximate_n_for_80_percent_power': n80}


def design_counts(n, replicas=3):
    # First draws shared within the full pair and within the scoped triple.
    return {'defects': n, 'main_arms': 5, 'replicas': replicas,
            'logical_episodes': n * replicas * 5,
            'maximum_logical_candidates': n * replicas * 5 * 4,
            'maximum_unique_candidates_with_shared_first_draws': n * replicas * (2 + 5 * 3),
            'optional_12b_episodes_on_24_frozen_tasks': 24 * 2 * 2,
            'optional_mechanism_candidates_max': 24 * 3 * 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [scenario(n, delta, q, rho)
            for n in [40, 60, 80, 120]
            for delta in [.10, .15, .20]
            for q in [.30, .50] for rho in [0., .5, 1.]]
    result = {'status': 'hypothetical_planning_not_experimental_evidence',
        'method': __doc__, 'observed_data_used': False,
        'assumptions': ['Independent root defects; exchangeable replicas within a defect.',
            'Single-replica discordance and replica correlation are hypothetical, not estimated from the two old bugs.',
            'Normal planning approximation; no claim that observed task differences are normally distributed.',
            'Does not model repository effects, task curation bias, bootstrap error or endpoint multiplicity beyond three contrasts.'],
        'counts': [design_counts(n) for n in [40, 60, 80]], 'sensitivity': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for row in rows:
        if row['defects'] in [60, 80] and row['difference'] == .15 and row['within_defect_replica_correlation'] == .5:
            print(json.dumps(row))


if __name__ == '__main__':
    main()
