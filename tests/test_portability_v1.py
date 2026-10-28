"""Behavioral safeguards for paired, frozen second-device evaluation."""
import copy
from pathlib import Path
import pytest
from research.portability_v1.common import PLAN, read
from research.portability_v1.backend import docker_command
from research.portability_v1.report import aggregate, effects


def grid(tasks=('a', 'b')):
    plan = read(PLAN)
    return [{'task': t, 'policy': p, 'sample': s,
             'remote': {'numerically_correct': p != 'base'},
             'result': {'numerically_correct': p != 'base', 'correct': False, 'component': 'contract_violation'}}
            for t in tasks for p in plan['policies'] for s in range(8)]


def test_all_attempt_denominator_and_unknown_counts_are_retained():
    plan = read(PLAN); rows = grid(('a',))
    rows[0]['result']['numerically_correct'] = None
    complete, counts = aggregate(rows, plan['policies'])
    assert len(complete) == 1
    assert counts['base']['attempts'] == 8
    assert counts['base']['local_numeric_unknown'] == 1
    assert counts['C-20261021']['both_numeric_pass'] == 8
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate(rows + rows[:1], plan['policies'])


def test_partial_task_cannot_change_aggregate_or_intervals():
    plan = read(PLAN)
    rows = grid()
    full, counts = aggregate(rows[:56], plan['policies'])
    partial, other = aggregate(rows[:-1], plan['policies'])
    assert full.keys() == partial.keys() == {'a'}
    assert counts == other
    assert effects(full, plan) == effects(partial, plan)


def test_shared_base_and_seed_average_with_paired_effect_change():
    plan = read(PLAN); rows = grid()
    for r in rows:
        if r['policy'] == 'C-20261021':
            r['result']['numerically_correct'] = False
    complete, _ = aggregate(rows, plan['policies'])
    result = effects(complete, plan)
    assert result['C-base/local']['estimate'] == pytest.approx(2 / 3)
    assert result['C-base/remote_same_tasks']['estimate'] == 1.
    assert result['C-base/local_minus_remote_effect']['estimate'] == pytest.approx(-1 / 3)
    assert result['C-base/local']['tasks'] == 2
    assert result['C-base/local']['confidence'] == pytest.approx(1 - .05 / 3)


def test_candidate_container_cannot_mount_reference_or_host_root(tmp_path):
    plan = read(PLAN)
    cmd = docker_command('test', tmp_path / 'runner.py', tmp_path / 'inputs', tmp_path / 'work', plan,
                         extra=[(str(tmp_path / 'candidate.py'), '/program/candidate.py', True)])
    assert '--privileged' not in cmd and '--pid=host' not in cmd
    assert cmd[cmd.index('--network') + 1] == 'none'
    assert cmd[cmd.index('--cap-drop') + 1] == 'ALL'
    assert '--read-only' in cmd and 'no-new-privileges' in cmd
    mounts = [cmd[i + 1] for i, v in enumerate(cmd) if v == '--mount']
    assert len([m for m in mounts if not m.endswith(',readonly')]) == 1
    assert not any('/oracle' in m or 'src=/,' in m or 'docker.sock' in m for m in mounts)


def test_comparison_does_not_relabel_unknown_as_observed_numerical_failure():
    plan = read(PLAN); rows = grid(('a',))
    for r in rows:
        r['result']['numerically_correct'] = None
        r['result']['component'] = 'gpu_oom'
    _, counts = aggregate(rows, plan['policies'])
    assert counts['C-20261021']['local_numeric_unknown'] == 8
    assert counts['C-20261021']['remote_pass_local_not_pass'] == 8
    assert counts['C-20261021']['attempts'] == 8
