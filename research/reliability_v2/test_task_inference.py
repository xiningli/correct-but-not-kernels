"""K1/K4: meaningful null-centering, invariance and boundary checks."""
import math

import pytest

from research.reliability_v2.task_inference import aggregate_grids, centered_bootstrap_test, mean_se


def test_hand_calculated_standard_error():
    mean, se = mean_se([1, 2, 3])
    assert mean == 2
    assert se == pytest.approx(math.sqrt(1 / 3))


def test_exact_zero_observed_statistic_has_full_tail():
    result = centered_bootstrap_test([-2, -1, 0, 1, 2], repeats=1000)
    assert result['p_raw'] == 1
    assert result['tail_exceedances'] == 1000


def test_two_sided_sign_and_scale_invariance():
    values = [-2, -1, 0, 1, 1, 2, 3]
    first = centered_bootstrap_test(values, repeats=1000)
    second = centered_bootstrap_test([-4 * x for x in values], repeats=1000)
    assert first['p_raw'] == second['p_raw']


def test_null_centering_detects_large_shift_without_reporting_zero():
    # Uncentered bootstrap would put the resample distribution near +10 and
    # give a tail near 1/2; the null-centered version must put it near zero.
    result = centered_bootstrap_test([9, 10, 11] * 10, repeats=2000)
    assert result['tail_exceedances'] == 0
    assert result['p_raw'] == 1 / 2001


@pytest.mark.parametrize('values', [[0] * 12, [1] * 12])
def test_degenerate_samples_do_not_prove_equality_or_difference(values):
    assert centered_bootstrap_test(values, repeats=10)['p_raw'] is None


def test_seed_policies_are_averaged_without_inventing_extra_tasks():
    grids = [{'a': .25, 'b': .5}, {'a': .75, 'b': 1}]
    assert aggregate_grids(grids) == {'a': .5, 'b': .75}
    with pytest.raises(ValueError):
        aggregate_grids([grids[0], {'a': 1}])
