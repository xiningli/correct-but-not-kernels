"""K1: analytic GRPO signal must not mistake all-pass groups for useful signal."""
import pytest
from eval.signal import summarize

def rows(correct):
    return [{'task':'train','tokens':100,'finish_reason':'stop',
             'result':{'correct':i<correct,'speedup':1.0 if i<correct else None,
                       'component':'correct' if i<correct else 'incorrect'}} for i in range(32)]

def test_all_equal_has_zero_signal():
    for n in [0,32]:
        assert summarize(rows(n))['mean_p_signal'] == 0

def test_one_success_is_seen_in_one_quarter_of_groups():
    s = summarize(rows(1))['tasks']['train']
    assert s['pass_at_1'] == 1/32
    assert s['pass_at_8'] == pytest.approx(0.25)
    assert s['p_signal'] == pytest.approx(0.25)
