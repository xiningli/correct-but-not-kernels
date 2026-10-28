"""K1–K4: failed programs stay in denominators and paired CIs preserve pairing."""
import pytest
from eval.report import measurements, bootstrap, paired_change

def rows(correct):
    return [{'task':f'level1/task-{i}', 'result':{'component':'correct' if passed else 'incorrect',
            'correct':passed,'speedup':2. if passed else None,'hacks':[]}}
            for i,passed in enumerate(correct)]

def test_speed_is_explicitly_conditional_but_fast1_is_not():
    got=measurements(rows([True,False]))
    assert got['correctness']==.5 and got['fast1']==.5
    assert got['correct_geomean_speedup']==2.
    assert measurements(rows([False,False]))['correct_geomean_speedup'] is None

def test_paired_no_change_has_zero_interval():
    data=rows([True,False,True,False])
    assert paired_change(data,data,repeats=100)=={'value':0.,'ci95':[0.,0.]}

def test_mismatched_tasks_and_infrastructure_are_not_valid_comparisons():
    with pytest.raises(ValueError):
        paired_change(rows([True]),rows([True,True]))
    data=rows([False]);data[0]['result']['component']='harness_error'
    with pytest.raises(ValueError):
        measurements(data)

def test_bootstrap_is_reproducible_and_does_not_drop_undefined_speed():
    one=bootstrap(rows([True,False]),repeats=100)
    assert one==bootstrap(rows([True,False]),repeats=100)
    assert one['correct_geomean_speedup']['ci95'] is None
