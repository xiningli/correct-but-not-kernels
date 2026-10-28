"""K1–K3: stopping tests cover findings, loss corruption and Arm S saturation."""
import pytest
from train.kill import KillController

def config():
    return dict(saturation_pass_rate=.95,saturation_steps=5,reward_zero_std_fraction=.95,
                reward_mean_for_zero_std=.95,zero_std_steps=5,kl_max=.2,median_length_floor=32,
                truncation_fraction_max=.5,sandbox_violations_max=0,detector_fire_fraction_max=.25,
                max_wall_seconds=28800,importance_weight_mean_min=.1)

def test_continuous_reward_still_saturates_on_correctness():
    control=KillController(config())
    for i in range(4):
        assert control.observe({'env/valid':1.,'reward':.55,'frac_reward_zero_std':0.},i) is None
    assert control.observe({'env/valid':1.,'reward':.55,'frac_reward_zero_std':0.},5)=='outcome_saturation'

@pytest.mark.parametrize('logs,reason',[
    ({'env/sandbox_violations':1},'sandbox_violation'),
    ({'env/detector_fraction':.5},'detector_finding'),
    ({'env/harness_errors':1},'harness_error'),
    ({'kl':.3},'kl'),({'loss':float('nan')},'nonfinite_metric'),
    ({'env/median_length':5},'length_collapse'),
    ({'sampling/importance_sampling_ratio/mean':1e-9},'importance_weight_collapse'),
    ({'env/truncation_fraction':.75},'truncation_spike')])
def test_immediate_stops(logs,reason):
    assert KillController(config()).observe(logs,0)==reason

def test_missing_threshold_is_not_a_disabled_guard():
    cfg=config(); cfg['kl_max']=None
    with pytest.raises(ValueError):
        KillController(cfg)
