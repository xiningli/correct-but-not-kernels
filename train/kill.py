"""K1–K3: declared stopping conditions, including continuous-reward saturation.

Outcome saturation is adapted from glomergence/train/launch.py at
e602f63207fe4e67eeda4dff7b53d49e92cc318b; it must not depend on Arm S reaching 1.
"""
from collections import deque
import math

REQUIRED = {'saturation_pass_rate','saturation_steps','reward_zero_std_fraction',
            'reward_mean_for_zero_std','zero_std_steps','kl_max','median_length_floor',
            'truncation_fraction_max','sandbox_violations_max','detector_fire_fraction_max',
            'max_wall_seconds','importance_weight_mean_min'}

class KillController:
    def __init__(self, config):
        if not REQUIRED <= config.keys() or any(config[k] is None for k in REQUIRED):
            raise ValueError('Every kill criterion must have a concrete value')
        self.cfg = config
        self.saturated = deque(maxlen=config['saturation_steps'])
        self.zero = deque(maxlen=config['zero_std_steps'])

    def observe(self, logs, elapsed):
        c = self.cfg
        if elapsed >= c['max_wall_seconds']:
            return 'wall_budget'
        for key,value in logs.items():
            if isinstance(value,(int,float)) and not math.isfinite(value):
                return 'nonfinite_metric'
        if logs.get('env/harness_errors',0) > 0:
            return 'harness_error'
        if logs.get('sampling/importance_sampling_ratio/mean',1.) < c['importance_weight_mean_min']:
            return 'importance_weight_collapse'
        if logs.get('env/sandbox_violations',0) > c['sandbox_violations_max']:
            return 'sandbox_violation'
        if logs.get('env/detector_fraction',0) > c['detector_fire_fraction_max']:
            return 'detector_finding'
        if logs.get('kl',0) > c['kl_max']:
            return 'kl'
        if 'env/median_length' in logs and logs['env/median_length'] < c['median_length_floor']:
            return 'length_collapse'
        if logs.get('env/truncation_fraction',0) > c['truncation_fraction_max']:
            return 'truncation_spike'
        if 'env/valid' in logs:
            self.saturated.append(logs['env/valid'] >= c['saturation_pass_rate'])
            if len(self.saturated)==self.saturated.maxlen and all(self.saturated):
                return 'outcome_saturation'
        if 'frac_reward_zero_std' in logs and 'reward' in logs:
            self.zero.append(logs['frac_reward_zero_std'] >= c['reward_zero_std_fraction']
                             and logs['reward'] >= c['reward_mean_for_zero_std'])
            if len(self.zero)==self.zero.maxlen and all(self.zero):
                return 'zero_std_saturation'
        return None
