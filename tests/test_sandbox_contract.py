"""K3: fail closed locally; hostile GPU execution belongs only to formal C1."""
import os
import pytest
from env import sandbox
from env.reward import failure_reward, FAILURE_COMPONENTS
from tools.gatelib import require_gate

def test_no_weak_sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    with pytest.raises(RuntimeError, match='no weak fallback'):
        sandbox.run("raise AssertionError('must not execute')", tmp_path/'run', monitor_gpu=False)
    assert not (tmp_path/'run').exists()

def test_gate_pass_is_required(tmp_path):
    with pytest.raises(RuntimeError, match='PASS required'):
        require_gate('C1', tmp_path)

def test_failure_rewards_cannot_be_positive():
    for component in FAILURE_COMPONENTS:
        assert failure_reward(component, 'C') == -0.1
        assert failure_reward(component, 'S') == 0
    with pytest.raises(ValueError):
        failure_reward('ok')
