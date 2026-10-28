"""K1–K3: scientific reward semantics and persistent stop evidence."""
import json
import sys
from types import SimpleNamespace
import pytest
from env.reward import score_result
from env.run_kernel import RunResult
from train.launch import callbacks,is_text_parameter

def test_text_sync_skips_only_declared_frozen_multimodal_weights():
    assert is_text_parameter('model.language_model.layers.0.self_attn.q_proj.weight')
    assert is_text_parameter('lm_head.weight')
    assert not is_text_parameter('model.embed_vision.pos_embedding')
    with pytest.raises(ValueError):
        is_text_parameter('unexpected_new_module.weight')

def test_correctness_and_speed_rewards_keep_failure_semantics():
    assert score_result(RunResult('correct',True,.5),'C')==1.
    assert score_result(RunResult('incorrect'),'C')==0.
    assert score_result(RunResult('compile'),'C')==-.1
    assert score_result(RunResult('compile'),'S')==0.
    assert score_result(RunResult('harness_error'),'C') is None
    rewards=[score_result(RunResult('correct',True,s),'S') for s in [.1,1.,10.]]
    assert 0<rewards[0]<rewards[1]<rewards[2]<1
    assert rewards[1]==.5

def test_nonfinite_stop_evidence_survives_strict_json(tmp_path,monkeypatch):
    # This callback test requires no GPU/model and exercises the failure path
    # that previously would fail while trying to serialize its own NaN cause.
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(TrainerCallback=object))
    (tmp_path/'metrics.jsonl').touch()
    config=dict(saturation_pass_rate=.95,saturation_steps=5,reward_zero_std_fraction=.95,
        reward_mean_for_zero_std=.95,zero_std_steps=5,kl_max=.2,median_length_floor=32,
        truncation_fraction_max=.5,sandbox_violations_max=0,detector_fire_fraction_max=.25,
        max_wall_seconds=28800,importance_weight_mean_min=.1)
    cb=callbacks({'kill':config},tmp_path,SimpleNamespace(latest={}))[0]
    control=SimpleNamespace(should_training_stop=False,should_save=False)
    cb.on_log(None,SimpleNamespace(global_step=1),control,logs={'loss':float('nan')})
    assert control.should_training_stop and control.should_save
    evidence=json.loads((tmp_path/'kill.json').read_text())
    assert evidence['reason']=='nonfinite_metric'
    assert evidence['metrics']['loss']=='nan'
    assert json.loads((tmp_path/'metrics.jsonl').read_text())['step']==1
