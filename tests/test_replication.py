"""K1–K4: frozen protocols, no split leakage, independent draws and honest denominators."""
import copy
import json
from pathlib import Path
import pytest
import yaml
from eval.replication import validate_rows
from eval.replication_report import paired_interval,common_correct_speed,source_category
from train.sft import encode_example
from train.sft_data import select_successes
from tools.replication import load_plan,stable_seed,sha256,verify_frozen,candidate_seeds,evaluation_request
from tools.replication_worker import record_once

TRAIN='level1/40_LayerNorm'

def success(task=TRAIN):
    return {'task':task,'sample':0,'text':'```python\nclass ModelNew: pass\n```',
        'finish_reason':'stop','token_ids':[4,5],
        'result':{'correct':True,'component':'correct','hacks':[],'speedup':2.},'evaluation_trials':4}

def test_generation_and_input_namespaces_are_independent():
    assert stable_seed(42,TRAIN)==stable_seed(42,TRAIN)
    assert len({stable_seed(42,TRAIN,i) for i in range(8)})==8
    assert stable_seed(42,TRAIN)!=stable_seed(43,TRAIN)
    assert stable_seed(42,TRAIN)!=stable_seed(42,'level1/another')

def test_adjacent_sft_seeds_do_not_share_candidate_pools_or_validation_inputs():
    assert not set(candidate_seeds(20261021,320)) & set(candidate_seeds(20261022,320))
    plan=load_plan();pilot={'evaluation':{}}
    train=evaluation_request(plan,pilot,seeds=candidate_seeds(20261021,320),split='train',tasks=[TRAIN])
    validation=evaluation_request(plan,pilot,seeds=[20264001+i for i in range(32)],split='train',tasks=[TRAIN])
    heldout=evaluation_request(plan,pilot)
    assert len({train['input_seed'],validation['input_seed'],heldout['input_seed']})==3

def test_declared_seed_is_checked_in_results():
    request={'tasks':[TRAIN],'samples_per_task':1,'fresh_checks':4,'sample_seeds':[123],'input_seed':456}
    row=success();row.update(sampling_seed=stable_seed(123,TRAIN),input_seed=stable_seed(456,TRAIN,0))
    validate_rows([row],request,scores=True)
    row['sampling_seed']+=1
    with pytest.raises(ValueError,match='generation seed'):validate_rows([row],request,scores=True)

def test_no_duplicate_or_missing_programs_and_no_harness_failure_scores():
    request={'tasks':[TRAIN],'samples_per_task':2,'fresh_checks':4}
    one=success();two={**success(),'sample':1}
    validate_rows([one,two],request,scores=True)
    for rows in [[one],[one,one],[one,{**two,'task':'unexpected'}]]:
        with pytest.raises(ValueError):validate_rows(rows,request,scores=True)
    two=copy.deepcopy(two);two['result']['component']='harness_error'
    with pytest.raises(RuntimeError):validate_rows([one,two],request,scores=True)

def test_sft_rejects_heldout_and_never_accepts_flagged_or_truncated_success():
    with pytest.raises(ValueError):select_successes([success('level2/task')],['level2/task'])
    with pytest.raises(ValueError):select_successes([success('level2/task')],[TRAIN])
    bad=success();bad['result']['hacks']=['input_alias']
    truncated=success();truncated['finish_reason']='length'
    assert select_successes([bad,truncated,success()],[TRAIN])==[success()]
    with pytest.raises(ValueError):select_successes([bad,truncated],[TRAIN])

def test_completion_only_labels_mask_prompt_and_keep_eos_without_silent_truncation():
    class Tokenizer:
        eos_token_id=106
        def apply_chat_template(self,*args,**kwargs):
            assert kwargs['enable_thinking'] is False
            # Match the pinned Transformers API rather than hiding its default.
            return [1,2,3] if kwargs.get('return_dict',True) is False else {'input_ids':[1,2,3]}
    encoded=encode_example(Tokenizer(),[],[7,8],6)
    assert encoded['input_ids']==[1,2,3,7,8,106]
    assert encoded['labels']==[-100,-100,-100,7,8,106]
    assert encode_example(Tokenizer(),[],[7,8,106],6)==encoded
    with pytest.raises(ValueError):encode_example(Tokenizer(),[],[7,8],5)

def test_plan_rejects_task_leakage_and_seed_reuse(tmp_path):
    good=load_plan()
    for field,value in [('train_seeds',[1,1,2]),('evaluation_seeds',[1]*8),('train_tasks',['level2/task'])]:
        changed={**good,field:value};p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError):load_plan(p)

def test_frozen_request_tampering_fails(tmp_path,monkeypatch):
    import tools.replication as module
    plan=Path(module.DEFAULT_PLAN).read_bytes();(tmp_path/'plan.yaml').write_bytes(plan)
    request=tmp_path/'request.json';request.write_text('{}')
    monkeypatch.setattr(module,'source_hashes',lambda:{'trusted.py':'abc'})
    frozen={'plan_sha256':sha256(tmp_path/'plan.yaml'),'source_sha256':{'trusted.py':'abc'},
            'generated_sha256':{'request.json':sha256(request)}}
    (tmp_path/'frozen.json').write_text(json.dumps(frozen))
    verify_frozen(tmp_path)
    request.write_text('{"adapter":"different"}')
    with pytest.raises(ValueError,match='Frozen job changed'):verify_frozen(tmp_path)

def test_paired_task_intervals_preserve_pairing_and_adjust_confidence():
    rates={'a':.25,'b':.75,'c':1.}
    same=paired_interval(rates,rates,seed=12,repeats=100,alpha=.05/3)
    assert same['value']==0 and same['ci']==[0,0]
    assert same['confidence']==pytest.approx(1-.05/3)
    with pytest.raises(ValueError):paired_interval(rates,{'x':1},seed=1,repeats=10)

def test_common_correct_speed_does_not_drop_failure_denominator():
    a=success();b=success();b['result']['speedup']=1.
    failed=success();failed.update(sample=1);failed['result'].update(correct=False,speedup=None)
    value=common_correct_speed([a,failed],[b,failed])
    assert value['geomean_speedup_ratio']==2
    assert value['common_successful_draws']==1 and value['all_draws']==2
    assert common_correct_speed([failed],[failed])['geomean_speedup_ratio'] is None

def test_existing_records_are_idempotent_only_if_identical(tmp_path):
    path=tmp_path/'record.json';record_once(path,{'step':35});record_once(path,{'step':35})
    with pytest.raises(ValueError):record_once(path,{'step':30})

def test_source_markers_do_not_claim_runtime_backend():
    assert source_category('```python\nimport triton\nclass ModelNew: pass\n```')=='mentions_triton'

def test_resume_receipt_retries_preserve_original(tmp_path):
    from train.launch import resume_receipt
    checkpoint=tmp_path/'checkpoint-10'
    resume_receipt(tmp_path,checkpoint,'abc');resume_receipt(tmp_path,checkpoint,'abc')
    assert len(list((tmp_path/'resume-attempts').glob('*.json')))==1
    with pytest.raises(ValueError):resume_receipt(tmp_path,checkpoint,'changed')
