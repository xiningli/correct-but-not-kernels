"""CPU regression checks for experiment identity and missing-data semantics."""
import ast
import json

import pytest

from research.reliability_v2.capability_run import report
from research.reliability_v2.decomposition import empty_result, summarize_observations
from research.reliability_v2.fixed_interface import ABI, assemble, control_kernel, extract_code, messages, wrapper
from research.reliability_v2.new_tasks import catalog


def tasks():
    return [t for t in catalog() if t['split'] == 'development']


def observation(ops=(), kernels=('run_kernel',)):
    return {'complete': True, 'cuda_available': True, 'dispatch_ops': list(ops), 'cuda_kernels': list(kernels)}


def test_unknown_is_not_a_negative_observation():
    assert summarize_observations({'observations': []}, ['run_kernel'], 4)[1] is None
    assert summarize_observations({'observations': [observation()]}, ['run_kernel'], 4)[1] is None
    assert summarize_observations({'observations': [observation()] * 4}, ['run_kernel'], 4)[1] is True
    assert summarize_observations({'observations': [observation(['aten.add.Tensor'])]}, ['run_kernel'], 4)[1] is False


def test_bad_observer_is_a_harness_error():
    with pytest.raises(ValueError):
        summarize_observations({'observations': [{}]}, ['run_kernel'], 4)


def test_fixed_host_supplies_no_tensor_math():
    for task in tasks():
        tree = ast.parse(wrapper(task))
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        assert set(calls) <= {'torch.empty', 'x.stride', 'y.stride', 'triton.next_power_of_2', 'max', 'run_kernel[m,]'}
        assert not any(isinstance(n, (ast.BinOp, ast.AugAssign)) for n in ast.walk(tree))
        assert 'data_ptr' not in wrapper(task)


def test_no_silent_candidate_repair():
    task = tasks()[0]
    kernel = control_kernel(task)
    with pytest.raises(ValueError):
        assemble(kernel + wrapper(task), task, 'fixed_host')
    for condition in ['full_program', 'fixed_host']:
        with pytest.raises(ValueError):
            assemble(kernel.replace('def run_kernel(X,', 'def run_kernel(Z,'), task, condition)
    assert assemble(kernel + wrapper(task), task, 'full_program') == kernel + wrapper(task)


def test_kernel_only_extraction_and_complete_program_share_format_rules():
    from env.reward import extract_program
    for task in tasks():
        kernel = control_kernel(task)
        fenced = '```python\n' + kernel + '```'
        assert extract_program(fenced) is None  # the old full-program requirement
        assert extract_code(fenced) == kernel
        ast.parse(assemble(extract_code(fenced), task, 'fixed_host'))
        full = kernel + wrapper(task)
        assert assemble(extract_code('```python\n' + full + '```'), task, 'full_program') == full
        with pytest.raises(ValueError):
            assemble(kernel, task, 'full_program')
        assert extract_code('<|channel>thought' + fenced) is None
        assert extract_code(kernel) is None


def test_prompts_have_shared_abi_and_development_only():
    for task in tasks():
        for condition in ['full_program', 'fixed_host']:
            prompt = messages(task, condition)
            assert ABI in prompt[0]['content']
            assert task['reference_source'] in prompt[1]['content']
            ast.parse(assemble(control_kernel(task), task, 'fixed_host'))
    test_task = next(t for t in catalog() if t['split'] == 'test_candidate')
    with pytest.raises(ValueError):
        messages(test_task, 'fixed_host')


def test_report_all_attempt_denominator_and_unknown(tmp_path):
    rows = [{'policy': 'base', 'condition': 'fixed_host', 'task': f'dev/{i}', 'result': empty_result('crash')} for i in range(2)]
    rows[1]['result'].update(numerically_correct=True, contract_compliant=False)
    (tmp_path / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    cell, = report(tmp_path)['cells']
    assert cell['attempts'] == 2
    assert cell['numerically_correct'] == {'true': 1, 'false': 0, 'unknown': 1}
    assert cell['contract_compliant'] == {'true': 0, 'false': 1, 'unknown': 1}
    assert cell['correct']['false'] == 2


def test_report_rejects_duplicate_attempt(tmp_path):
    row = {'policy': 'base', 'task': 'x', 'sample': 0, 'result': empty_result('format')}
    (tmp_path / 'metrics.jsonl').write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(ValueError):
        report(tmp_path)
