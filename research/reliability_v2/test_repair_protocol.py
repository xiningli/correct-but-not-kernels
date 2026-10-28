"""K3/K4: meaningful leakage, budget, and strict-observation regressions."""
import pytest

from research.reliability_v2.repair_protocol import Budget, run_episode, select_patch, visible_result
from research.reliability_v2.strict_contract import assess_observation, declared_kernel_names


def test_feedback_changes_history_but_independent_condition_never_sees_it():
    histories = {}
    for mode in ['test_feedback', 'independent_attempts']:
        seen = []
        def generate(messages, limit, seed):
            seen.append(messages)
            return dict(text='patch', new_tokens=limit, prompt_tokens=10, finish_reason='length')
        def test(text, index):
            return dict(passed=index, total=3, diagnostic='visible_only_marker', infrastructure_error=False)
        result = run_episode(task='dev/bug', issue='issue', initial_code='broken', system_prompt='contract',
                             mode=mode, generate=generate, test_visible=test, namespace=100,
                             budget=Budget(tokens_per_attempt=4, total_new_tokens=10))
        assert result['total_new_tokens'] == 10
        assert [x['new_tokens'] for x in result['attempts']] == [4, 4, 2]
        assert result['selected'] == 2
        assert result['visible_test_calls'] == 3
        histories[mode] = seen
    assert [len(x) for x in histories['test_feedback']] == [2, 4, 6]
    assert [len(x) for x in histories['independent_attempts']] == [2, 2, 2]
    assert 'visible_only_marker' not in str(histories['independent_attempts'])


def test_hidden_fields_and_infrastructure_failures_cannot_be_scored():
    public = dict(passed=1, total=2, diagnostic='', infrastructure_error=False)
    with pytest.raises(ValueError):
        visible_result({**public, 'hidden_passed': True})
    with pytest.raises(RuntimeError):
        visible_result({**public, 'infrastructure_error': True})
    attempts = [{'visible': public}, {'visible': public}]
    assert select_patch(attempts) == 0  # fixed tie-break, no oracle selection
    with pytest.raises(ValueError):
        select_patch([attempts[0], {'visible': {**public, 'total': 4}}])


@pytest.mark.parametrize('ops,kernels,component', [
    (['aten.empty_like.default'], ['add_one'], 'compliant'),
    (['aten.add.Tensor'], ['add_one', 'at::vectorized_elementwise_kernel'], 'contract_violation'),
    ([], [], 'contract_violation'),
    ([], ['cublas_gemm'], 'contract_violation'),
    ([], ['not_add_one_spoof'], 'contract_violation'),
])
def test_actual_execution_and_no_fallback_required(ops, kernels, component):
    obs = dict(complete=True, cuda_available=True, dispatch_ops=ops, cuda_kernels=kernels)
    assert assess_observation(obs, ['add_one'])['component'] == component


def test_missing_observation_is_harness_failure():
    assert assess_observation({}, ['add_one'])['component'] == 'harness_error'
    obs = dict(complete=True, cuda_available=False, dispatch_ops=[], cuda_kernels=[])
    assert assess_observation(obs, ['add_one'])['component'] == 'harness_error'


def test_native_and_triton_names_are_only_declarations():
    source = '@triton.jit\ndef foo(X): pass\ncode = "__global__ void bar(float* x) {}"\n'
    assert declared_kernel_names(source) == ['bar', 'foo']
    obs = dict(complete=True, cuda_available=True, dispatch_ops=[], cuda_kernels=[])
    assert assess_observation(obs, declared_kernel_names(source))['component'] == 'contract_violation'


def test_profiler_preload_keeps_original_seccomp_filter_byte_identical():
    from env.sandbox_launcher import filter_bytes as original
    from research.reliability_v2.strict_launcher import filter_bytes as strict
    assert strict() == original()


def test_development_prompts_cannot_include_test_graphs_or_hidden_answers():
    from research.reliability_v2.new_tasks import catalog
    from research.reliability_v2.prepare_transfer import messages
    tasks = catalog()
    dev = [t for t in tasks if t['split'] == 'development']
    test = [t for t in tasks if t['split'] == 'test_candidate']
    assert len(dev) == 6 and len(test) == 48
    assert not {x['graph_sha256'] for x in dev} & {x['graph_sha256'] for x in test}
    # Intervention leaves task/reference text exactly equal.
    for task in tasks:
        a, b = messages(task, 'minimal_contract'), messages(task, 'api_scaffold')
        assert a[1] == b[1]
        assert 'case-0' not in str(a) and 'expected_output' not in str(a)
