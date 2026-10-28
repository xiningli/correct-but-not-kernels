"""Behavioral tests for the new development experiment; no CUDA/model needed."""
import ast
import json
from pathlib import Path
import pytest


def test_operation_contracts_keep_profiler_and_memcheck_calibrations_separate():
    from research.repair_v8.pilot.evaluate import operation_allowlist, undeclared_operations
    # Minimal reproducer of the archived r6 official-patch observations. Kineto
    # observed detach; the same patch under memcheck used allocation and copy.
    profile = ['aten.detach.default', 'aten.empty_like.default']
    memory = ['aten.empty_like.default', 'aten.new_empty_strided.default', 'aten.copy_.default']
    runtime = {'allowed_ops': profile, 'allowed_memcheck_ops': memory}
    assert undeclared_operations(memory, operation_allowlist(runtime, True)) == []
    assert undeclared_operations(profile, operation_allowlist(runtime, False)) == []
    assert undeclared_operations(memory, operation_allowlist(runtime, False)) == [
        'aten.copy_.default', 'aten.new_empty_strided.default']
    for sanitizer in [False, True]:
        allowed = operation_allowlist(runtime, sanitizer)
        assert undeclared_operations(allowed + ['aten.relu.default'], allowed) == ['aten.relu.default']
    assert runtime == {'allowed_ops': profile, 'allowed_memcheck_ops': memory}


@pytest.mark.parametrize('runtime,sanitizer', [({}, False), ({}, True),
    ({'allowed_ops': []}, True), ({'allowed_memcheck_ops': []}, False),
    ({'allowed_memcheck_ops': None}, True)])
def test_missing_mode_contract_stops_before_candidate_execution(runtime, sanitizer, tmp_path):
    from research.repair_v8.pilot.evaluate import evaluate, InfrastructureError
    target = tmp_path / 'must-not-start'
    with pytest.raises(InfrastructureError, match='Missing calibrated operation contract'):
        evaluate(tmp_path, 'unused', {}, 'hidden', target, {}, runtime=runtime, sanitizer=sanitizer)
    assert not target.exists()


def test_sanitizer_requires_explicit_instrumentation_and_zero_errors():
    from research.repair_v8.pilot.isolation import sanitizer_verified
    assert sanitizer_verified('========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n')
    for log in ['', 'CUDA_TRITON_OK\n', '========= COMPUTE-SANITIZER\n',
                '========= ERROR SUMMARY: 0 errors\n',
                '========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 1 error\n',
                '========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n========= ERROR SUMMARY: 2 errors\n']:
        assert not sanitizer_verified(log)


@pytest.mark.parametrize('operation,expected', [('thread_name', 0), ('read_thread_name', 0), ('dumpability', -31), ('socket', -31)])
def test_seccomp_allows_only_thread_name_prctl(operation, expected):
    import platform
    import subprocess
    import sys
    from research.repair_v8.pilot.launcher import filter_bytes
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        pytest.skip('Linux x86_64 syscall filter')
    code = '''import ctypes, resource, socket
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
libc=ctypes.CDLL(None, use_errno=True)
libc.syscall.restype=ctypes.c_long
assert libc.prctl(38,1,0,0,0)==0
class Prog(ctypes.Structure):
 _fields_=[('len',ctypes.c_ushort),('filter',ctypes.c_void_p)]
raw=RAW
buf=ctypes.create_string_buffer(raw)
prog=Prog(len(raw)//8,ctypes.addressof(buf))
assert libc.syscall(317,1,1,ctypes.byref(prog))==0
CALL
'''.replace('RAW', repr(filter_bytes())).replace('CALL', {
        'thread_name': "assert libc.prctl(15,ctypes.c_char_p(b'llvm-worker-0'),0,0,0)==0",
        'read_thread_name': 'name=ctypes.create_string_buffer(16); assert libc.prctl(16,name,0,0,0)==0',
        'dumpability': 'libc.prctl(3,0,0,0,0)',
        'socket': 'socket.socket()',
    }[operation])
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
    assert result.returncode == expected, result.stderr

from research.repair_v8.pilot.editor import EditError, apply_response, build_scope, digest, validate
from research.repair_v8.pilot.protocol import projection, selected_source, seed_for, canonical

SOURCE = '''# Copyright example
import math
SCALE = 2

def kernel(x, width):
    return x * width

class Public:
    @staticmethod
    def forward(x):
        return kernel(x, SCALE)
'''


def response(current, operations):
    return canonical({'files': [{'path': 'ops.py', 'expected_sha256': digest(current), 'operations': operations}]})


def test_same_boundary_supports_signature_caller_helper_import_and_constant_changes():
    scope = build_scope(SOURCE, ['Public.forward'])
    ops = [
        {'op': 'replace_function', 'name': 'kernel', 'source': 'def kernel(x, width, offset):\n    return _repair_add(x * width, offset)'},
        {'op': 'replace_function', 'name': 'Public.forward', 'source': '@staticmethod\ndef forward(x):\n    return kernel(x, SCALE, 3)'},
        {'op': 'upsert_helper', 'name': '_repair_add', 'source': 'def _repair_add(x, y):\n    return x + y'},
        {'op': 'replace_constant', 'name': 'SCALE', 'source': 'SCALE = 4'},
        {'op': 'replace_import_block', 'name': 'imports', 'source': 'import math\nimport operator'},
    ]
    edited = apply_response({'ops.py': SOURCE}, {'ops.py': scope}, response(SOURCE, ops), 'atomic_edits')
    full = canonical({'files': [{'path': 'ops.py', 'expected_sha256': digest(SOURCE), 'source': edited['ops.py']}]})
    assert apply_response({'ops.py': SOURCE}, {'ops.py': scope}, full, 'full_module') == edited
    assert 'width, offset' in edited['ops.py']
    assert 'SCALE = 4' in edited['ops.py']


@pytest.mark.parametrize('changed', [
    SOURCE.replace('def forward(x)', 'def forward(x, extra)'),
    SOURCE.replace('class Public:', 'class Public(object):'),
    SOURCE.replace('# Copyright example', '# Removed'),
    SOURCE + '\nopen("outside", "w")\n',
    SOURCE + '\ndef arbitrary_new_name(x):\n    return x\n',
    SOURCE + '\nimport json as kernel\n',
])
def test_both_interfaces_share_protected_boundary(changed):
    with pytest.raises(EditError):
        validate(build_scope(SOURCE, ['Public.forward']), changed)


def test_atomic_failure_never_partially_mutates_current():
    current = {'ops.py': SOURCE}
    ops = [{'op': 'replace_constant', 'name': 'SCALE', 'source': 'SCALE = 10'},
           {'op': 'replace_function', 'name': 'missing', 'source': 'def missing():\n    pass'}]
    with pytest.raises(EditError):
        apply_response(current, {'ops.py': build_scope(SOURCE, ['Public.forward'])}, response(SOURCE, ops), 'atomic_edits')
    assert current == {'ops.py': SOURCE}


def test_stale_hash_duplicate_JSON_and_path_escape_rejected():
    scope = {'ops.py': build_scope(SOURCE, ['Public.forward'])}
    for text in [
        canonical({'files': [{'path': 'ops.py', 'expected_sha256': 'stale', 'source': SOURCE}]}),
        canonical({'files': [{'path': '../ops.py', 'expected_sha256': digest(SOURCE), 'source': SOURCE}]}),
        '{"files": [], "files": []}',
    ]:
        with pytest.raises(EditError):
            apply_response({'ops.py': SOURCE}, scope, text, 'full_module')


def test_feedback_only_projects_visible_measured_fields():
    result = {'component': 'incorrect', 'tests_observed': 2, 'tests_passed': 1,
              'gold_patch': 'SECRET', 'hidden': {'answer': 'SECRET'},
              'cases': [{'passed': False, 'spec': {'id': 'visible-case'}, 'component': 'incorrect',
                         'worker': {'error_type': 'RuntimeError', 'error': '/root/private/patch.py failure', 'secret': 'SECRET'},
                         'input_metadata': {'x': {'shape': [4, 5]}},
                         'checks': [{'name': 'output', 'finite': True, 'max_error': 1.0, 'passed': False, 'expected': 'SECRET'}]}]}
    small = projection(result, False); detail = projection(result, True)
    assert set(small) == {'component', 'tests_observed', 'tests_passed'}
    assert 'SECRET' not in canonical(detail) and '/root/' not in canonical(detail)
    assert detail['diagnostic']['checks'][0]['max_error'] == 1.0


def test_selection_never_uses_hidden_best_of_n():
    attempts = [{'applied': True, 'source': 'first', 'visible_pass': False, 'hidden_pass': True},
                {'applied': True, 'source': 'second', 'visible_pass': True, 'hidden_pass': False},
                {'applied': True, 'source': 'third', 'visible_pass': True, 'hidden_pass': True}]
    assert selected_source('original', attempts) == 'second'
    assert selected_source('original', [{'applied': False, 'source': 'wrong'}]) == 'original'


def test_seed_pairing_is_stable_and_replicas_differ():
    assert seed_for('v8', 'task', 0, 1) == seed_for('v8', 'task', 0, 1)
    assert seed_for('v8', 'task', 0, 1) != seed_for('v8', 'task', 1, 1)


def test_preexisting_compile_decorator_is_not_a_general_waiver():
    from research.repair_v8.pilot.evaluate import source_flags
    original = 'import torch\n@torch.compile(fullgraph=True)\ndef helper(x):\n    return x + 1\n'
    assert source_flags(original, original) == []
    assert 'compiler_wrapper' in source_flags(original + '\n@torch.compile(fullgraph=True)\ndef extra(x):\n    return x\n', original)


def test_cached_execution_retries_identical_candidate_and_preserves_partial_logs(tmp_path, monkeypatch):
    import research.repair_v8.pilot.run as runner
    prepared = tmp_path / 'prepared'; prepared.mkdir(); (prepared / 'manifest.json').write_text('{}')
    target = tmp_path / 'evaluation'; target.mkdir(); (target / 'unfinished.log').write_text('partial')
    calls = []
    def fake(*args, **kwargs):
        calls.append(args[2]); args[4].mkdir()
        value = {'passed': False}; (args[4] / 'result.json').write_text(json.dumps(value)); return value
    monkeypatch.setattr(runner, 'evaluate', fake)
    candidate = {'ops.py': 'saved exact code'}
    runner.cached_evaluate(prepared, 'task', candidate, 'visible', target, {}, {})
    runner.cached_evaluate(prepared, 'task', candidate, 'visible', target, {}, {})
    assert calls == [candidate]
    assert (tmp_path / 'evaluation.interrupted-0/unfinished.log').read_text() == 'partial'
    with pytest.raises(ValueError):
        runner.cached_evaluate(prepared, 'task', {'ops.py': 'changed'}, 'visible', target, {}, {})


@pytest.mark.parametrize('fail_first', [False, True])
def test_scheduler_pairs_first_draws_and_freezes_all_patches_before_judging(tmp_path, monkeypatch, fail_first):
    import research.repair_v8.pilot.run as runner
    import research.repair_v8.pilot.report as reporting
    task = 'unit'; prepared = tmp_path / 'prepared'; prepared.mkdir()
    (prepared / 'manifest.json').write_text('{}')
    (prepared / 'public').mkdir(); (prepared / 'public/unit.json').write_text('{}')
    (prepared / 'modules/unit').mkdir(parents=True)
    (prepared / 'modules/unit/before.py').write_text(SOURCE)
    controls = tmp_path / 'controls.json'
    from research.repair_v8.prepare import sha
    baseline = {'component': 'incorrect', 'tests_observed': 2, 'tests_passed': 1, 'cases': []}
    controls.write_text(json.dumps({'status': 'PASS', 'preparation_sha256': sha(prepared / 'manifest.json'),
        'tasks': {task: {'before_visible': baseline, 'before_hidden_pass_ids': [], 'upstream_passed_nodeids': ['original']}}}))
    cfg = {'tasks': [task], 'replicas': 1, 'attempts': 2, 'max_detail_tokens': 2048, 'max_prompt_tokens': 16384,
           'seed_namespace': 'test', 'upstream_seconds': 1,
           'arms': {'A': {'interface': 'full_module', 'diagnostics': False}, 'B': {'interface': 'full_module', 'diagnostics': True},
                    'C': {'interface': 'atomic_edits', 'diagnostics': False}, 'D': {'interface': 'atomic_edits', 'diagnostics': True}}}
    monkeypatch.setattr(runner, 'load', lambda p: ({}, cfg))
    monkeypatch.setitem(runner.TASKS, task, {'path': 'ops.py', 'protected': ['Public.forward']})
    class Tok:
        def encode(self, text, **kwargs): return list(text.encode())
        def apply_chat_template(self, messages, **kwargs): return list(canonical(messages).encode())
    monkeypatch.setattr(runner, 'tokenizer', lambda p: Tok())
    monkeypatch.setattr(runner, 'messages', lambda task, current, interface, feedback: [{'role': 'user', 'content': canonical({'interface': interface, 'feedback': feedback})}])
    batches = []
    def fake_generate(p, entries, directory, *args):
        batches.append(entries)
        result = {}
        for e in entries:
            text = canonical({'files': [{'path': 'ops.py', 'expected_sha256': digest(SOURCE), 'source': SOURCE}]}) if e['interface'] == 'full_module' else response(SOURCE, [{'op': 'replace_constant', 'name': 'SCALE', 'source': 'SCALE = 2'}])
            result[e['id']] = {**e, 'text': text, 'prompt_tokens': 10, 'output_tokens': 10, 'generation_seconds': 1.,
                              'finish_reason': 'length' if fail_first and e['step'] == 0 else 'stop'}
        return result
    monkeypatch.setattr(runner, 'generate', fake_generate)
    judged = []
    out = tmp_path / 'run'
    def fake_eval(p, t, source, panel, path, config, runtime, sanitizer=False):
        if panel == 'hidden':
            assert (out / 'selected-patches.json').exists()
            frozen = json.loads((out / 'selected-patches.json').read_text())
            assert len(frozen) == 4
            judged.append(path)
        (path / 'checkout').mkdir(parents=True, exist_ok=True)
        return {'passed': True, 'component': 'pass', 'tests_observed': 2, 'tests_passed': 2, 'cases': []}
    monkeypatch.setattr(runner, 'cached_evaluate', fake_eval)
    monkeypatch.setattr(runner, 'upstream_run', lambda *args: {'passed': True})
    monkeypatch.setattr(reporting, 'report', lambda *args: None)
    runner.execute(prepared, controls, out, Path('/inference/python'), Path('/model'))
    assert [len(b) for b in batches] == ([2, 4] if fail_first else [2])  # only the first draws share
    assert len(judged) == 8  # four hidden panels + four memcheck panels
    selected = json.loads((out / 'selected-patches.json').read_text())
    assert all(x['attempts'] == (2 if fail_first else 1) and x['logical_output_tokens'] == (20 if fail_first else 10) for x in selected.values())
    # A complete resume rebuilds identical selection and does no new judging.
    judged.clear()
    runner.execute(prepared, controls, out, Path('/inference/python'), Path('/model'))
    assert not judged


def test_output_reader_rejects_huge_declared_shape_before_loading(tmp_path):
    pytest.importorskip('torch')
    import torch
    from research.repair_v8.pilot.evaluate import compare_file
    header = json.dumps({'output': {'dtype': 'F32', 'shape': [10**12], 'data_offsets': [0, 4]}}).encode()
    path = tmp_path / 'output.safetensors'; path.write_bytes(len(header).to_bytes(8, 'little') + header + b'0000')
    checks, flags = compare_file(path, {'output': torch.zeros(1, dtype=torch.float64)}, {'kind': 'attn'})
    assert flags == ['output_contract'] and not checks[0]['passed']
