import pytest

from research.reliability_v2.repair_cases import SOURCES

NEEDS_UPSTREAM = pytest.mark.skipif(not SOURCES.is_dir(), reason='Upstream Liger-Kernel snapshots are fetched, not committed: python -m research.reliability_v2.repair_sources (see its --help)')

from research.scoped_repair.edits import (
    EditError, accept_full_program, apply_edits, build_scope, function_text,
    sha256, validate_candidate,
)

SOURCE = '''# Frozen license
import torch
import triton
import triton.language as tl
BLOCK = 32

@triton.jit
def kernel(X, Y, stride):
    tl.store(Y, tl.load(X + stride))

class ModelNew(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        kernel[(1,)](x, y, 1)
        return y
'''


def apply(source, replacements, scope=None, expected=None):
    return apply_edits(source, scope or build_scope(SOURCE), replacements,
                       expected_source_sha256=expected or sha256(source))


def test_atomic_kernel_signature_and_host_call_edit_matches_full_program_arm():
    scope = build_scope(SOURCE)
    kernel = function_text(SOURCE, 'kernel').replace('stride):', 'stride, offset):').replace('X + stride', 'X + stride + offset')
    host = function_text(SOURCE, 'ModelNew.forward').replace('(x, y, 1)', '(x, y, x.stride(0), 0)')
    result = apply(SOURCE, {'kernel': kernel, 'ModelNew.forward': host}, scope)
    assert result.startswith('# Frozen license\nimport torch\n') and 'BLOCK = 32' in result
    assert 'x.stride(0)' in result and 'stride, offset' in result
    assert accept_full_program(SOURCE, scope, result, expected_source_sha256=sha256(SOURCE)) == result


@pytest.mark.parametrize('mutated', [
    SOURCE.replace('import torch', 'import os\nimport torch'),
    SOURCE.replace('BLOCK = 32', 'BLOCK = 64'),
    SOURCE.replace('# Frozen license', '# Changed license'),
    SOURCE.replace('class ModelNew(torch.nn.Module)', 'class ModelNew(object)'),
    SOURCE + '\nprint("outside")\n',
])
def test_full_program_rejects_changes_outside_same_boundary(mutated):
    with pytest.raises(EditError, match='outside_editable'):
        accept_full_program(SOURCE, build_scope(SOURCE), mutated, expected_source_sha256=sha256(SOURCE))


def test_external_signature_protected_internal_signature_editable():
    replacement = function_text(SOURCE, 'ModelNew.forward').replace('self, x', 'self, x, extra')
    with pytest.raises(EditError, match='external_signature'):
        apply(SOURCE, {'ModelNew.forward': replacement})


def test_stale_or_unknown_target_rejected_and_current_source_unchanged():
    before = sha256(SOURCE)
    with pytest.raises(EditError, match='stale_source'):
        apply(SOURCE, {'kernel': function_text(SOURCE, 'kernel')}, expected='0'*64)
    with pytest.raises(EditError, match='unknown_function'):
        apply(SOURCE, {'test_score': 'def test_score():\n    return True\n'})
    assert sha256(SOURCE) == before


def test_invalid_multifunction_patch_rejects_whole_transaction():
    host = function_text(SOURCE, 'ModelNew.forward').replace('return y', 'return x')
    with pytest.raises(EditError):
        apply(SOURCE, {'ModelNew.forward': host, 'kernel': 'def kernel(:\n    pass\n'})
    assert 'return y' in SOURCE and 'return x' not in SOURCE


def test_candidate_code_is_never_executed(tmp_path):
    target = tmp_path/'must-not-exist'
    definition = f'@open({str(target)!r}, "w")\ndef kernel(X, Y, stride):\n    pass\n'
    candidate = apply(SOURCE, {'kernel': definition})
    assert not target.exists() and '@open(' in candidate
    # Passing the editing contract is intentionally not a safety certification.


def test_extra_top_level_code_and_duplicate_or_overlapping_names_rejected():
    with pytest.raises(EditError, match='one_function'):
        apply(SOURCE, {'kernel': 'def kernel(X, Y, stride):\n    pass\nprint(1)\n'})
    with pytest.raises(EditError, match='ambiguous_function'):
        build_scope(SOURCE+'\ndef kernel():\n    pass\n')


def test_multiple_rounds_recompute_locations_but_keep_original_boundary():
    scope = build_scope(SOURCE)
    first = apply(SOURCE, {'kernel': function_text(SOURCE, 'kernel').replace('    tl.store', '    offset = 0\n    tl.store')}, scope)
    second = apply(first, {'ModelNew.forward': function_text(first, 'ModelNew.forward').replace('(x, y, 1)', '(x, y, x.stride(0))')}, scope)
    validate_candidate(scope, second)
    assert 'offset = 0' in second and 'x.stride(0)' in second


@NEEDS_UPSTREAM
@pytest.mark.parametrize('task_id', ['development/case-001', 'development/case-002'])
def test_existing_public_tasks_roundtrip_without_gold_fix_or_GPU(task_id):
    from research.diagnostics_v1.tasks import public_task
    source = public_task(task_id)['source']
    scope = build_scope(source)
    edits = {name: function_text(source, name) for name in scope.describe()['editable']}
    assert apply(source, edits, scope) == source
    assert scope.describe()['origin'].endswith('no_gold_localization')
