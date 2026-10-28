"""CPU controls against false qualification and broken numerical oracles."""
from copy import deepcopy
import io
import tarfile

import pytest
import torch

from research.repair_v8.cases import TASKS, attnres_reference, check_close, groupnorm_reference, run_case
from research.repair_v8.prepare import safe_extract
from research.repair_v8.qualify import SEEDS, bug_reproduced, summarize_task, suite_pass


def test_reference_groupnorm_gradients_match_independent_autograd():
    torch.manual_seed(7)
    inputs = [torch.randn(2, 6, 3, 5, dtype=torch.float64),
              torch.randn(6, dtype=torch.float64), torch.randn(6, dtype=torch.float64)]
    a = [x.clone().requires_grad_() for x in inputs]
    b = [x.clone().requires_grad_() for x in inputs]
    y = groupnorm_reference(*a, 3)
    z = torch.nn.functional.group_norm(b[0], 3, b[1], b[2], 1e-6)
    torch.testing.assert_close(y, z, atol=1e-10, rtol=1e-10)
    grad = torch.randn_like(y)
    y.backward(grad); z.backward(grad)
    for x, reference in zip(a, b):
        torch.testing.assert_close(x.grad, reference.grad, atol=1e-9, rtol=1e-9)


def test_attnres_oracle_rejects_ignored_tail_blocks():
    v = torch.zeros(40, 1, 1, 8, dtype=torch.float64)
    v[32:] = 4
    q, norm = torch.zeros(8, dtype=torch.float64), torch.ones(8, dtype=torch.float64)
    wanted = attnres_reference(v, q, norm)
    wrong = attnres_reference(v[:32], q, norm).float()
    with pytest.raises(AssertionError, match='numerical mismatch'):
        check_close('output', wrong, wanted, torch.float32, [])


def test_tail_probe_rejects_guard_writes():
    class Mutant:
        def rwkv_relu_and_square_fwd(self, x, inplace):
            full = x.as_strided((x.untyped_storage().nbytes() // x.element_size(),), (1,))
            full.copy_(full.clamp_min(0).square())
            return x
    with pytest.raises(AssertionError, match='out-of-bounds store changed guard'):
        run_case('fla_rwkv7', 'rwkv_tail_guard', 9, Mutant(), 'cpu')


def complete_synthetic_receipts():
    key = 'liger_attnres'; task = TASKS[key]; rows = []
    for seed in SEEDS:
        for rev in ['before', 'fixed']:
            for kind, case in [('upstream', 'suite')] + [('probe', c) for c in task['cases']] + (
                    [('profile', task['profile_case']), ('memcheck', task['memcheck_case'])] if rev == 'fixed' else []):
                receipt = {'status': 'pass', 'oracle_only': False, 'stage': 'complete',
                           'result': {'mode': 'GPU_original_repository'}}
                if kind == 'upstream':
                    receipt = {'exitstatus': 0, 'rows': [{'outcome': 'passed'}]}
                if kind == 'profile':
                    receipt['cuda_kernel_names'] = [task['kernel_name']]
                if rev == 'before' and case == task['bug_case']:
                    receipt = {'status': 'test_failure', 'stage': 'probe', 'oracle_only': False,
                               'error': 'output: numerical mismatch'}
                rows.append({'seed': seed, 'revision': rev, 'kind': kind, 'case': case, 'receipt': receipt,
                             'process': {'returncode': 2 if receipt.get('status') == 'test_failure' else 0, 'timeout': False}})
    return rows


def test_qualification_requires_every_repeat_and_real_gpu_evidence():
    rows = complete_synthetic_receipts()
    assert summarize_task('liger_attnres', rows, True)['status'] == 'development_controls_pass'
    # A missing repetition cannot quietly reduce the denominator.
    incomplete = [r for r in rows if r['seed'] != SEEDS[-1]]
    assert summarize_task('liger_attnres', incomplete, True)['status'] == 'not_qualified'
    cpu = deepcopy(rows)
    for row in cpu:
        if row['revision'] == 'fixed' and row['kind'] == 'probe':
            row['receipt']['oracle_only'] = True
    assert summarize_task('liger_attnres', cpu, True)['status'] == 'not_qualified'
    assert summarize_task('liger_attnres', rows, False)['status'] == 'not_qualified'
    regress = deepcopy(rows)
    next(r for r in regress if r['kind'] == 'upstream' and r['revision'] == 'fixed')['receipt']['exitstatus'] = 1
    assert summarize_task('liger_attnres', regress, True)['status'] == 'not_qualified'


def test_infrastructure_failure_is_not_bug_reproduction():
    for stage, status, error in [('original_package_import', 'infrastructure_error', 'numerical mismatch'),
                                 ('probe', 'probe_exception', 'CUDA out of memory'),
                                 ('environment', 'infrastructure_error', 'CUDA unavailable')]:
        assert not bug_reproduced('liger_attnres', {'stage': stage, 'status': status, 'error': error})
        assert not bug_reproduced('liger_groupnorm', {'stage': stage, 'status': status, 'error': error})
    assert not suite_pass({'exitstatus': 0, 'rows': [{'outcome': 'skipped'}]})
    assert not suite_pass({'exitstatus': 0, 'rows': []})


@pytest.mark.parametrize('attack', ['traversal', 'symlink'])
def test_archive_extraction_rejects_paths_outside_root(tmp_path, attack):
    source = tmp_path / 'source.tgz'
    with tarfile.open(source, 'w:gz') as tar:
        entry = tarfile.TarInfo('repo/../../escaped' if attack == 'traversal' else 'repo/link')
        if attack == 'symlink':
            entry.type = tarfile.SYMTYPE; entry.linkname = '../../escaped'
            tar.addfile(entry)
        else:
            entry.size = 1; tar.addfile(entry, io.BytesIO(b'x'))
    with pytest.raises(ValueError):
        safe_extract(source, tmp_path / 'destination')
    assert not (tmp_path / 'escaped').exists()
