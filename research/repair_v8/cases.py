"""Independent development-qualification probes, not sealed formal test tasks.

Original packages are imported normally from a pinned complete checkout. No
kernel extraction, stubs, monkeypatch repair, or production-source rewrites.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import torch


TASKS = {
    'liger_attnres': {
        'pr': 'linkedin/Liger-Kernel#1344', 'slug': 'linkedin--Liger-Kernel-1344',
        'package': 'liger_kernel', 'source_subdir': 'src',
        'owner': 'liger_kernel.ops.attn_res',
        'kernel_name': '_attn_res_fwd_kernel', 'bug_case': 'attnres_40',
        'cases': ['attnres_40', 'attnres_33', 'attnres_31', 'attnres_32', 'attnres_3', 'attnres_fp16'],
        'preserve_cases': ['attnres_3', 'attnres_31', 'attnres_32', 'attnres_fp16'],
        'upstream_tests': ['test/transformers/test_attn_res.py'], 'test_tree': 'test',
        'memcheck_case': 'attnres_33', 'profile_case': 'attnres_3',
        'public_issue': 'Attention residual aggregation must use every input block. For more than 32 blocks its output or gradients disagree with the documented depth-softmax weighted sum. Preserve supported shapes, dtypes and public APIs.',
    },
    'liger_groupnorm': {
        'pr': 'linkedin/Liger-Kernel#1376', 'slug': 'linkedin--Liger-Kernel-1376',
        'package': 'liger_kernel', 'source_subdir': 'src',
        'owner': 'liger_kernel.ops.group_norm',
        'kernel_name': '_group_norm_forward_kernel', 'bug_case': 'groupnorm_4d',
        'cases': ['groupnorm_4d', 'groupnorm_5d', 'groupnorm_odd', 'groupnorm_3d', 'groupnorm_one_group', 'groupnorm_channel_groups'],
        'preserve_cases': ['groupnorm_3d', 'groupnorm_one_group', 'groupnorm_channel_groups'],
        'upstream_tests': ['test/transformers/test_group_norm.py'], 'test_tree': 'test',
        'memcheck_case': 'groupnorm_4d', 'profile_case': 'groupnorm_3d',
        'public_issue': 'LigerGroupNorm forward accepts spatial inputs, but backward on 4D/5D inputs can fail or access invalid memory. Match the public GroupNorm output and input/affine gradients while preserving existing 3D behavior.',
    },
    'flaggems_addmm': {
        'pr': 'flagos-ai/FlagGems#553', 'slug': 'flagos-ai--FlagGems-553',
        'package': 'flag_gems', 'source_subdir': 'src', 'owner': 'flag_gems.ops.addmm',
        'kernel_name': 'addmm_kernel', 'bug_case': 'addmm_matrix',
        'cases': ['addmm_matrix', 'addmm_column', 'addmm_scalar', 'addmm_vector', 'addmm_row', 'addmm_transpose', 'addmm_beta_zero', 'addmm_fp16'],
        'preserve_cases': ['addmm_vector', 'addmm_row', 'addmm_beta_zero', 'addmm_fp16'],
        'upstream_tests': ['tests/test_blas_ops.py', '-k', 'addmm or test_accuracy_mm'], 'test_tree': 'tests',
        'memcheck_case': 'addmm_column', 'profile_case': 'addmm_vector',
        'public_issue': 'The registered addmm operator disagrees with torch.addmm when the additive input is a broadcastable matrix. Honor the documented broadcast, alpha and beta contract without breaking vector-bias inputs.',
    },
    'fla_rwkv7': {
        'pr': 'fla-org/flash-linear-attention#1103', 'slug': 'fla-org--flash-linear-attention-1103',
        'package': 'fla', 'source_subdir': '.', 'owner': 'fla.ops.rwkv7.channel_mixing',
        'kernel_name': 'rwkv_channel_mixing_pow_and_relu', 'bug_case': 'rwkv_tail_guard',
        'cases': ['rwkv_tail_guard', 'rwkv_tail_small', 'rwkv_tail_aligned', 'rwkv_full_odd', 'rwkv_full_aligned', 'rwkv_full_prev3d'],
        'preserve_cases': ['rwkv_tail_aligned', 'rwkv_full_aligned'],
        'upstream_tests': ['tests/ops/test_rwkv7.py', '-k', 'channel_mixing_gradients'], 'test_tree': 'tests',
        'memcheck_case': 'rwkv_full_odd', 'profile_case': 'rwkv_full_aligned',
        'public_issue': 'RWKV7 channel mixing should preserve memory outside its logical tensors for non-tile-multiple sizes, and return the documented outputs and gradients. Preserve both inplace settings and supported previous-state layouts.',
    },
}


def configure_torch():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_owner(task, checkout):
    module = importlib.import_module(TASKS[task]['owner'])
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(Path(checkout).resolve()):
        raise RuntimeError('Imported package escaped pinned checkout: ' + str(origin))
    return module


def tolerance(dtype, gradient=False):
    if dtype == torch.float32:
        return (1e-3, 1e-4) if gradient else (2e-4, 1e-4)
    if dtype == torch.float16:
        return (5e-2, 5e-3) if gradient else (1e-2, 1e-3)
    raise ValueError('Unreviewed dtype')


def check_close(name, actual, expected, dtype, checks, gradient=False):
    assert actual is not None, name + ': missing output/gradient'
    assert actual.shape == expected.shape, name + ': shape mismatch'
    assert actual.dtype == dtype, name + ': dtype mismatch'
    actual, expected = actual.detach().double(), expected.detach().double()
    assert torch.isfinite(actual).all().item(), name + ': nonfinite result'
    assert torch.isfinite(expected).all().item(), name + ': nonfinite oracle'
    atol, rtol = tolerance(dtype, gradient)
    error = (actual - expected).abs()
    checks.append({'name': name, 'max_abs_error': error.max().item(), 'atol': atol, 'rtol': rtol})
    assert (error <= atol + rtol * expected.abs()).all().item(), name + ': numerical mismatch'


def attnres_reference(v, query, norm, eps=1e-6):
    # Independent direct tensor formula in FP64; no upstream reference imported.
    scale = (v.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    keys = v * scale * norm
    scores = (keys * query).sum(dim=-1)
    weights = scores.softmax(dim=0)
    return (weights.unsqueeze(-1) * v).sum(dim=0)


def groupnorm_reference(x, weight, bias, groups, eps=1e-6):
    shape = x.shape
    grouped = x.reshape(shape[0], groups, -1)
    mean = grouped.mean(dim=-1, keepdim=True)
    centered = grouped - mean
    normalized = centered * (centered.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    affine_shape = (1, shape[1]) + (1,) * (len(shape) - 2)
    return normalized.reshape(shape) * weight.reshape(affine_shape) + bias.reshape(affine_shape)


def rwkv_reference(x, previous, mixing, key_weight, value_weight):
    previous = previous.reshape(x.shape[0], 1, x.shape[-1])
    shifted = torch.cat([previous, x[:, :-1]], dim=1)
    mixed = x * (1 - mixing) + shifted * mixing
    activated = (mixed @ key_weight).clamp_min(0).square()
    return activated @ value_weight


def differentiable_reference(values):
    return [x.detach().double().clone().requires_grad_(True) for x in values]


def run_case(task, case, seed, module=None, device='cpu'):
    """With module=None validate only reference arithmetic, never GPU eligibility."""
    if case not in TASKS[task]['cases']:
        raise ValueError('Case outside frozen development list')
    torch.manual_seed(seed)
    checks = []
    rand = lambda *shape: torch.randn(*shape, device=device, dtype=torch.float32)
    if task == 'liger_attnres':
        n = 4 if case == 'attnres_fp16' else int(case.rsplit('_', 1)[1])
        dtype = torch.float16 if case == 'attnres_fp16' else torch.float32
        d = 32 if dtype == torch.float16 else 19
        v = rand(n, 1, 3, d)
        if n > 32:
            v[32:] += 3
        query = rand(d) * (0 if n > 32 else 0.03)
        norm = torch.ones(d, device=device) + rand(d) * 0.02
        inputs = [t.to(dtype).requires_grad_(True) for t in [v, query, norm]]
        refs = differentiable_reference(inputs)
        expected = attnres_reference(*refs)
        grad = rand(1, 3, d).to(dtype)
        expected.backward(grad.double())
        if module:
            # Full original public wrapper, including backend routing and autograd.
            api = importlib.import_module('liger_kernel.transformers.functional').liger_attn_res
            actual = api(*inputs, eps=1e-6)
            actual.backward(grad)
            check_close('output', actual, expected, dtype, checks)
            for name, x, ref in zip(['dV', 'dQuery', 'dNorm'], inputs, refs):
                check_close(name, x.grad, ref.grad, dtype, checks, True)
        else:
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in refs)
    elif task == 'liger_groupnorm':
        shape, groups = {
            'groupnorm_4d': ((2, 6, 4, 8), 3), 'groupnorm_5d': ((2, 8, 2, 3, 5), 4),
            'groupnorm_odd': ((2, 9, 3, 7), 3), 'groupnorm_3d': ((2, 6, 17), 3),
            'groupnorm_one_group': ((2, 6, 17), 1), 'groupnorm_channel_groups': ((2, 6, 17), 6),
        }[case]
        x = rand(*shape).requires_grad_(True)
        weight, bias = rand(shape[1]) * 0.1 + 1, rand(shape[1]) * 0.1
        refs = differentiable_reference([x, weight, bias])
        expected = groupnorm_reference(*refs, groups)
        independently = torch.nn.functional.group_norm(refs[0], groups, refs[1], refs[2], 1e-6)
        torch.testing.assert_close(expected, independently, atol=1e-10, rtol=1e-10)
        grad = rand(*shape)
        expected.backward(grad.double())
        if module:
            cls = importlib.import_module('liger_kernel.transformers.group_norm').LigerGroupNorm
            layer = cls(shape[1], groups, eps=1e-6).to(device)
            with torch.no_grad():
                layer.weight.copy_(weight); layer.bias.copy_(bias)
            actual = layer(x); actual.backward(grad)
            check_close('output', actual, expected, torch.float32, checks)
            for name, value, ref in zip(['dx', 'dw', 'db'], [x.grad, layer.weight.grad, layer.bias.grad], refs):
                check_close(name, value, ref.grad, torch.float32, checks, True)
    elif task == 'flaggems_addmm':
        m, k, n = 5, 13, 17
        dtype = torch.float16 if case == 'addmm_fp16' else torch.float32
        a, b = rand(m, k).to(dtype) * 0.1, rand(k, n).to(dtype) * 0.1
        shape = {'addmm_matrix': (m, n), 'addmm_column': (m, 1), 'addmm_scalar': (),
                 'addmm_row': (1, n), 'addmm_transpose': (m, n)}.get(case, (n,))
        bias = torch.randn(shape, device=device, dtype=dtype)
        if case == 'addmm_matrix':
            bias += torch.arange(m, device=device).reshape(m, 1)
        if case == 'addmm_transpose':
            a = a.T.contiguous().T; b = b.T.contiguous().T
        alpha, beta = 0.75, 0.0 if case == 'addmm_beta_zero' else -0.5
        expected = alpha * (a.double() @ b.double()) + beta * bias.double()
        independent = torch.addmm(bias.double(), a.double(), b.double(), alpha=alpha, beta=beta)
        torch.testing.assert_close(expected, independent, atol=1e-12, rtol=1e-12)
        if module:
            import flag_gems
            saved = [t.clone() for t in [a, b, bias]]
            with flag_gems.use_gems():
                actual = torch.addmm(bias, a, b, alpha=alpha, beta=beta)
            check_close('output', actual, expected, dtype, checks)
            for old, new in zip(saved, [a, b, bias]):
                assert torch.equal(old, new), 'input mutation'
    elif task == 'fla_rwkv7':
        if case.startswith('rwkv_tail'):
            n = {'rwkv_tail_guard': 4103, 'rwkv_tail_small': 37, 'rwkv_tail_aligned': 4096}[case]
            window = ((n + 4095) // 4096) * 4096
            storage = torch.full((2 * window,), 37.0, device=device)
            storage[:n] = rand(n) * 0.1
            view = storage[:n]
            expected = view.double().clamp_min(0).square()
            if module:
                actual = module.rwkv_relu_and_square_fwd(view, inplace=True)
                check_close('relu_square', actual, expected, torch.float32, checks)
                assert actual.data_ptr() == view.data_ptr(), 'inplace alias lost'
                assert (storage[n:] == 37).all().item(), 'out-of-bounds store changed guard'
        else:
            aligned = case == 'rwkv_full_aligned'
            # Both x and FFN counts divide every original autotune block (<=8192).
            batch, length, width, hidden = (1, 64, 128, 128) if aligned else (1, 3, 17, 37)
            values = [rand(batch, length, width) * 0.1, rand(batch, width) * 0.1,
                      rand(1, 1, width) * 0.1, rand(width, hidden) * 0.1, rand(hidden, width) * 0.1]
            if case == 'rwkv_full_prev3d':
                values[1] = values[1].unsqueeze(1)
            inputs = [v.requires_grad_(True) for v in values]
            refs = differentiable_reference(inputs)
            expected = rwkv_reference(*refs)
            grad = rand(*expected.shape)
            expected.backward(grad.double())
            if module:
                saved = [x.detach().clone() for x in inputs]
                actual, last = module.channel_mixing_rwkv7(*inputs, inplace=False)
                actual.backward(grad)
                check_close('output', actual, expected, torch.float32, checks)
                # Preserve the upstream API's actual last-output contract; no silent redefinition.
                assert torch.equal(last, inputs[0][-1, :]), 'secondary output changed'
                for i, (x, ref, original) in enumerate(zip(inputs, refs, saved)):
                    check_close('gradient_' + str(i), x.grad, ref.grad, torch.float32, checks, True)
                    assert torch.equal(x.detach(), original), 'unexpected inplace=False input mutation'
    assert torch.isfinite(expected).all().item(), 'invalid oracle'
    return {'case': case, 'seed': seed, 'mode': 'GPU_original_repository' if module else 'CPU_reference_only',
            'checks': checks, 'reference_shape': list(expected.shape)}
