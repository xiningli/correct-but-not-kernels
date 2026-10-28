"""CPU input/oracle construction. This file is never exposed to candidate code."""
from __future__ import annotations

TASKS = {
    'liger_attnres': {
        'slug': 'linkedin--Liger-Kernel-1344', 'repo': 'linkedin/Liger-Kernel',
        'path': 'src/liger_kernel/ops/attn_res.py', 'source_subdir': 'src',
        'owner': 'liger_kernel.ops.attn_res',
        'protected': ['attn_res_forward', 'attn_res_backward', 'LigerAttnResFunction.forward', 'LigerAttnResFunction.backward'],
        'kernel': '_attn_res_fwd_kernel', 'tests': ['test/transformers/test_attn_res.py'],
        'issue': 'Attention residual aggregation must use every input block. With more than 32 blocks, output or gradients disagree with the depth-softmax weighted sum. Preserve the supported dtype, shape and public API contracts.',
        'contract': 'V [blocks,batch,sequence,width], query/norm [width]; depth-softmax weighted sum after RMS normalization, eps=1e-6. Test FP32 and FP16, output and all first derivatives. Inputs must be preserved.',
        'visible': [dict(id='v_bug', kind='attn', n=40, d=19), dict(id='v_preserve', kind='attn', n=3, d=19)],
        'hidden': [dict(id='h_boundary_low', kind='attn', n=31, d=19), dict(id='h_boundary', kind='attn', n=32, d=19),
                   dict(id='h_boundary_high', kind='attn', n=33, d=19), dict(id='h_larger', kind='attn', n=65, d=31),
                   dict(id='h_fp16', kind='attn', n=4, d=32, dtype='float16')],
    },
    'liger_groupnorm': {
        'slug': 'linkedin--Liger-Kernel-1376', 'repo': 'linkedin/Liger-Kernel',
        'path': 'src/liger_kernel/ops/group_norm.py', 'source_subdir': 'src',
        'owner': 'liger_kernel.ops.group_norm',
        'protected': ['group_norm_forward', 'group_norm_backward', 'LigerGroupNormFunction.forward', 'LigerGroupNormFunction.backward'],
        'kernel': '_group_norm_forward_kernel', 'tests': ['test/transformers/test_group_norm.py'],
        'issue': 'GroupNorm accepts spatial inputs, but backward on 4D/5D tensors can fail or access invalid memory. Match the GroupNorm output and input/affine gradients, preserving existing 3D behavior.',
        'contract': 'Contiguous FP32 x [batch,channels,*spatial], channels divisible by groups, affine weight/bias [channels], eps=1e-6; output and all first derivatives. Inputs must be preserved.',
        'visible': [dict(id='v_bug', kind='group', shape=[2,6,4,8], groups=3), dict(id='v_preserve', kind='group', shape=[2,6,17], groups=3)],
        'hidden': [dict(id='h_5d', kind='group', shape=[2,8,2,3,5], groups=4),
                   dict(id='h_odd', kind='group', shape=[2,9,3,7], groups=3),
                   dict(id='h_one_group', kind='group', shape=[2,6,19], groups=1),
                   dict(id='h_channel_groups', kind='group', shape=[2,6,19], groups=6)],
    },
    'fla_rwkv7': {
        'slug': 'fla-org--flash-linear-attention-1103', 'repo': 'fla-org/flash-linear-attention',
        'path': 'fla/ops/rwkv7/channel_mixing.py', 'source_subdir': '.',
        'owner': 'fla.ops.rwkv7.channel_mixing',
        'protected': ['rwkv_relu_and_square_fwd', 'channel_mixing_rwkv7', 'Rwkv7ChannelMixing.forward', 'Rwkv7ChannelMixing.backward'],
        'kernel': 'rwkv_channel_mixing_pow_and_relu', 'tests': ['tests/ops/test_rwkv7.py', '-k', 'channel_mixing_gradients'],
        'issue': 'RWKV7 channel mixing must not write beyond logical tensor bounds for non-tile-multiple sizes. Preserve outputs, gradients, inplace/out-of-place behavior and previous-state layouts.',
        'contract': 'FP32 contiguous inputs. ReLU-square helper supports inplace=True/False and must preserve guard storage. Full channel_mixing_rwkv7 uses inplace=False, accepts previous state [batch,width] or [batch,1,width], returns output and the existing secondary-output API. All first derivatives and input preservation checked. Existing host matmuls are permitted; target GPU operations must still execute custom kernels.',
        'visible': [dict(id='v_bug', kind='tail', n=4103, inplace=True),
                    dict(id='v_preserve', kind='rwkv', shape=[1,64,128,128])],
        'hidden': [dict(id='h_small', kind='tail', n=37, inplace=True), dict(id='h_aligned', kind='tail', n=4096, inplace=True),
                   dict(id='h_outplace', kind='tail', n=4097, inplace=False),
                   dict(id='h_odd', kind='rwkv', shape=[1,3,17,37]),
                   dict(id='h_prev3d', kind='rwkv', shape=[1,5,19,41], prev3d=True)],
    },
}

# Explicitly fixed before model outputs. Hidden values and shape identifiers do
# not enter prompts. These are development tests, never formal independent tasks.
SEEDS = {'visible': [2801], 'hidden': [3901, 3902]}


def make_case(spec, seed):
    import torch
    from research.repair_v8.cases import attnres_reference, groupnorm_reference, rwkv_reference
    gen = torch.Generator(device='cpu').manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=gen)
    kind = spec['kind']
    dtype = getattr(torch, spec.get('dtype', 'float32'))
    if kind == 'attn':
        n, d = spec['n'], spec['d']
        v = rand(n, 1, 3, d)
        if n > 32:
            v[32:] += 3
        inputs = [v, rand(d) * (0 if n > 32 else .03), 1 + rand(d) * .02]
        names = ['v', 'query', 'norm']
        reference = lambda values: attnres_reference(*values)
    elif kind == 'group':
        shape = spec['shape']
        inputs = [rand(*shape), 1 + rand(shape[1]) * .1, rand(shape[1]) * .1]
        names = ['x', 'weight', 'bias']
        reference = lambda values: groupnorm_reference(*values, spec['groups'])
    elif kind == 'rwkv':
        b, t, c, h = spec['shape']
        inputs = [rand(b,t,c)*.1, rand(b,c)*.1, rand(1,1,c)*.1, rand(c,h)*.1, rand(h,c)*.1]
        if spec.get('prev3d'):
            inputs[1] = inputs[1].unsqueeze(1)
        names = ['x', 'previous', 'mixing', 'key', 'value']
        reference = lambda values: rwkv_reference(*values)
    elif kind == 'tail':
        n = spec['n']; window = ((n + 4095) // 4096) * 4096
        storage = torch.full((2 * window,), 37.)
        storage[:n] = rand(n) * .1
        wanted_storage = storage.double().clone()
        if spec['inplace']:
            wanted_storage[:n] = storage[:n].double().clamp_min(0).square()
        return {'storage': storage}, {
            'output': storage[:n].double().clamp_min(0).square(), 'storage_after': wanted_storage,
        }
    else:
        raise ValueError('Unknown trusted case')
    inputs = [x.to(dtype) for x in inputs]
    refs = [x.double().clone().requires_grad_(True) for x in inputs]
    expected = reference(refs)
    grad = rand(*expected.shape).to(dtype)
    expected.backward(grad.double())
    if kind == 'group':
        other = torch.nn.functional.group_norm(refs[0], spec['groups'], refs[1], refs[2], 1e-6)
        torch.testing.assert_close(expected, other, atol=1e-10, rtol=1e-10)
    outputs = {'output': expected.detach(), **{'grad_' + name: x.grad for name, x in zip(names, refs)}}
    outputs.update({'input_' + name: x.double() for name, x in zip(names, inputs)})
    outputs['input_grad_output'] = grad.double().clone()
    if kind == 'rwkv':
        outputs['last'] = inputs[0][-1, :].double().clone()
    if not all(torch.isfinite(x).all() for x in outputs.values()):
        raise ValueError('Nonfinite oracle')
    return {**dict(zip(names, inputs)), 'grad_output': grad}, outputs


def tolerance(name, dtype):
    if name.startswith('input_') or name in {'storage_after', 'last'}:
        # storage_after includes computed ReLU values; its guard is separately exact.
        return (2e-4, 1e-4) if name == 'storage_after' else (0., 0.)
    gradient = name.startswith('grad_')
    if dtype == 'float16':
        return (5e-2, 5e-3) if gradient else (1e-2, 1e-3)
    return (1e-3, 1e-4) if gradient else (2e-4, 1e-4)


def public_task(task):
    row = TASKS[task]
    return {k: row[k] for k in ['repo', 'path', 'owner', 'protected', 'issue', 'contract', 'visible', 'kernel']}
