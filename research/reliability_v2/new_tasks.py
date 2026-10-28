"""K1/K4, T1/T3: new graph candidates, disjoint development and test IDs.

These are proposed tasks, not a sealed benchmark. Every item has a distinct
operator expression; shape/dtype/layout cases remain checks within that item.
The finite domain, reference and source hashes must be reviewed before freeze.
No assertion is made about absence from pretraining or other benchmarks.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


# Trusted, handwritten expressions. Never evaluate policy-provided expressions.
GRAPHS = {
    'elementwise': {
        'residual_silu': 'x * torch.sigmoid(x) + y',
        'tanh_gate': 'torch.tanh(x) * torch.sigmoid(y)',
        'softplus_product': 'torch.nn.functional.softplus(x) * y',
        'bounded_residual': 'torch.clamp(x + y, -1.0, 1.0) + x',
        'softsign_gate': 'x / (1.0 + x.abs()) * torch.sigmoid(y)',
        'elu_product': 'torch.nn.functional.elu(x) * torch.tanh(y)',
        'gelu_difference': 'torch.nn.functional.gelu(x, approximate="tanh") - torch.nn.functional.gelu(y, approximate="tanh")',
        'softshrink_bias': 'torch.nn.functional.softshrink(x + y, lambd=0.5)',
    },
    'reductions': {
        'weighted_mean': '(x * torch.sigmoid(y)).sum(-1) / torch.sigmoid(y).sum(-1)',
        'centered_energy': '((x - x.mean(-1, keepdim=True)) ** 2).mean(-1)',
        'logsumexp_difference': 'torch.logsumexp(x, -1) - torch.logsumexp(y, -1)',
        'squared_distance': '((x - y) ** 2).sum(-1)',
        'cosine_similarity': '(x * y).sum(-1) / ((x.square().sum(-1) * y.square().sum(-1)).sqrt() + 1e-5)',
        'centered_covariance': '((x - x.mean(-1, keepdim=True)) * (y - y.mean(-1, keepdim=True))).mean(-1)',
        'masked_positive_sum': 'torch.where(y > 0, x, 0.0).sum(-1)',
        'range_plus_mean': 'x.amax(-1) - x.amin(-1) + y.mean(-1)',
    },
    'normalization': {
        'rms_residual': 'x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) + y',
        'l1_gate': 'x / (x.abs().sum(-1, keepdim=True) + 1e-5) * torch.sigmoid(y)',
        'softmax_residual': 'torch.softmax(x, -1) + y',
        'log_softmax_gate': 'torch.log_softmax(x, -1) * torch.sigmoid(y)',
        'minmax_residual': '(x - x.amin(-1, keepdim=True)) / (x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True) + 1e-5) + y',
        'centered_rms_gate': '(x - x.mean(-1, keepdim=True)) * torch.rsqrt((x - x.mean(-1, keepdim=True)).square().mean(-1, keepdim=True) + 1e-5) * torch.sigmoid(y)',
        'softmax_product': 'torch.softmax(x, -1) * torch.softmax(y, -1)',
        'column_rms_bias': 'x * torch.rsqrt(x.square().mean(0, keepdim=True) + 1e-5) + y',
    },
    'indexing': {
        'reverse_product': 'torch.flip(x, [-1]) * y',
        'roll_residual': 'torch.roll(x, 1, -1) + y',
        'causal_gate': 'torch.tril(x) * torch.sigmoid(y)',
        'alternating_columns': 'torch.where(torch.arange(x.shape[1], device=x.device) % 2 == 0, x, y)',
        'row_reverse_sum': 'torch.flip(x, [0]) + y',
        'transpose_tanh': 'torch.tanh(x.transpose(0, 1) + y.transpose(0, 1))',
        'cyclic_difference': 'x - torch.roll(y, 1, 0)',
        'diagonal_mask_residual': 'torch.where(torch.arange(x.shape[0], device=x.device)[:, None] == torch.arange(x.shape[1], device=x.device)[None, :], x, y) + x',
    },
    'matrix_epilogue': {
        'matmul_softsign': '(x @ y) / (1.0 + (x @ y).abs())',
        'matmul_row_center': '(x @ y) - (x @ y).mean(-1, keepdim=True)',
        'matmul_sigmoid_square': 'torch.sigmoid(x @ y).square()',
        'matmul_column_sum': '(x @ y).sum(0)',
        'matmul_row_range': '(x @ y).amax(-1) - (x @ y).amin(-1)',
        'matmul_normalized_energy': '(x @ y).square().sum(-1) / (x.square().sum(-1) + 1e-5)',
        'matmul_log_softmax': 'torch.log_softmax(x @ y, -1)',
        'matmul_centered_tanh': 'torch.tanh((x @ y) - (x @ y).mean(0, keepdim=True))',
    },
    'fused_reduction': {
        'residual_mean_gate': '(x + y) * torch.sigmoid(x.mean(-1, keepdim=True))',
        'softmax_weighted_value': '(torch.softmax(x, -1) * y).sum(-1)',
        'softplus_row_rms': 'torch.nn.functional.softplus(x) * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-5)',
        'row_max_threshold': 'torch.where(x > x.mean(-1, keepdim=True), x * y, y)',
        'centered_product_relu': 'torch.relu(x * y - (x * y).mean(-1, keepdim=True))',
        'row_variance_gate': 'y * torch.sigmoid((x - x.mean(-1, keepdim=True)).square().mean(-1, keepdim=True))',
        'cumulative_residual': 'torch.cumsum(x, -1) + y',
        'neighbor_reduction': '(x * torch.roll(y, 1, -1)).sum(-1)',
    },
}
DEV = {'add_one': 'x + 1.0', 'row_sum': 'x.sum(-1)', 'row_softmax': 'torch.softmax(x, -1)',
       'reverse': 'torch.flip(x, [-1])', 'matrix': 'x @ y', 'center': 'x - x.mean(-1, keepdim=True)'}


def reference_source(expression):
    return ('import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n'
            '    def forward(self, x, y):\n        x, y = x.float(), y.float()\n'
            '        return ' + expression + '\n')


def catalog():
    result = []
    for split, families in [('development', {'development': DEV}), ('test_candidate', GRAPHS)]:
        for family, expressions in families.items():
            for name, expression in expressions.items():
                source = reference_source(expression)
                graph = ast.dump(ast.parse(expression, mode='eval'), include_attributes=False)
                result.append({'id': f'{family}/{name}', 'split': split, 'family': family,
                    'matrix_inputs': family == 'matrix_epilogue' or name == 'matrix',
                    'expression': expression, 'reference_source': source,
                    'reference_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'graph_sha256': hashlib.sha256(graph.encode()).hexdigest(),
                    'output_dtype': 'torch.float32', 'init_inputs': [],
                    'domain': {'rank': 2, 'dtypes': ['float32', 'float16', 'bfloat16'],
                               'layouts': ['contiguous', 'column_stride_2', 'transposed_storage'],
                               'finite_inputs': True, 'max_dimension': 129},
                    'status': 'requires_reference_domain_and_overlap_review'})
    if len({x['id'] for x in result}) != len(result) or len({x['graph_sha256'] for x in result}) != len(result):
        raise ValueError('Duplicate task or operator expression')
    return result


def make_inputs(task, *, seed, case):
    import torch
    generator = torch.Generator().manual_seed(seed)
    m, n, k = [(3, 17, 11), (7, 33, 19), (5, 65, 31), (9, 129, 23)][case % 4]
    dtype = [torch.float32, torch.float16, torch.bfloat16][case % 3]

    def tensor(rows, cols):
        if case % 3 == 1:
            value = torch.randn(rows, cols * 2, generator=generator, dtype=dtype)[:, ::2]
        elif case % 3 == 2:
            value = torch.randn(cols, rows, generator=generator, dtype=dtype).t()
        else:
            value = torch.randn(rows, cols, generator=generator, dtype=dtype)
        return value
    return [tensor(m, k), tensor(k, n)] if task['matrix_inputs'] else [tensor(m, n), tensor(m, n)]


def trusted_reference(task):
    # Require exact identity with our handwritten registry, never arbitrary files.
    expected = {x['id']: x for x in catalog()}.get(task['id'])
    if task != expected:
        raise ValueError('Reference is not in the trusted task registry')
    namespace = {}
    exec(compile(task['reference_source'], '<trusted-new-task>', 'exec'), namespace)
    return namespace['Model']()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--check-cpu', action='store_true')
    args = p.parse_args()
    tasks = catalog()
    report = {'status': 'candidate_manifest_not_frozen', 'tasks': tasks}
    if args.check_cpu:
        import torch
        torch.set_num_threads(2)
        records = []
        for task in tasks:
            model = trusted_reference(task)
            for case in range(4):
                inputs = make_inputs(task, seed=2026101301 + case, case=case)
                before = [x.clone() for x in inputs]
                actual = model(*inputs)
                if actual.dtype != torch.float32 or not bool(torch.isfinite(actual).all()):
                    raise ValueError('Nonfinite or wrong-dtype trusted output: ' + task['id'])
                if any(not torch.equal(x, y) for x, y in zip(inputs, before)):
                    raise ValueError('Trusted reference mutated an input')
                records.append({'task': task['id'], 'case': case, 'shape': list(actual.shape),
                                'input_strides': [list(x.stride()) for x in inputs]})
        report['cpu_domain_checks'] = records
        report['limitation'] = 'CPU finite/domain smoke only; no independent mathematical validation or GPU kernel result.'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'test_candidates': sum(x['split'] == 'test_candidate' for x in tasks),
                      'development': sum(x['split'] == 'development' for x in tasks),
                      'cpu_checks': len(report.get('cpu_domain_checks', [])), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
