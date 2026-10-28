"""Equal Triton ABI for complete-program versus supplied-host interventions.

Development only. The wrapper allocates and launches; all mathematical work,
dtype conversion, bounds handling and tensor indexing remain model-written.
No candidate is imported here. AST validation never repairs candidate code.
"""
import ast
import json

from env.reward import _FENCE, _THINK
from research.reliability_v2.new_tasks import catalog

CONDITIONS = ('full_program', 'fixed_host')
SIGNATURE = ('X, Y, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, '
             'SX0: tl.constexpr, SX1: tl.constexpr, SY0: tl.constexpr, SY1: tl.constexpr, '
             'BLOCK: tl.constexpr')
ABI = '''Use a @triton.jit function named run_kernel with this exact argument order:
def run_kernel(''' + SIGNATURE + '''):
    # implement the task here
X, Y, O are tensor pointers. X and Y retain their declared strides and dtype.
O is fresh contiguous float32 storage. Grid is (M,); program_id(0) identifies
an output row. M = x.shape[0], K = x.shape[1]. For matrix multiplication,
N = y.shape[1]; otherwise N = x.shape[1]. BLOCK = triton.next_power_of_2(max(N, K)).
SX0/SX1 and SY0/SY1 are x.stride() and y.stride(). The host passes Tensor objects,
not integer data_ptr values. Implement all math, pointer indexing, dtype casts,
and tail masks inside your kernel. Output has shape (M,) for row_sum and (M,N)
for the other development tasks. No initialization arguments.
'''


def extract_code(completion):
    """Same fenced-code/thought handling, without v1's full-class requirement.

Both intervention conditions use this extractor. The condition-specific AST
contract is enforced by assemble; original archived extraction is unchanged.
"""
    text = _THINK.sub('', completion)
    if '<|channel>thought' in text:
        return None
    blocks = _FENCE.findall(text)
    if not blocks:
        return None
    try:
        ast.parse(blocks[-1])
    except (SyntaxError, ValueError):
        return None
    return blocks[-1]


def check_task(task):
    known = {t['id']: t for t in catalog() if t['split'] == 'development'}
    if known.get(task['id']) != task:
        raise ValueError('Only exact registered development tasks are enabled')


def wrapper(task):
    check_task(task)
    n = 'y.shape[1]' if task['matrix_inputs'] else 'x.shape[1]'
    shape = '(m,)' if task['id'] == 'development/row_sum' else '(m, n)'
    return f'''\nimport torch
import torch.nn as nn
class ModelNew(nn.Module):
    def forward(self, x, y):
        m, k = x.shape
        n = {n}
        out = torch.empty({shape}, device=x.device, dtype=torch.float32)
        run_kernel[(m,)](x, y, out, m, n, k, x.stride(0), x.stride(1),
                        y.stride(0), y.stride(1), triton.next_power_of_2(max(n, k)))
        return out
'''


def messages(task, condition):
    check_task(task)
    if condition not in CONDITIONS:
        raise ValueError('Unknown interface condition')
    common = '''Implement the mathematical reference using custom Triton kernels.
PyTorch is allowed only for empty output allocation and tensor metadata/views.
Do not use PyTorch arithmetic, dtype conversions, vendor compute, CPU math on
tensor values, runtime inspection or reference replay. Preserve inputs and
return fresh output. Return exactly one Python code block without explanation.
''' + ABI
    if condition == 'full_program':
        specific = ('Return the complete program, including imports, run_kernel and '
                    'ModelNew(torch.nn.Module) with forward(self, x, y). Implement '
                    'the host allocation and kernel launch yourself using the ABI above.')
    else:
        specific = ("Return only imports of triton/triton.language and @triton.jit kernel "
                    "definitions, including run_kernel. Do not define ModelNew or other "
                    "host code. We append this fixed host wrapper unchanged:\n```python\n"
                    + wrapper(task) + '```')
    content = ('Task: ' + task['id'] + '\nValid input domain: '
               + json.dumps(task['domain'], sort_keys=True)
               + '\nOutput dtype: float32.\n```python\n' + task['reference_source'] + '```')
    return [{'role': 'system', 'content': common + '\n' + specific},
            {'role': 'user', 'content': content}]


def assemble(source, task, condition):
    """Syntactic contract only; numerical/runtime verification remains in C1."""
    check_task(task)
    if condition not in CONDITIONS:
        raise ValueError('Unknown condition')
    tree = ast.parse(source)
    entrypoints = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_kernel']
    expected_args = ['X', 'Y', 'O', 'M', 'N', 'K', 'SX0', 'SX1', 'SY0', 'SY1', 'BLOCK']
    if (len(entrypoints) != 1 or [a.arg for a in entrypoints[0].args.args] != expected_args
            or entrypoints[0].args.vararg or entrypoints[0].args.kwarg
            or entrypoints[0].args.posonlyargs or entrypoints[0].args.kwonlyargs):
        raise ValueError('Both conditions require the identical run_kernel ABI')
    if condition == 'full_program':
        if sum(isinstance(n, ast.ClassDef) and n.name == 'ModelNew' for n in tree.body) != 1:
            raise ValueError('full_program requires exactly one ModelNew class')
        return source
    functions = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(a.name not in {'triton', 'triton.language'} for a in node.names):
                raise ValueError('fixed_host allows only Triton imports and JIT definitions')
        elif isinstance(node, ast.ImportFrom):
            if node.module not in {'triton', 'triton.language'}:
                raise ValueError('fixed_host import outside contract')
        elif isinstance(node, ast.FunctionDef):
            if len(node.decorator_list) != 1 or ast.unparse(node.decorator_list[0]) not in {'triton.jit', 'jit'}:
                raise ValueError('fixed_host permits only @triton.jit functions')
            if node.name == 'run_kernel':
                functions.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        else:
            raise ValueError('fixed_host contains host code; do not silently discard it')
    if len(functions) != 1:
        raise ValueError('Need exactly one run_kernel definition')
    return 'import triton\nimport triton.language as tl\n' + source + '\n' + wrapper(task)


def control_kernel(task):
    """Handwritten controls never enter model prompts or count as model successes."""
    check_task(task)
    name = task['id'].split('/')[-1]
    prefix = ('import triton\nimport triton.language as tl\n@triton.jit\n'
              'def run_kernel(' + SIGNATURE + '):\n'
              '    r = tl.program_id(0)\n    c = tl.arange(0, BLOCK)\n')
    load = '    v = tl.load(X + r*SX0 + c*SX1, c < N, other=0).to(tl.float32)\n'
    store = '    tl.store(O + r*N + c, z, c < N)\n'
    bodies = {
        'add_one': load + '    z = v + 1.0\n' + store,
        'row_sum': load + '    tl.store(O + r, tl.sum(v, 0))\n',
        'row_softmax': load + '    v = tl.where(c < N, v, -float("inf"))\n'
                       '    e = tl.exp(v - tl.max(v, 0))\n    z = e / tl.sum(e, 0)\n' + store,
        'reverse': '    z = tl.load(X + r*SX0 + (N-1-c)*SX1, c < N, other=0).to(tl.float32)\n' + store,
        'center': load + '    z = v - tl.sum(v, 0) / N\n' + store,
        'matrix': '    z = tl.full((BLOCK,), 0, tl.float32)\n'
                  '    for k in range(K):\n'
                  '        a = tl.load(X + r*SX0 + k*SX1).to(tl.float32)\n'
                  '        b = tl.load(Y + k*SY0 + c*SY1, c < N, other=0).to(tl.float32)\n'
                  '        z = z + a*b\n' + store,
    }
    return prefix + bodies[name]
