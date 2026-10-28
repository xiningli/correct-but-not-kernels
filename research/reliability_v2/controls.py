"""K1/K3: isolated GPU controls for streaming replay, not strict-kernel attribution."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

from env.tasks import verified_catalog
from research.reliability_v2.evaluator import evaluate_reference, evaluate_task
from research.reliability_v2.sandbox import Limits, run, uid_processes
from tools.gatelib import ROOT, require_gate

PREFIX = 'import torch\nimport torch.nn as nn\n'
REFERENCE = PREFIX + '''class ModelNew(nn.Module):
    def forward(self, x):
        return x + 1
'''
TRITON = PREFIX + '''import triton
import triton.language as tl
@triton.jit
def add_one(X, Y, N: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + offsets, offsets < N, other=0.)
    tl.store(Y + offsets, value + 1., offsets < N)
class ModelNew(nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        add_one[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), 256)
        return y
'''
PERSISTENCE = PREFIX + '''class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.buffer = None
    def forward(self, x):
        if self.buffer is None:
            self.buffer = torch.empty_like(x)
        self.buffer.copy_(x + 1)
        return self.buffer
'''
L1_TRITON = PREFIX + '''import triton
import triton.language as tl
@triton.jit
def normalize_l1(X, Y, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, B)
    total = tl.full((), 0., tl.float32)
    for start in range(0, N, B):
        cols = start + offsets
        value = tl.load(X + row * N + cols, cols < N, other=0.)
        total += tl.sum(tl.abs(value), 0)
    mean = total / N
    for start in range(0, N, B):
        cols = start + offsets
        value = tl.load(X + row * N + cols, cols < N, other=0.)
        tl.store(Y + row * N + cols, value / mean, cols < N)
class ModelNew(nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        normalize_l1[(x.shape[0],)](x, y, x.shape[1], 1024, num_warps=8)
        return y
'''


def small_reference():
    import torch
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 1
    return SimpleNamespace(Model=Model, get_init_inputs=lambda: [], get_inputs=lambda: [torch.rand(4096)])


def run_controls(out):
    require_gate('C1'); require_gate('C3')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cases = [
        ('reference', REFERENCE, 'correct', None),
        ('custom_triton', TRITON, 'correct', None),
        ('wrong_shape', REFERENCE.replace('return x + 1', 'return x[:-1] + 1'), 'incorrect', 'shape_dtype'),
        ('wrong_dtype', REFERENCE.replace('return x + 1', 'return (x + 1).double()'), 'incorrect', 'shape_dtype'),
        ('unchanged_nan', REFERENCE.replace('return x + 1', 'return torch.full_like(x, float("nan"))'), 'incorrect', 'nonfinite'),
        ('input_mutation', REFERENCE.replace('return x + 1', 'return x.add_(1)'), 'incorrect', 'input_mutation'),
        ('output_mutation', PERSISTENCE, 'incorrect', 'output_persistence'),
    ]
    checks = []
    for name, source, component, detail in cases:
        result = evaluate_reference(source, small_reference(), out / name, seed=2026101201)
        evidence = [x['component'] for x in result.details.get('checks', [])] + result.hacks
        passed = result.component == component and (detail is None or detail in evidence)
        if name == 'unchanged_nan':
            passed &= 'output_persistence' not in result.hacks
        checks.append({'name': name, 'passed': bool(passed), 'result': asdict(result)})
        (out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        print(json.dumps({'control': name, 'passed': bool(passed), 'component': result.component}), flush=True)
        if not passed:
            raise RuntimeError(f'Control failed: {name}; retain evidence before any replay')
    # Recheck boundary probes on the separate supervisor used by this replay.
    outside = out / 'outside.txt'; outside.write_text('UNCHANGED'); outside.chmod(0o666)
    targets = json.dumps({'write': [str(outside.resolve()), '/tmp/cuda-gemma4-v2-outside'],
                          'read': [str((ROOT / 'research/reliability_v2/evaluator.py').resolve())]}).encode()
    for name, marker in [('socket', None), ('write_outside', 'OUTSIDE_WRITE_BLOCKED'),
                         ('read_oracle', 'ORACLE_READ_BLOCKED')]:
        source = (ROOT / f'tests/hostile/{name}.py').read_text()
        result = run(source, out / name, replace(Limits(), wall_seconds=60), files={'targets.json': targets})
        log = Path(result.log).read_text()
        passed = result.component == 'sandbox_blocked' and (marker is None or marker in log)
        if 'ESCAPE:' in log or outside.read_text() != 'UNCHANGED':
            raise RuntimeError('Possible sandbox escape; STOP for operator')
        checks.append({'name': name, 'passed': passed, 'result': asdict(result)})
        (out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        if not passed or uid_processes():
            raise RuntimeError(f'Isolation regression failed: {name}')
    task = next(t for t in verified_catalog() if t.task_id == 'level1/38_L1Norm_')
    replay_source = task.source() + '\nclass ModelNew(Model):\n    pass\n'
    result = evaluate_task(replay_source, task, out / 'full_size_reference', seed=2026101201)
    checks.append({'name': 'full_size_reference', 'passed': result.correct, 'result': asdict(result)})
    (out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    if not result.correct:
        raise RuntimeError('Full-size reference did not pass streaming storage/correctness control')
    result = evaluate_task(L1_TRITON, task, out / 'full_size_custom', seed=2026101201)
    checks.append({'name': 'full_size_custom', 'passed': result.correct, 'result': asdict(result)})
    (out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    if not result.correct:
        raise RuntimeError('Full-size custom kernel did not pass storage/correctness control')
    return checks
