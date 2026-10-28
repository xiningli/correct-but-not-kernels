"""K3/T1: positive/negative runtime-contract development controls inside C1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from research.reliability_v2.controls import REFERENCE, TRITON
from research.reliability_v2.strict_evaluator import evaluate
from tools.gatelib import ROOT

CUDA = '''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
extension = load_inline(name="strict_add", cpp_sources="torch::Tensor add_one(torch::Tensor x);", cuda_sources=r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void strict_add_kernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1.0f;
}
torch::Tensor add_one(torch::Tensor x) {
  auto y = torch::empty_like(x);
  strict_add_kernel<<<(x.numel()+255)/256,256>>>(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
  return y;
}
""", functions=["add_one"], verbose=False)
class ModelNew(nn.Module):
    def forward(self, x):
        return extension.add_one(x)
'''


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(2026101301)
    inputs = [[torch.randn(n)] for n in [257, 1024, 4099, 13]]
    expected = [x[0] + 1 for x in inputs]
    cases = [
        ('triton_actual_compute', TRITON, 'correct', True),
        ('native_CUDA_actual_compute', CUDA, 'correct', True),
        ('torch_reference_only', REFERENCE, 'contract_violation', True),
        ('unused_triton_definition', TRITON[:TRITON.index('class ModelNew')] +
         REFERENCE[REFERENCE.index('class ModelNew'):], 'contract_violation', True),
        ('dummy_triton_plus_torch_result', TRITON.replace('return y', 'return x + 1'), 'contract_violation', True),
        ('wrong_custom_result', TRITON.replace('value + 1.', 'value + 2.'), 'incorrect', False),
        ('no_kernel_alias', REFERENCE.replace('return x + 1', 'return x'), 'contract_violation', False),
    ]
    checks = []
    for name, source, component, numerical in cases:
        result = evaluate(source, inputs, expected, args.out / name, seed=2026101301)
        passed = result['component'] == component and result['numerically_correct'] == numerical
        checks.append({'name': name, 'passed': passed, 'result': result})
        (args.out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        print(json.dumps({'name': name, 'passed': passed, 'component': result['component'],
                          'error': result.get('error')}), flush=True)
        if not passed:
            raise RuntimeError('Strict observer development control failed; policy evaluation remains disabled')
    sources = {str(x.relative_to(ROOT)): hashlib.sha256(x.read_bytes()).hexdigest()
               for x in sorted((ROOT / 'research/reliability_v2').glob('strict_*.py'))}
    (args.out / 'receipt.json').write_text(json.dumps({
        'status': 'PASS', 'scope': 'development_controls_not_frozen_test_protocol',
        'utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sources,
        'controls': len(checks), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
