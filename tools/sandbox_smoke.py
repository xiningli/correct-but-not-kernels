"""K3: trusted CUDA compatibility probe, never substitutes for hostile C1."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
from env.sandbox import run
from tools.gatelib import ROOT, timestamp

SOURCE = '''import torch
import triton
import triton.language as tl
@triton.jit
def add(X, Y, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(Y+i, tl.load(X+i, i<N, 0)+1, i<N)
x = torch.arange(1024, device='cuda', dtype=torch.float32)
y = torch.empty_like(x)
add[(4,)](x,y,1024)
torch.cuda.synchronize()
assert torch.equal(y,x+1)
print('CUDA_TRITON_OK', flush=True)
'''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpp', action='store_true', help='K3: nvcc/inline C++ compatibility control')
    a = p.parse_args()
    source = (ROOT/'gold/cuda_add.py').read_text() if a.cpp else SOURCE
    result = run(source, ROOT / 'runs/sandbox-smoke' / timestamp())
    print(json.dumps(asdict(result)))
    print(Path(result.log).read_text()[-12000:])
    raise SystemExit(0 if result.component == 'ok' else 1)

if __name__ == '__main__':
    main()
