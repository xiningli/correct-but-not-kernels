"""K2: known-fused equivalent of training task 88_MinGPTNewGelu, C2 only."""
import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    x = tl.load(X+i, i<N, 0)
    z = 0.7978845608028654 * (x + 0.044715*x*x*x)
    y = 0.5*x*(1 + 2/(1+tl.exp(-2*z))-1)
    tl.store(Y+i, y, i<N)

def fused(x):
    y = torch.empty_like(x)
    gelu_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), 1024)
    return y
