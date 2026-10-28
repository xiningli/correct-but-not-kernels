"""K3/C1: volatile device-side loop must require the supervisor wall cap."""
import torch
import triton
import triton.language as tl
@triton.jit
def spin(X):
    while tl.load(X, volatile=True) == 0:
        pass
x = torch.zeros(1, dtype=torch.int32, device='cuda')
spin[(1,)](x)
print('INFINITE_KERNEL_LAUNCHED', flush=True)
torch.cuda.synchronize()
print('UNEXPECTED_KERNEL_RETURN', flush=True)
