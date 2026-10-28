"""K3: trusted C++/nvcc compatibility control, executed only under C1."""
import torch
from torch.utils.cpp_extension import load_inline
extension = load_inline(name='c1_cuda_add', cpp_sources='torch::Tensor plus_one(torch::Tensor);',
    cuda_sources=r'''
#include <torch/extension.h>
__global__ void plus_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + 1.0f;
}
torch::Tensor plus_one(torch::Tensor x) {
    auto y = torch::empty_like(x);
    plus_kernel<<<(x.numel()+255)/256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
    return y;
}
''', functions=['plus_one'], extra_cflags=['-O0'], extra_cuda_cflags=['-O0'], with_cuda=True)
x = torch.arange(1024, dtype=torch.float32, device='cuda')
y = extension.plus_one(x)
torch.cuda.synchronize()
assert torch.equal(y, x+1)
print('CUDA_CPP_OK', flush=True)
