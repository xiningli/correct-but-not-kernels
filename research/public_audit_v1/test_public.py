"""Static classes behind the public-corpus audit (Section 4.5)."""
from research.public_audit_v1.kernelbook_static import classify as kernelbook_class
from research.public_audit_v1.sakana_static import classify, host_gemm_or_conv, profile

KERNEL = '__global__ void bias_relu_kernel(float* x, int n) { }\n'


def test_cuda_launch_sites():
    assert classify(KERNEL + 'void f() { bias_relu_kernel<<<1, 1>>>(p, n); }')[0] == 'launch_site'
    templated = '__global__ void k(T* x) { }\nvoid f() { k<float><<<g, b>>>(p); }'
    assert classify('template <typename T>\n' + templated)[0] == 'launch_site'
    assert classify(KERNEL + 'void f() { cudaLaunchKernel((void*)bias_relu_kernel, g, b, a); }')[0] == 'launch_site'
    bounded = '__global__ void __launch_bounds__(256) k(float* x) { }\nvoid f() { k<<<1, 256>>>(p); }'
    assert classify(bounded) == ('launch_site', ['k'])


def test_cuda_decorative_and_kernel_free():
    assert classify(KERNEL + 'torch::Tensor f(torch::Tensor x) { return torch::relu(x); }')[0] == 'definition_only'
    assert classify('torch::Tensor f(torch::Tensor x) { return at::conv2d(x, w); }')[0] == 'no_declared_kernel'


def test_host_library_compute():
    assert host_gemm_or_conv('auto y = at::conv2d(x, w, b);')
    assert host_gemm_or_conv('auto y = torch::matmul(a, b);')
    # Allocation and elementwise helpers are not GEMM/conv delegation.
    assert not host_gemm_or_conv('auto y = torch::empty_like(x); auto z = torch::relu(y);')


def test_profile_ignores_fill_kernels_and_own_gemm():
    fill = "{'void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, x>(int)': {}}"
    assert profile(fill, ['k'])['vendor_seen'] is False
    own = "{'optimized_tiled_gemm(float const*)': {}}"
    assert profile(own, ['optimized_tiled_gemm'])['vendor_seen'] is False
    assert profile("{'aten::cudnn_convolution': {}}", ['k'])['vendor_seen'] is True


def test_kernelbook_entry_point_alias():
    code = ('import triton\n@triton.jit\ndef triton_poi_0(x):\n    pass\n'
            'def call(args):\n    triton_poi_0[(1,)](args)\n'
            'class FooNew:\n    def forward(self, x):\n        return call([x])\n')
    cls, extern = kernelbook_class({'triton_code': code, 'entry_point': 'Foo'})
    assert cls == 'reachable' and extern == []
