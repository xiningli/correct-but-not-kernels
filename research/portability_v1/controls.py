"""Local positive/negative evaluator and container checks before replay."""
import json
from pathlib import Path
import shutil
import torch
from safetensors.torch import save_file
from .backend import run
from .judge import judge
from .common import write, read, utc, PYTHON, IMAGE

TRITON = '''import torch
import torch.nn as nn
import triton
import triton.language as tl
@triton.jit
def add_one(X,Y,N:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 tl.store(Y+i,tl.load(X+i,i<N,0)+1,i<N)
class ModelNew(nn.Module):
 def forward(self,x):
  y=torch.empty_like(x)
  add_one[(triton.cdiv(x.numel(),256),)](x,y,x.numel(),256)
  return y
'''
CUDA = '''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
ext=load_inline(name="port_control",cpp_sources="torch::Tensor add_one(torch::Tensor x);",
 cuda_sources=r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void add_one_kernel(const float* x,float* y,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[i]+1;
}
torch::Tensor add_one(torch::Tensor x){
 auto y=torch::empty_like(x);
 add_one_kernel<<<(x.numel()+255)/256,256>>>(x.data_ptr<float>(),y.data_ptr<float>(),x.numel());
 return y;
}
""",functions=["add_one"],verbose=False)
class ModelNew(nn.Module):
 def forward(self,x):return ext.add_one(x)
'''
TORCH = '''import torch
import torch.nn as nn
class ModelNew(nn.Module):
 def forward(self,x):return x+1
'''


def controls(directory, plan):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / 'inputs'; oracle = directory / 'oracle'
    inputs.mkdir(); oracle.mkdir()
    torch.save({'init': []}, inputs / 'inputs.pt')
    for i in range(4):
        x = torch.arange(1024, dtype=torch.float32) + 100 * i
        torch.save([x], inputs / f'case-{i}.pt')
        save_file({'output': x + 1}, oracle / f'expected-{i}.safetensors')
    write(inputs / 'request.json', {'model_seed': 1,
        'output_contracts': [{'shape': [1024], 'dtype': 'torch.float32'}] * 4})
    probe = directory / 'environment.py'
    probe.write_text('''import json,os,socket,torch,triton
assert not os.path.exists("/mnt/fastdata/cuda-gemma4-llm")
try: open("/input/forbidden-write","w")
except OSError: pass
else: raise RuntimeError("Input mount writable")
s=socket.socket(); s.settimeout(2)
try:s.connect(("1.1.1.1",443))
except OSError:pass
else:raise RuntimeError("External network reachable")
assert os.geteuid()!=0
print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"triton":triton.__version__,
 "gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability(),
 "uid":os.geteuid(),"filesystem_network_controls":"PASS"}))
assert torch.__version__=="2.13.0+cu130" and triton.__version__=="3.7.1"
''')
    environment = run(probe, inputs, directory / 'environment', plan, seconds=60)
    if environment['returncode'] or environment['reason']:
        raise RuntimeError('Container/environment control failed: ' + environment['log'])
    write(directory / 'environment.json', {
        'image': IMAGE, 'python': PYTHON, 'probe_log': Path(environment['log']).read_text()})
    checks = []
    candidates = [('triton', TRITON, True, True), ('cuda_cpp', CUDA, True, True),
                  ('pytorch', TORCH, True, False),
                  ('wrong_values', TORCH.replace('return x+1', 'return x+2'), False, False),
                  ('input_mutation', TORCH.replace('return x+1', 'return x.add_(1)'), False, False),
                  ('persistent_wrong', TORCH.replace('return x+1', 'return torch.arange(1024,device=x.device,dtype=x.dtype)+1'), False, False)]
    for name, source, numeric, compliant in candidates:
        result = judge(source, inputs, oracle, directory / name, plan)
        passed = result['numerically_correct'] is numeric and result['contract_compliant'] is compliant
        if name == 'input_mutation':
            passed &= 'input_mutation' in result.get('flags', [])
        if name == 'persistent_wrong':
            passed &= 'fresh_input_failure' in result.get('flags', [])
        checks.append({'name': name, 'passed': bool(passed), 'result': result})
        write(directory / 'controls.partial.json', checks)
        if not passed:
            raise RuntimeError('GPU positive/negative control failed: ' + name)
    # Confirm the dedicated container is stopped at a bounded wall time.
    sleeper = directory / 'timeout.py'; sleeper.write_text('import time\ntime.sleep(600)\n')
    expired = run(sleeper, inputs, directory / 'timeout', plan, seconds=3)
    if expired['reason'] != 'timeout' or expired['container_state'].get('Running'):
        raise RuntimeError('Container timeout cleanup failed')
    checks.append({'name': 'timeout_cleanup', 'passed': True})
    shutil.rmtree(environment['work'], ignore_errors=True)
    receipt = {'status': 'PASS', 'utc': utc(), 'checks': checks, 'model_outputs': 0}
    write(directory / 'receipt.json', receipt)
    return receipt
