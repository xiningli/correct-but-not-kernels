"""Trusted worker text injected after isolation; no references or scoring code.

Do not import this module. Each invocation handles one case or original tests.
Runtime observation is bounded, not a proof against arbitrary native tampering.
"""
import importlib
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
import traceback
import torch
from safetensors.torch import load_file, save_file
from torch.utils._python_dispatch import TorchDispatchMode


class Observe(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def invoke(module, spec, cpu):
    kind = spec['kind']
    values = {k: x.cuda() for k, x in cpu.items()}
    if kind == 'tail':
        storage = values['storage']; view = storage[:spec['n']]
        actual = module.rwkv_relu_and_square_fwd(view, inplace=spec['inplace'])
        flags = []
        alias = actual.data_ptr() == view.data_ptr()
        if alias != spec['inplace']:
            flags.append('inplace_alias_contract')
        return {'output': actual, 'storage_after': storage}, flags
    names = {'attn': ['v', 'query', 'norm'], 'group': ['x', 'weight', 'bias'],
             'rwkv': ['x', 'previous', 'mixing', 'key', 'value']}[kind]
    inputs = [values[n].requires_grad_(True) for n in names]
    extra = {}
    if kind == 'attn':
        api = importlib.import_module('liger_kernel.transformers.functional').liger_attn_res
        actual = api(*inputs, eps=1e-6)
    elif kind == 'group':
        cls = importlib.import_module('liger_kernel.transformers.group_norm').LigerGroupNorm
        layer = cls(inputs[0].shape[1], spec['groups'], eps=1e-6).cuda()
        with torch.no_grad():
            layer.weight.copy_(inputs[1]); layer.bias.copy_(inputs[2])
        actual = layer(inputs[0])
    else:
        actual, last = module.channel_mixing_rwkv7(*inputs, inplace=False)
        extra['last'] = last
    actual.backward(values['grad_output'])
    gradients = [x.grad for x in inputs]
    if kind == 'group':
        gradients[1:] = [layer.weight.grad, layer.bias.grad]
        # Return the actual affine parameters too, to detect persistent mutation.
        inputs[1:] = [layer.weight, layer.bias]
    return {'output': actual, **extra, **{'grad_' + n: g for n, g in zip(names, gradients)},
            **{'input_' + n: x for n, x in zip(names, inputs)}, 'input_grad_output': values['grad_output']}, []


def main():
    cfg = json.load(open('request.json'))
    checkout = Path(cfg['checkout'])
    sys.path[:0] = [str(checkout / cfg['source_subdir']), str(checkout)]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg.get('upstream'):
        print('V8_WORKER_READY', flush=True)
        import pytest
        class Receipt:
            def __init__(self):
                self.rows = []
                self.collection_errors = []
            def pytest_runtest_logreport(self, report):
                if report.when == 'call' or report.failed:
                    self.rows.append({'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome})
            def pytest_collectreport(self, report):
                if report.failed:
                    self.collection_errors.append(str(report.longrepr)[:2048])
        plugin = Receipt()
        args = [str(checkout / a) if not a.startswith('-') and (checkout / a).exists() else a for a in cfg['tests']]
        code = pytest.main(['-q', '-p', 'no:cacheprovider', '--rootdir=' + str(checkout), *args], plugins=[plugin])
        json.dump({'exitstatus': int(code), 'rows': plugin.rows, 'collection_errors': plugin.collection_errors}, open('worker-result.json', 'w'))
        raise SystemExit(int(code))
    record = {'status': 'candidate_error', 'stage': 'observer', 'flags': [], 'ops': [], 'kernels': []}
    memory_check = bool(cfg.get('sanitizer'))
    record['instrumentation'] = 'memcheck' if memory_check else 'profile'
    profile = torch.profiler.profile
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    protected = lambda: (torch.Tensor.cpu, torch.Tensor.clone, torch.cuda.synchronize,
                         torch.profiler.profile, Observe.__torch_dispatch__, load_file, save_file, json.dump)
    originals = protected()
    try:
        with (nullcontext() if memory_check else profile(activities=activities)) as calibration:
            z = torch.ones(32, device='cuda') + 1
            torch.cuda.synchronize()
        names = [] if memory_check else [e.name for e in calibration.events() if str(e.device_type).endswith('CUDA')]
        if not memory_check and not names:
            raise RuntimeError('Empty CUDA calibration trace')
        del z, calibration
    except Exception as e:
        print('PILOT_OBSERVER_FAILURE: ' + repr(e), flush=True)
        raise SystemExit(88)
    cpu = load_file('input.safetensors', device='cpu')
    print('V8_WORKER_READY', flush=True)
    try:
        record['stage'] = 'candidate_import'
        module = importlib.import_module(cfg['owner'])
        if not Path(module.__file__).resolve().is_relative_to(checkout.resolve()):
            raise RuntimeError('Package origin escaped pinned snapshot')
        record['stage'] = 'execution'
        observer = Observe()
        start = time.monotonic()
        with (nullcontext() if memory_check else profile(activities=activities)) as trace:
            with observer:
                output, flags = invoke(module, cfg['case'], cpu)
                torch.cuda.synchronize()
        record['execution_seconds'] = time.monotonic() - start
        record['ops'] = sorted(set(observer.ops))
        record['kernels'] = [] if memory_check else sorted({e.name for e in trace.events() if str(e.device_type).endswith('CUDA')})
        if protected() != originals:
            raise RuntimeError('Protected observer API changed')
        if not all(type(x) is torch.Tensor for x in output.values()):
            raise RuntimeError('Missing or invalid tensor/gradient')
        save_file({k: x.detach().cpu().contiguous() for k, x in output.items()}, 'output.safetensors')
        record.update(status='complete', stage='complete', flags=flags)
    except Exception as e:
        record.update(error_type=type(e).__name__, error=str(e)[:4096], traceback=traceback.format_exc()[-8192:])
    json.dump(record, open('worker-result.json', 'w'), allow_nan=False)
    raise SystemExit(0 if record['status'] == 'complete' else 2)


main()
