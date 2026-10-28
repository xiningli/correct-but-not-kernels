"""K3/T1: private C1 worker with per-input dispatch and GPU execution traces.

Assembled with tensor_checks.py by the supervisor. Never import this module as
a library. Oracle values are absent; observation is separate from timing.
"""
import importlib.util
import json
import sys
import time
import torch
from safetensors.torch import save_file
from torch.utils._python_dispatch import TorchDispatchMode


class DispatchObservation(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def gpu_names(profiler):
    return [x.name for x in profiler.events() if x.device_type == torch.autograd.DeviceType.CUDA]


def cuda_inputs_preserving_storage(values):
    # Tensor.cuda() may compact a strided slice and erase the very layout bug
    # being tested. Move backing storage once, then rebuild each original view.
    cache, result = {}, []
    for x in values:
        if not isinstance(x, torch.Tensor):
            result.append(x)
            continue
        key = (x.untyped_storage().data_ptr(), x.dtype)
        if key not in cache:
            count = x.untyped_storage().nbytes() // x.element_size()
            cache[key] = torch.as_strided(x, (count,), (1,), storage_offset=0).cuda()
        result.append(torch.as_strided(cache[key], x.shape, x.stride(), x.storage_offset()))
    return result


def strict_main():
    cfg = json.load(open('request.json'))
    payload = torch.load('inputs.pt', weights_only=True, map_location='cpu')
    torch.set_float32_matmul_precision('highest')
    profile = torch.profiler.profile
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    # Mandatory independent GPU event calibration: empty traces cannot silently
    # become either a compliant candidate or a model failure.
    try:
        probe = torch.ones(256, device='cuda')
        with profile(activities=activities) as calibration:
            probe = probe + 1
            torch.cuda.synchronize()
        calibration_kernels = gpu_names(calibration)
        if not calibration_kernels:
            raise RuntimeError('CUDA profiler returned no calibration kernel')
        del probe, calibration
    except Exception as error:
        print('STRICT_OBSERVER_FAILURE: ' + repr(error), flush=True)
        raise SystemExit(88)
    protected = lambda: (torch.profiler.profile, DispatchObservation.__torch_dispatch__,
                         torch.Tensor.cpu, torch.Tensor.clone, torch.cuda.synchronize,
                         torch.load, torch.equal, save_file, json.dump)
    original_api = protected()
    torch.manual_seed(cfg['model_seed'])
    spec = importlib.util.spec_from_file_location('candidate_policy', 'candidate.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    init_observation = DispatchObservation()
    with init_observation:
        spec.loader.exec_module(module)
        torch.manual_seed(cfg['model_seed'])
        model = module.ModelNew(*payload['init']).eval().cuda()
    result = {'flags': [], 'output_contract_failure': None,
              'calibration_kernels': calibration_kernels, 'observations': [],
              'init_dispatch_ops': init_observation.ops, 'candidate_seconds': []}
    if protected() != original_api:
        raise RuntimeError('Candidate changed protected observation APIs')
    previous, snapshot = None, None
    with torch.inference_mode():
        for i, contract in enumerate(cfg['output_contracts']):
            cpu_inputs = torch.load(f'case-{i}.pt', weights_only=True, map_location='cpu')
            inputs = cuda_inputs_preserving_storage(cpu_inputs)
            before = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
            warm = DispatchObservation()
            begin = time.perf_counter()
            with warm:
                first = model(*inputs)
                torch.cuda.synchronize()
            traced = DispatchObservation()
            with profile(activities=activities) as observation:
                with traced:
                    output = model(*inputs)
                    torch.cuda.synchronize()
            result['candidate_seconds'].append(time.perf_counter() - begin)
            result['observations'].append({'complete': True, 'cuda_available': True,
                'dispatch_ops': warm.ops + traced.ops, 'cuda_kernels': gpu_names(observation)})
            if protected() != original_api:
                raise RuntimeError('Candidate changed protected observation APIs')
            if type(output) is not torch.Tensor:
                result['output_contract_failure'] = 'output_type'
                break
            if list(output.shape) != contract['shape'] or str(output.dtype) != contract['dtype']:
                result['output_contract_failure'] = 'shape_dtype'
                break
            save_file({'output': output.detach().cpu().contiguous()}, f'output-{i}.safetensors')
            result['flags'].extend(input_flags_v2(inputs, before, output))
            if previous is not None and not same_values(previous, snapshot):
                result['flags'].append('output_persistence')
            saved = output.clone()
            for value in inputs:
                if isinstance(value, torch.Tensor):
                    value.fill_(0)
            if not same_values(output, saved):
                result['flags'].append('output_persistence')
            for value, old in zip(inputs, before):
                if isinstance(value, torch.Tensor):
                    value.copy_(old)
            previous, snapshot = output, saved
            del inputs, before, output, cpu_inputs, first, observation
    result['flags'] = sorted(set(result['flags']))
    with open('worker-result.json', 'w') as f:
        json.dump(result, f, allow_nan=False)


strict_main()
