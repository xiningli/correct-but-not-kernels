"""Private C1 worker: observe old programs without changing their input domain.

Assembled with tensor_checks.py. Reference values never enter this process.
Partial observations are retained; absence of a completed trace is unknown.
Model initialization is recorded separately from the forward-compute contract.
"""
import importlib.util
import json
import os
import sys
import traceback
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


def main():
    cfg = json.load(open('request.json'))
    payload = torch.load('inputs.pt', weights_only=True, map_location='cpu')
    torch.set_float32_matmul_precision('highest')
    profile = torch.profiler.profile
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    try:
        probe = torch.ones(256, device='cuda')
        with profile(activities=activities) as calibration:
            probe = probe + 1
            torch.cuda.synchronize()
        calibrated = gpu_names(calibration)
        if not calibrated:
            raise RuntimeError('Missing CUDA calibration events')
        del probe, calibration
    except Exception as error:
        print('DECOMPOSITION_OBSERVER_FAILURE: ' + repr(error), flush=True)
        raise SystemExit(88)
    result = {'calibration_kernels': calibrated, 'initialization_completed': False,
              'forward_completed': False, 'observations': [], 'init_dispatch_ops': [],
              'flags': [], 'output_contract_failure': None, 'candidate_error': None,
              'phase': 'initialization', 'outputs_written': 0}

    def save():
        with open('worker-result.tmp', 'w') as f:
            json.dump(result, f, allow_nan=False)
        os.replace('worker-result.tmp', 'worker-result.json')

    protected = lambda: (torch.profiler.profile, DispatchObservation.__torch_dispatch__,
                         torch.Tensor.cpu, torch.Tensor.clone, torch.cuda.synchronize,
                         torch.load, torch.equal, save_file, json.dump)
    original = protected()
    save()
    init_ops = DispatchObservation()
    try:
        torch.manual_seed(cfg['model_seed'])
        spec = importlib.util.spec_from_file_location('candidate_policy', 'candidate.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        with init_ops:
            spec.loader.exec_module(module)
            torch.manual_seed(cfg['model_seed'])
            model = module.ModelNew(*payload['init']).eval().cuda()
        result['init_dispatch_ops'] = init_ops.ops
        result['initialization_completed'] = True
        result['phase'] = 'forward'
        save()
        previous, snapshot = None, None
        with torch.inference_mode():
            for i, contract in enumerate(cfg['output_contracts']):
                cpu_inputs = torch.load(f'case-{i}.pt', weights_only=True, map_location='cpu')
                os.unlink(f'case-{i}.pt')
                # Deliberately match original v1.1 .cuda() semantics, not v3's
                # new noncontiguous/mixed-dtype input specification.
                inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in cpu_inputs]
                before = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
                warm = DispatchObservation()
                with warm:
                    first = model(*inputs)
                    torch.cuda.synchronize()
                traced = DispatchObservation()
                with profile(activities=activities) as profiler:
                    with traced:
                        output = model(*inputs)
                        torch.cuda.synchronize()
                result['observations'].append({'complete': True, 'cuda_available': True,
                    'dispatch_ops': warm.ops + traced.ops, 'cuda_kernels': gpu_names(profiler)})
                if protected() != original:
                    raise RuntimeError('Candidate changed protected observation APIs')
                if type(output) is not torch.Tensor:
                    result['output_contract_failure'] = 'output_type'
                    break
                if list(output.shape) != contract['shape'] or str(output.dtype) != contract['dtype']:
                    result['output_contract_failure'] = 'shape_dtype'
                    break
                try:
                    save_file({'output': output.detach().cpu().contiguous()}, f'output-{i}.safetensors')
                except Exception as error:
                    print('DECOMPOSITION_OUTPUT_IO_FAILURE: ' + repr(error), flush=True)
                    raise SystemExit(87)
                result['outputs_written'] += 1
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
                save()
                del inputs, before, output, cpu_inputs, first, profiler
        result['forward_completed'] = len(result['observations']) == len(cfg['output_contracts'])
        result['phase'] = 'complete'
    except Exception as error:
        result['candidate_error'] = f'{type(error).__name__}: {error}'[-4000:]
        traceback.print_exc()
    finally:
        result['init_dispatch_ops'] = init_ops.ops
        result['flags'] = sorted(set(result['flags']))
        save()


main()
