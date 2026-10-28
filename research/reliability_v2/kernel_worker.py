"""K1/K3: private v2 runner; stream inputs, check output contract before storage.

Assembled by the trusted supervisor; candidates still execute only in C1
isolation. Values and references never enter the worker's input contract.
This does not yet implement the prospective custom-kernel-only contract.
"""
import importlib.util
import json
import os
import sys
import time
import torch
from safetensors.torch import save_file


def execute_worker_v2():
    cfg = json.load(open('request.json'))
    payload = torch.load('inputs.pt', map_location='cpu', weights_only=True)
    torch.set_float32_matmul_precision('highest')
    protected = lambda: (torch.cuda.Event, torch.cuda.synchronize, torch.equal,
                         torch.allclose, torch.Tensor.cpu, torch.Tensor.clone,
                         torch.load, torch.isfinite, time.perf_counter, save_file, json.dump)
    original_api = protected()
    torch.manual_seed(cfg['model_seed'])
    spec = importlib.util.spec_from_file_location('candidate_policy', 'candidate.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(cfg['model_seed'])
    model = module.ModelNew(*payload['init']).eval().cuda()
    flags, previous, snapshot = [], None, None
    result = {'flags': flags, 'timing': None, 'output_contract_failure': None}
    if protected() != original_api:
        raise RuntimeError('Candidate changed protected evaluator APIs')
    with torch.inference_mode():
        for i, contract in enumerate(cfg['output_contracts']):
            filename = f'case-{i}.pt'
            cpu_inputs = torch.load(filename, map_location='cpu', weights_only=True)
            os.unlink(filename)
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in cpu_inputs]
            before = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
            output = model(*inputs)
            torch.cuda.synchronize()
            if protected() != original_api:
                raise RuntimeError('Candidate changed protected evaluator APIs')
            if type(output) is not torch.Tensor:
                result['output_contract_failure'] = 'output_type'
                break
            if list(output.shape) != contract['shape'] or str(output.dtype) != contract['dtype']:
                result['output_contract_failure'] = 'shape_dtype'
                break
            try:
                save_file({'output': output.detach().cpu().contiguous()}, f'output-{i}.safetensors')
            except Exception as error:
                if any(x in str(error) for x in ('No space left on device', 'Disk quota exceeded', 'File too large')):
                    print('V2_OUTPUT_SERIALIZATION_IO_FAILURE: ' + str(error), flush=True)
                    raise SystemExit(87) from error
                raise
            flags.extend(input_flags_v2(inputs, before, output))
            if previous is not None and not same_values(previous, snapshot):
                flags.append('output_persistence')
            saved = output.clone()
            for value in inputs:
                if isinstance(value, torch.Tensor):
                    value.fill_(0)
            if not same_values(output, saved):
                flags.append('output_persistence')
            for value, old in zip(inputs, before):
                if isinstance(value, torch.Tensor):
                    value.copy_(old)
            previous, snapshot = output, saved
            if i == 0:
                timer = Timer(TimingConfig(samples=cfg.get('timing_samples', 31)))
                fn = lambda: model(*inputs)
                timer.warm(fn)
                result['timing'] = timer.measure(fn)
            del inputs, before, output, cpu_inputs
    result['flags'] = sorted(set(flags))
    if protected() != original_api:
        raise RuntimeError('Candidate changed protected evaluator APIs')
    with open('worker-result.json', 'w') as f:
        json.dump(result, f, allow_nan=False)


execute_worker_v2()
