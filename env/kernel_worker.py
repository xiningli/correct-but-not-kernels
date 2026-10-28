"""K1–K3: private in-sandbox runner, assembled with timing/correctness helpers.

Never import this as a task: run_kernel supplies its source through the trusted
bootstrap. No expected outputs/reference are available. Reports stay untrusted;
the parent compares decoded tensors itself and checks the captured API surface.
"""
import importlib.util
import json
import sys
import time
import torch
from safetensors.torch import save_file

def execute_worker():
    cfg = json.load(open('request.json'))
    payload = torch.load('inputs.pt', map_location='cpu', weights_only=True)
    torch.set_float32_matmul_precision('highest')
    protected = lambda: (torch.cuda.Event, torch.cuda.synchronize, torch.equal,
                         torch.allclose, torch.Tensor.cpu, torch.Tensor.clone,
                         time.perf_counter, save_file, json.dump)
    original_api = protected()
    torch.manual_seed(cfg['model_seed'])
    spec = importlib.util.spec_from_file_location('candidate_policy', 'candidate.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(cfg['model_seed'])
    model = module.ModelNew(*payload['init']).eval().cuda()
    flags, previous, snapshot = [], None, None
    result = {'flags': flags, 'timing': None}
    if protected() != original_api:
        raise RuntimeError('Candidate changed protected evaluator APIs')
    with torch.inference_mode():
        for i, cpu_inputs in enumerate(payload['cases']):
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in cpu_inputs]
            before = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
            output = model(*inputs)
            torch.cuda.synchronize()
            if protected() != original_api:
                raise RuntimeError('Candidate changed protected evaluator APIs')
            if type(output) is not torch.Tensor:
                raise TypeError('Single tensor output required by this evaluator version')
            save_file({'output': output.detach().cpu().contiguous()}, f'output-{i}.safetensors')
            if not cfg.get('naive', False):
                flags.extend(input_flags(inputs, before, output))
                if previous is not None and not torch.equal(previous, snapshot):
                    flags.append('output_persistence')
                saved = output.clone()
                # Mutate caller buffers after return without another model call.
                # Correct persistent outputs must retain their original values.
                for value in inputs:
                    if isinstance(value, torch.Tensor):
                        value.fill_(0)
                if not torch.equal(output, saved):
                    flags.append('output_persistence')
                for value, old in zip(inputs, before):
                    if isinstance(value, torch.Tensor):
                        value.copy_(old)
                previous, snapshot = output, saved
            if i == 0:
                timer = Timer(TimingConfig(samples=cfg.get('timing_samples', 31)))
                fn = lambda: model(*inputs)
                timer.warm(fn)
                if cfg.get('naive', False):
                    samples = []
                    for _ in range(timer.config.samples):
                        timer.flush.zero_(); torch.cuda.synchronize()
                        timer.start.record()
                        fn()
                        timer.end.record(); timer.end.synchronize()
                        samples.append(timer.start.elapsed_time(timer.end))
                        torch.cuda.synchronize()
                    result['timing'] = summarize(samples)
                else:
                    result['timing'] = timer.measure(fn)
            del inputs, before, output
    result['flags'] = sorted(set(flags))
    if protected() != original_api:
        raise RuntimeError('Candidate changed protected evaluator APIs')
    with open('worker-result.json', 'w') as f:
        json.dump(result, f, allow_nan=False)

execute_worker()
