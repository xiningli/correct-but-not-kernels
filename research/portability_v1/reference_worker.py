"""Trusted pinned-reference qualification; run without any candidate mounted."""
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback
import torch
from safetensors.torch import save_file


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    cfg = json.load(open('/input/request.json')); plan = cfg['plan']
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(plan['reference_gpu_fraction'])
    spec = importlib.util.spec_from_file_location('trusted_reference', '/input/reference.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = {'eligible': False, 'samples': [], 'reason': None, 'error': None}
    total_bytes = 0
    try:
        for sample, seed in enumerate(cfg['seeds']):
            root = Path(f'sample-{sample}'); inputs = root / 'inputs'; oracle = root / 'oracle'
            inputs.mkdir(parents=True); oracle.mkdir()
            with torch.inference_mode():
                torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(seed); init = module.get_init_inputs()
                torch.manual_seed(seed); model = module.Model(*init).eval().cuda()
                torch.save({'init': init}, inputs / 'inputs.pt')
                contracts = []
                for i in range(4):
                    torch.manual_seed(seed + 1009 * (i + 1))
                    values = module.get_inputs()
                    nbytes = sum(x.untyped_storage().nbytes() for x in values if isinstance(x, torch.Tensor))
                    if nbytes > plan['reference_max_case_bytes']:
                        raise MemoryError('reference_input_storage_cap')
                    gpu = [x.cuda() if isinstance(x, torch.Tensor) else x for x in values]
                    wanted = model(*gpu); torch.cuda.synchronize()
                    if type(wanted) is not torch.Tensor or not bool(torch.isfinite(wanted).all()):
                        raise ValueError('reference_output_type_or_nonfinite')
                    expected = wanted.detach().cpu()
                    if expected.numel() * expected.element_size() > plan['reference_max_case_bytes']:
                        raise MemoryError('reference_output_storage_cap')
                    repeat = model(*gpu); torch.cuda.synchronize()
                    if not torch.allclose(repeat.cpu(), expected, rtol=1e-3, atol=1e-4):
                        raise ValueError('reference_repeat_disagreement')
                    torch.save(values, inputs / f'case-{i}.pt')
                    save_file({'output': expected.contiguous()}, oracle / f'expected-{i}.safetensors')
                    contracts.append({'shape': list(expected.shape), 'dtype': str(expected.dtype)})
                    total_bytes += (inputs / f'case-{i}.pt').stat().st_size + (oracle / f'expected-{i}.safetensors').stat().st_size
                    if total_bytes > plan['reference_max_task_storage_bytes']:
                        raise MemoryError('reference_total_storage_cap')
                    del values, gpu, wanted, expected, repeat
                peak = torch.cuda.max_memory_allocated()
                if peak > plan['reference_max_allocated_bytes']:
                    raise MemoryError('reference_peak_gpu_cap')
                (inputs / 'request.json').write_text(json.dumps({'model_seed': seed, 'output_contracts': contracts}))
                result['samples'].append({'sample': sample, 'seed': seed, 'peak_allocated_bytes': peak,
                    'files': {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}})
                del model
                torch.cuda.empty_cache()
        result['eligible'] = True
    except (torch.OutOfMemoryError, MemoryError) as e:
        result.update(reason='reference_resource_exclusion', error=str(e)[-2000:])
    except Exception as e:
        result.update(reason='reference_unavailable', error=f'{type(e).__name__}: {e}'[-2000:])
        traceback.print_exc()
    Path('reference-result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
