"""K1/K3: separately versioned correctness replay with bounded streaming storage.

The old evaluator is immutable. This evaluator preserves input/model seeds,
four fresh checks and timing settings; staging and NaN persistence semantics
are changed explicitly. It does NOT certify the strict custom-kernel contract.
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil

from env.correctness import compare, read_output
from env.hacks.detectors import classify, source_flags
from env.run_kernel import RunResult
from research.reliability_v2.sandbox import Limits, run
from research.reliability_v2.storage import HEADROOM, require_free, storage_budget
from tools.gatelib import ROOT, require_gate


def execute(source, files, expected, directory, reference_ms, *, model_seed, caps, timing_samples=31):
    require_gate('C1')
    runner = '\n'.join((ROOT / p).read_text() for p in (
        'env/timing.py', 'research/reliability_v2/tensor_checks.py',
        'research/reliability_v2/kernel_worker.py'))
    request = {'model_seed': model_seed, 'timing_samples': timing_samples,
               'output_contracts': [{'shape': list(x.shape), 'dtype': str(x.dtype)} for x in expected]}
    static = source_flags(source)
    sandbox = run(source, directory, caps, files={**files, 'request.json': json.dumps(request).encode()}, runner=runner)
    details = {'sandbox': asdict(sandbox), 'evaluator': 'reliability-v2-streaming',
               'implementation_contract': 'original_permissive'}
    if sandbox.component != 'ok':
        if sandbox.returncode == 87 and 'V2_OUTPUT_SERIALIZATION_IO_FAILURE:' in Path(sandbox.log).read_text():
            return RunResult('harness_error', details={**details, 'error': 'Output serialization storage failure; do not score as model error'})
        return RunResult(sandbox.component, hacks=static, details=details)
    try:
        work = Path(sandbox.workdir)
        meta = work / 'worker-result.json'
        if meta.is_symlink() or meta.stat().st_size > 1 << 20:
            raise ValueError('Invalid worker metadata')
        data = json.loads(meta.read_text())
        if data['output_contract_failure'] in ('output_type', 'shape_dtype'):
            details['checks'] = [{'correct': False, 'component': data['output_contract_failure']}]
            return RunResult('incorrect', hacks=static, details=details)
        if data['output_contract_failure'] is not None:
            raise ValueError('Invalid worker output contract report')
        checks = [compare(read_output(work / f'output-{i}.safetensors', wanted), wanted)
                  for i, wanted in enumerate(expected)]
        flags = classify(checks, data['flags'], static)
        correct = all(c['correct'] for c in checks) and not flags
        ms = data['timing']['median_ms']
        if type(ms) not in (int, float) or not math.isfinite(ms) or ms <= 0:
            raise ValueError('Invalid candidate timing')
        details.update(checks=checks, candidate_timing=data['timing'], reference_ms=reference_ms)
        return RunResult('correct' if correct else 'incorrect', correct,
                         reference_ms / ms if correct else None, flags, details)
    except Exception as error:
        details['error'] = f'{type(error).__name__}: {error}'
        return RunResult('crash', hacks=static, details=details)


def evaluate_reference(source, reference, directory, *, seed, trials=4, task_id='control', source_sha256=None):
    """Only a trusted reference object may be passed; source executes in C1 only."""
    import torch
    from env.timing import Timer
    require_gate('C3')
    if trials != 4:
        raise ValueError('Replay/control requires the original four checks')
    directory = Path(directory)
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    (directory / 'program.py').write_text(source)
    stage = directory / 'staged-inputs'
    stage.mkdir(mode=0o700)
    details = {'task': task_id, 'source_sha256': source_sha256, 'seed': seed, 'trials': trials,
               'evaluator': 'reliability-v2-streaming', 'implementation_contract': 'original_permissive'}
    result = None
    try:
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(seed)
            init = reference.get_init_inputs()
            torch.manual_seed(seed)
            model = reference.Model(*init).eval().cuda()
            expected, inputs, output_sizes = [], [], []
            # Disk input copies are staged once and consumed one case at a time.
            # Expected values remain only in the trusted parent's CPU memory.
            for i in range(trials):
                torch.manual_seed(seed + 1009 * (i + 1))
                values = reference.get_inputs()
                estimated = sum(x.numel() * x.element_size() for x in values if isinstance(x, torch.Tensor))
                require_free(directory, estimated + HEADROOM + (1 << 20))
                gpu_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in values]
                output = model(*gpu_inputs)
                if type(output) is not torch.Tensor or not bool(torch.isfinite(output).all()):
                    raise RuntimeError('Trusted reference output unsupported or nonfinite')
                expected.append(output.cpu())
                output_sizes.append(output.numel() * output.element_size() + 65544)
                case = stage / f'case-{i}.pt'
                torch.save(values, case)
                case.chmod(0o600)
                inputs.append(case.stat().st_size)
                del values, output, gpu_inputs
            if max(output_sizes) > 16 << 30 or sum(inputs) + sum(output_sizes) > 80 << 30:
                raise RuntimeError('Task exceeds declared tensor envelope')
            torch.save({'init': init}, stage / 'inputs.pt')
            (stage / 'inputs.pt').chmod(0o600)
            budget = storage_budget(inputs, output_sizes)
            details['storage'] = {**budget, **require_free(directory, budget['required_after_staging_bytes'] + (1 << 20))}
            values = torch.load(stage / 'case-0.pt', weights_only=True, map_location='cpu')
            gpu_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in values]
            timer = Timer()
            timer.warm(lambda: model(*gpu_inputs))
            timing = timer.measure(lambda: model(*gpu_inputs))
            details['reference_timing'] = timing
            del values, gpu_inputs, model, timer
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        caps = replace(Limits(), wall_seconds=300, cpu_seconds=270, rss_bytes=80 << 30,
                       gpu_bytes=80 << 30, file_bytes=max(512 << 20, max(output_sizes) + (1 << 20)),
                       work_bytes=sum(inputs) + sum(output_sizes) + HEADROOM)
        files = {p.name: p for p in stage.iterdir()}
        result = execute(source, files, expected, directory / 'candidate', timing['median_ms'],
                         model_seed=seed, caps=caps)
        result.details.update(details)
    except Exception as error:
        result = RunResult('harness_error', details={**details, 'error': f'{type(error).__name__}: {error}'})
    finally:
        # Preserve metadata and hashes, not large reproducible tensor scratch.
        hashes = {}
        work = directory / 'candidate/work'
        if work.exists():
            for path in work.glob('output-*.safetensors'):
                if path.is_file() and not path.is_symlink():
                    with path.open('rb') as stream:
                        hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
            shutil.rmtree(work)
        shutil.rmtree(stage, ignore_errors=True)
        if result is not None:
            result.details['output_sha256'] = hashes
            (directory / 'verdict.json').write_text(json.dumps(asdict(result), indent=2, allow_nan=False) + '\n')
    return result


def evaluate_task(source, task, directory, *, seed):
    from env.tasks import load_reference, split_ids
    if task.task_id not in split_ids()[1] or task.task_id != 'level1/38_L1Norm_':
        raise ValueError('This replay version is scoped to the previously affected held-out task')
    return evaluate_reference(source, load_reference(task), directory, seed=seed,
                              task_id=task.task_id, source_sha256=task.sha256)
