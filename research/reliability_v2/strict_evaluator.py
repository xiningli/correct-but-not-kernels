"""K1/K3/K4, T1: correctness + runtime contract, separate from frozen v1.1.

Development evaluator. Do not call it a validated gate before GPU controls pass.
No timing claims: profiling/JIT wall time is recorded only as evaluation cost.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

from env.correctness import compare, read_output
from env.hacks.detectors import classify, source_flags
from research.reliability_v2.strict_sandbox import Limits, run
from research.reliability_v2.strict_contract import (
    ALLOWED_ATEN, assess_observation, declared_kernel_names, source_contract_flags,
)
from tools.gatelib import ROOT, require_gate


def evaluate(source, cases, expected, directory, *, seed, init=()):
    """Only trusted input/oracle tensors enter here; candidate executes in C1.

    Cases can have different valid shapes/layouts/dtypes. The supervisor compares
    output values; only shape/dtype metadata enters the worker. Infrastructure
    errors are distinct and cannot be averaged as unsuccessful model attempts.
    """
    import torch
    require_gate('C1')
    if not cases or len(cases) != len(expected):
        raise ValueError('Every input requires a trusted expected output')
    out = Path(directory)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    (out / 'program.py').write_text(source)
    result = {'component': 'harness_error', 'correct': False, 'numerically_correct': False,
              'contract_compliant': False, 'evaluator': 'strict-runtime-development-v1'}
    stage = out / 'staged-inputs'
    stage.mkdir(mode=0o700)
    try:
        try:
            names = declared_kernel_names(source)
            static = source_flags(source) + source_contract_flags(source)
        except SyntaxError:
            result['component'] = 'format'
            return result
        result['static_flags'] = sorted(set(static))
        result['declared_kernel_names'] = names
        if static:
            result['component'] = 'contract_violation'
            result['reason'] = 'source_access_outside_declared_contract'
            return result
        byte_count = 0
        for i, (inputs, wanted) in enumerate(zip(cases, expected, strict=True)):
            if not isinstance(wanted, torch.Tensor) or not bool(torch.isfinite(wanted).all()):
                raise ValueError('Invalid trusted reference output')
            if any(isinstance(x, torch.Tensor) and x.device.type != 'cpu' for x in inputs):
                raise ValueError('Stage CPU input tensors, preserving their strides')
            path = stage / f'case-{i}.pt'
            torch.save(inputs, path)
            byte_count += path.stat().st_size + wanted.numel() * wanted.element_size()
        if byte_count > 2 << 30:
            raise ValueError('Development task exceeds the 2 GiB envelope')
        if shutil.disk_usage(out).free < byte_count + (8 << 30):
            raise RuntimeError('Insufficient disk space before candidate execution')
        torch.save({'init': list(init)}, stage / 'inputs.pt')
        request = {'model_seed': seed,
                   'output_contracts': [{'shape': list(x.shape), 'dtype': str(x.dtype)} for x in expected]}
        runner = '\n'.join((ROOT / p).read_text() for p in [
            'research/reliability_v2/tensor_checks.py', 'research/reliability_v2/strict_worker.py'])
        caps = replace(Limits(), wall_seconds=300, cpu_seconds=240, rss_bytes=16 << 30,
                       gpu_bytes=16 << 30, work_bytes=byte_count + (8 << 30))
        sandbox = run(source, out / 'candidate', caps,
                      files={**{p.name: p for p in stage.iterdir()}, 'request.json': json.dumps(request).encode()},
                      runner=runner)
        result['sandbox'] = asdict(sandbox)
        if sandbox.component != 'ok':
            result['component'] = 'harness_error' if sandbox.returncode == 88 else sandbox.component
            return result
        work = Path(sandbox.workdir)
        meta = work / 'worker-result.json'
        if meta.is_symlink() or meta.stat().st_size > 4 << 20:
            raise ValueError('Invalid observation metadata')
        record = json.loads(meta.read_text())
        result['runtime'] = record
        if not record['calibration_kernels']:
            raise ValueError('Missing runtime calibration')
        if record['output_contract_failure'] is not None:
            result.update(component='incorrect', reason=record['output_contract_failure'])
            return result
        observations = record['observations']
        if len(observations) != len(cases):
            raise ValueError('Missing per-input observation')
        assessments = [assess_observation(x, names) for x in observations]
        result['contract_checks'] = assessments
        if any(x['component'] == 'harness_error' for x in assessments):
            raise ValueError('Incomplete runtime observation')
        bad_init = sorted(set(record['init_dispatch_ops']) - ALLOWED_ATEN)
        result['forbidden_init_ops'] = bad_init
        checks = [compare(read_output(work / f'output-{i}.safetensors', wanted), wanted)
                  for i, wanted in enumerate(expected)]
        result['checks'] = checks
        result['flags'] = classify(checks, record['flags'], [])
        result['numerically_correct'] = all(x['correct'] for x in checks)
        result['contract_compliant'] = not bad_init and all(x['component'] == 'compliant' for x in assessments)
        result['correct'] = result['numerically_correct'] and result['contract_compliant'] and not result['flags']
        result['component'] = ('correct' if result['correct'] else 'contract_violation'
                               if not result['contract_compliant'] else 'incorrect')
        return result
    except Exception as error:
        result.update(component='harness_error', error=f'{type(error).__name__}: {error}')
        return result
    finally:
        (out / 'verdict.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        # Small developer cases retain observed outputs for inspection. Only the
        # staging directory is scratch; the sandbox has already consumed its files.
        shutil.rmtree(stage)
