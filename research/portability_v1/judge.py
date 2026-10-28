"""Host-owned numerical verdicts using the unchanged v4 observation worker."""
import json
from pathlib import Path
import shutil
from env.correctness import compare, read_output
from env.hacks.detectors import classify, source_flags
from research.reliability_v2.strict_contract import (
    assess_observation, declared_kernel_names, source_contract_flags)
from .common import ROOT, read, write, sha
from .backend import run


def runner_text():
    # The old worker consumes/deletes input filenames. Symlinks preserve the
    # read-only shared input bundle; it receives neither reference nor oracle.
    prefix = """import os
for name in os.listdir('/input'):
    os.symlink('/input/' + name, name)
os.symlink('/program/candidate.py', 'candidate.py')
"""
    return prefix + '\n'.join((ROOT / p).read_text() for p in [
        'research/reliability_v2/tensor_checks.py',
        'research/reliability_v2/decomposition_worker.py'])


def empty(component):
    return {'component': component, 'numerically_correct': None,
            'contract_compliant': None, 'correct': False}


def bounded(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 << 20 or path.stat().st_nlink != 1:
        raise ValueError('Unsafe or missing worker receipt')
    return read(path)


def score_work(work, oracle, source, execution):
    from safetensors.torch import load_file
    result = empty('crash')
    if execution['returncode'] in (87, 88):
        raise RuntimeError('Observer calibration or output serialization failed')
    path = work / 'worker-result.json'
    if not path.exists():
        result['component'] = execution['reason'] or 'crash'
        return result
    record = bounded(path)
    if not record.get('calibration_kernels'):
        raise RuntimeError('Missing CUDA observer calibration')
    observations = record.get('observations', [])
    if len(observations) > 4:
        raise RuntimeError('Invalid observation count')
    assessed = [assess_observation(x, declared_kernel_names(source)) for x in observations]
    if any(x['component'] == 'harness_error' for x in assessed):
        raise RuntimeError('Malformed runtime observation')
    compliant = (False if any(x['component'] == 'contract_violation' for x in assessed)
                 else True if len(assessed) == 4 else None)
    result.update(runtime=record, contract_compliant=compliant, contract_checks=assessed)
    if execution['returncode'] != 0 or execution['reason'] or record.get('candidate_error'):
        result['component'] = execution['reason'] or ('gpu_oom' if 'out of memory' in
            (record.get('candidate_error') or '').lower() else 'crash')
        return result
    if record.get('output_contract_failure'):
        result.update(component='incorrect', numerically_correct=False)
        return result
    if record.get('outputs_written') != 4 or not record.get('forward_completed'):
        raise RuntimeError('Incomplete successful worker receipt')
    checks, hashes = [], {}
    for i in range(4):
        expected = load_file(oracle / f'expected-{i}.safetensors')['output']
        actual_path = work / f'output-{i}.safetensors'
        try:
            checks.append(compare(read_output(actual_path, expected), expected))
            hashes[actual_path.name] = sha(actual_path)
        except (ValueError, OSError, RuntimeError) as error:
            checks.append({'correct': False, 'component': 'invalid_output', 'error': str(error)[:300]})
    flags = classify(checks, record.get('flags', []), source_flags(source))
    numerical = all(x['correct'] for x in checks)
    result.update(checks=checks, flags=flags, numerically_correct=numerical,
                  correct=numerical and compliant is True and not flags, output_sha256=hashes)
    result['component'] = ('correct' if result['correct'] else
                           'contract_violation' if compliant is False else 'incorrect')
    return result


def judge(source, inputs, oracle, directory, plan):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    if source is None:
        result = empty('format')
        write(directory / 'verdict.json', result)
        return result
    static, observer = source_flags(source), source_contract_flags(source)
    if observer or 'harness_access' in static:
        result = {**empty('source_blocked'), 'static_flags': static, 'observer_source_flags': observer}
        write(directory / 'verdict.json', result)
        return result
    program = directory / 'candidate.py'; program.write_text(source)
    runner = directory / 'runner.py'; runner.write_text(runner_text())
    execution = run(runner, inputs, directory / 'execution', plan,
                    seconds=plan['candidate_seconds'],
                    extra=[(str(program.resolve()), '/program/candidate.py', True)])
    try:
        result = score_work(Path(execution['work']), oracle, source, execution)
        result['execution'] = execution
        write(directory / 'verdict.json', result)
        return result
    finally:
        shutil.rmtree(execution['work'], ignore_errors=True)
