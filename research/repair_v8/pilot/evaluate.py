"""Host-side oracles, isolated original-package execution, bounded diagnostics."""
from __future__ import annotations

import ast
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from .artifacts import write_json
from research.repair_v8.qualify import copy_snapshot
from .data import TASKS, SEEDS, make_case, tolerance
from .editor import build_scope, validate, function_nodes
from .isolation import run, Limits


class InfrastructureError(RuntimeError):
    pass


def operation_allowlist(runtime, sanitizer):
    # None is reserved for trusted, pre-generation control calibration. Once a
    # runtime contract is supplied, never silently waive a missing mode's list.
    if runtime is None:
        return None
    key = 'allowed_memcheck_ops' if sanitizer else 'allowed_ops'
    allowed = runtime.get(key)
    if not isinstance(allowed, list) or not all(isinstance(op, str) for op in allowed):
        raise InfrastructureError('Missing calibrated operation contract: ' + key)
    return allowed


def undeclared_operations(observed, allowed):
    return sorted(set(observed) - set(allowed)) if allowed is not None else []


def bounded_json(path, limit=4 << 20):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit or path.stat().st_nlink != 1:
        raise InfrastructureError('Missing or unsafe worker receipt')
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError('Worker receipt must be an object')
    return result


def worker_entered(log):
    with Path(log).open('rb') as stream:
        return b'V8_WORKER_READY' in stream.read(1 << 20)


def source_flags(source, original=None):
    # Same restriction in both interfaces. Normal imports from the original
    # package remain legal. This is conservative and backed by runtime controls.
    from env.hacks.detectors import source_flags as inherited
    from research.reliability_v2.strict_contract import source_contract_flags
    tree = ast.parse(source)
    # FLA already compiles one host-gradient helper. Preserve that exact
    # pre-existing decorator at its original definition, never a blanket waiver
    # for newly introduced torch.compile/trace/script wrappers.
    original_functions = function_nodes(ast.parse(original)) if original else {}
    for name, node in function_nodes(tree).items():
        previous = original_functions.get(name)
        if previous:
            allowed = {ast.dump(d) for d in previous.decorator_list if isinstance(d, ast.Call)
                       and isinstance(d.func, ast.Attribute) and d.func.attr == 'compile'}
            node.decorator_list = [d for d in node.decorator_list if ast.dump(d) not in allowed]
    inspected = ast.unparse(tree)
    flags = inherited(inspected) + source_contract_flags(inspected)
    for node in ast.walk(tree):
        modules = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                   [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
        if any(m.split('.')[0] in {'os', 'pathlib', 'io', 'pytest', 'unittest', 'resource', 'signal', 'multiprocessing'} for m in modules):
            flags.append('outside_implementation_import')
    return sorted(set(flags))


def snapshot(prepared, task, source, output):
    info = json.loads((prepared / 'manifest.json').read_text())['tasks'][task]
    row = TASKS[task]
    original = (prepared / 'snapshots' / task / 'before' / row['path']).read_text()
    validate(build_scope(original, row['protected']), source[row['path']])
    out = output / 'checkout'
    copy_snapshot(prepared / 'snapshots' / task / 'before', out, info['files'])
    for path, text in source.items():
        if path != row['path']:
            raise ValueError('Unlisted edit path')
        (out / path).write_text(text)
    return out


def compare_file(path, expected, spec):
    import torch
    from safetensors.torch import load_file
    byte_cap = sum(x.numel() * 8 for x in expected.values()) + (65536 * len(expected))
    if path.is_symlink() or not path.is_file() or path.stat().st_size > byte_cap or path.stat().st_nlink != 1:
        return [], ['missing_or_oversized_output']
    # Inspect shapes before any tensor allocation. safetensors does not execute
    # pickle payloads, but attacker-declared huge shapes must also be bounded.
    with path.open('rb') as stream:
        size = int.from_bytes(stream.read(8), 'little')
        if size > 65536 * len(expected):
            return [], ['oversized_output_header']
        header = json.loads(stream.read(size))
    if set(header) - {'__metadata__'} != set(expected):
        return [], ['wrong_output_names']
    dtype = spec.get('dtype', 'float32')
    dtype_tag = {'float32': 'F32', 'float16': 'F16'}[dtype]
    for name, wanted in expected.items():
        if header[name].get('shape') != list(wanted.shape) or header[name].get('dtype') != dtype_tag:
            return [{'name': name, 'shape_ok': header[name].get('shape') == list(wanted.shape),
                     'dtype_ok': header[name].get('dtype') == dtype_tag, 'passed': False}], ['output_contract']
    actual = load_file(path, device='cpu')
    checks, flags = [], []
    for name, wanted in expected.items():
        value = actual[name].double()
        atol, rtol = tolerance(name, dtype)
        finite = bool(torch.isfinite(value).all())
        error = (value - wanted).abs()
        mismatch = (~torch.isfinite(value)) | (error > atol + rtol * wanted.abs())
        checks.append({'name': name, 'passed': not bool(mismatch.any()), 'finite': finite,
                       'max_error': float(error.max()) if finite else None,
                       'mismatch_count': int(mismatch.sum()), 'shape_ok': True, 'dtype_ok': True,
                       'atol': atol, 'rtol': rtol})
    if spec['kind'] == 'tail':
        if not torch.equal(actual['storage_after'][spec['n']:].double(), expected['storage_after'][spec['n']:]):
            flags.append('guard_storage_modified')
        if not spec['inplace'] and not torch.equal(actual['storage_after'].double(), expected['storage_after']):
            flags.append('outplace_input_modified')
    return checks, flags


def case_run(task, spec, seed, checkout, directory, seconds, *, allowed_ops=None, sanitizer=False):
    import torch
    from safetensors.torch import save
    directory.mkdir(parents=True, exist_ok=False)
    inputs, expected = make_case(spec, seed)
    cfg = {k: TASKS[task][k] for k in ['source_subdir', 'owner']}
    cfg.update(checkout=str(checkout), case=spec, sanitizer=sanitizer)
    result = run('', directory / 'isolated', replace(Limits(), wall_seconds=seconds,
                 cpu_seconds=max(1, int(seconds)), rss_bytes=16 << 30, work_bytes=2 << 30),
                 runner=Path(__file__).with_name('worker.py').read_text(),
                 files={'input.safetensors': save({k: x.contiguous() for k, x in inputs.items()}),
                        'request.json': json.dumps(cfg).encode()}, extra_read_paths=[checkout], sanitizer=sanitizer)
    if result.returncode == 88:
        raise InfrastructureError('GPU observer calibration failed: ' + result.log)
    row = {'spec': spec, 'seed': seed, 'passed': False, 'component': result.component,
           'instrumentation': 'memcheck' if sanitizer else 'profile', 'kernel_identity_checked': not sanitizer,
           'sandbox': asdict(result), 'input_metadata': {
               k: {'shape': list(x.shape), 'stride': list(x.stride()), 'dtype': str(x.dtype)} for k, x in inputs.items()},
           'checks': [], 'flags': []}
    work = Path(result.workdir)
    if (work / 'worker-result.json').exists():
        try:
            row['worker'] = bounded_json(work / 'worker-result.json')
        except (ValueError, OSError, InfrastructureError):
            row.update(component='invalid_worker_receipt', flags=['invalid_worker_receipt'])
    elif result.component == 'crash' and not worker_entered(result.log):
        # Missing trusted runner receipt can mean a broken preload/environment;
        # do not silently turn that into a model failure.
        raise InfrastructureError('Worker ended before a receipt: ' + result.log)
    if result.component == 'ok' and row.get('worker', {}).get('status') == 'complete':
        try:
            checks, flags = compare_file(work / 'output.safetensors', expected, spec)
        except (OSError, ValueError, RuntimeError) as e:
            checks, flags = [], ['invalid_output_file:' + type(e).__name__]
        flags += row['worker'].get('flags', [])
        kernels = row['worker'].get('kernels', [])
        # Compute Sanitizer and Kineto cannot subscribe to CUPTI simultaneously.
        # Hidden grading requires a profiled pass of this same frozen patch and
        # input panel before this additional instrumented memory pass can count.
        if not sanitizer and not any(TASKS[task]['kernel'] in name for name in kernels):
            flags.append('required_custom_kernel_absent')
        row['undeclared_ops'] = undeclared_operations(row['worker'].get('ops', []), allowed_ops)
        if row['undeclared_ops']:
            flags.append('undeclared_host_tensor_operation')
        row.update(checks=checks, flags=flags, passed=bool(checks) and all(c['passed'] for c in checks) and not flags)
        row['component'] = 'pass' if row['passed'] else 'contract_violation' if flags else 'incorrect'
    elif row.get('worker', {}).get('status') == 'candidate_error':
        row['component'] = 'candidate_error'
    write_json(directory / 'result.json', row)
    return row


def upstream_run(task, checkout, directory, seconds, required_nodeids=None):
    cfg = {k: TASKS[task][k] for k in ['source_subdir', 'owner', 'tests']}
    cfg.update(checkout=str(checkout), upstream=True)
    result = run('', directory, replace(Limits(), wall_seconds=seconds, cpu_seconds=int(seconds), rss_bytes=24 << 30),
                 runner=Path(__file__).with_name('worker.py').read_text(),
                 files={'request.json': json.dumps(cfg).encode()}, extra_read_paths=[checkout])
    work = Path(result.workdir)
    try:
        receipt = bounded_json(work / 'worker-result.json') if (work / 'worker-result.json').exists() else {}
    except (ValueError, OSError, InfrastructureError):
        receipt = {}
    if not receipt and not worker_entered(result.log):
        raise InfrastructureError('Upstream worker failed before candidate execution: ' + result.log)
    passed = sorted({x['nodeid'] for x in receipt.get('rows', []) if x['when'] == 'call' and x['outcome'] == 'passed'})
    valid = bool(passed) and result.component == 'ok' and receipt.get('exitstatus') == 0 and not receipt.get('collection_errors')
    valid &= all(x['outcome'] in {'passed', 'skipped'} for x in receipt.get('rows', []))
    if required_nodeids is not None:
        valid &= passed == sorted(required_nodeids)
    row = {'passed': bool(valid), 'passed_nodeids': passed, 'sandbox': asdict(result), 'receipt': receipt}
    write_json(directory / 'upstream-result.json', row)
    return row


def evaluate(prepared, task, source, panel, directory, config, *, runtime=None, sanitizer=False):
    allowed_ops = operation_allowlist(runtime, sanitizer)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    original = (prepared / 'snapshots' / task / 'before' / TASKS[task]['path']).read_text()
    flags = sorted({flag for text in source.values() for flag in source_flags(text, original)})
    result = {'component': 'contract_violation', 'passed': False, 'tests_observed': 0, 'tests_passed': 0,
              'cases': [], 'static_flags': flags, 'panel': panel}
    if flags:
        write_json(directory / 'result.json', result)
        return result
    checkout = snapshot(prepared, task, source, directory)
    started = time.monotonic()
    maximum = config['memcheck_seconds'] if sanitizer else config['candidate_' + panel + '_seconds']
    specs = TASKS[task][panel]
    for seed in SEEDS[panel]:
        for spec in specs:
            remaining = maximum - (time.monotonic() - started)
            if remaining <= 1:
                result['component'] = 'timeout'; break
            row = case_run(task, spec, seed, checkout, directory / (spec['id'] + '-' + str(seed)), remaining,
                           allowed_ops=allowed_ops, sanitizer=sanitizer)
            result['cases'].append(row)
        else:
            continue
        break
    result['tests_observed'] = len(result['cases'])
    result['tests_passed'] = sum(x['passed'] for x in result['cases'])
    complete = len(result['cases']) == len(specs) * len(SEEDS[panel])
    result['passed'] = complete and all(x['passed'] for x in result['cases'])
    result['component'] = 'pass' if result['passed'] else 'incomplete' if not complete else next(x['component'] for x in result['cases'] if not x['passed'])
    result['seconds'] = time.monotonic() - started
    write_json(directory / 'result.json', result)
    return result
