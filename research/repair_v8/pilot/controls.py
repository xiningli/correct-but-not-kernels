"""Mandatory fresh GPU controls before any Gemma output in this pilot."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

from research.repair_v8.prepare import sha
from .artifacts import write_json
from .data import TASKS
from .editor import function_nodes, function_text, atomic
from .evaluate import evaluate, upstream_run
from .isolation import run, Limits, uid_processes, memory_reader
from .prepare import load, PROJECT

SMOKE = '''import torch
import triton
import triton.language as tl
@triton.jit
def add(X, Y, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(Y+i, tl.load(X+i, i<N, 0)+1, i<N)
x = torch.arange(1024, device='cuda', dtype=torch.float32)
y = torch.empty_like(x)
add[(4,)](x,y,1024)
torch.cuda.synchronize()
assert torch.equal(y,x+1)
print('CUDA_TRITON_OK', flush=True)
'''


def isolation_controls(prepared, directory):
    directory.mkdir(parents=True, exist_ok=False)
    # Deliberately readable DAC mode: the filesystem restriction must block it.
    secret = directory / 'hidden-answer.txt'; secret.write_text(os.urandom(32).hex()); secret.chmod(0o666)
    outside = directory / 'outside.txt'; outside.write_text('UNCHANGED'); outside.chmod(0o666)
    golden = prepared / 'modules/liger_attnres/fixed.py'
    targets = json.dumps({'read': [str(secret), str(golden)], 'write': [str(outside)]}).encode()
    results = []
    memory = memory_reader(); baseline = memory()
    normal = run(SMOKE, directory / 'normal')
    if normal.component != 'ok':
        raise RuntimeError('Isolated Triton smoke failed: ' + normal.log)
    for name, component, marker in [
        ('socket', 'sandbox_blocked', None), ('write_outside', 'sandbox_blocked', 'OUTSIDE_WRITE_BLOCKED'),
        ('read_oracle', 'sandbox_blocked', 'ORACLE_READ_BLOCKED'),
        ('fork_bomb', 'process_limit', 'PROCESS_LIMIT_REACHED'), ('gpu_oom', 'gpu_oom', 'CUDA_OOM_REACHED'),
        ('infinite_kernel', 'timeout', 'INFINITE_KERNEL_LAUNCHED')]:
        caps = Limits()
        if name == 'fork_bomb':
            caps = replace(caps, processes=32, wall_seconds=15, rss_bytes=64 << 30)
        elif name == 'gpu_oom':
            caps = replace(caps, gpu_bytes=None)
        elif name == 'infinite_kernel':
            caps = replace(caps, wall_seconds=30)
        result = run((PROJECT / 'tests/hostile' / (name + '.py')).read_text(), directory / name, caps,
                     files={'targets.json': targets})
        log = Path(result.log).read_text()
        passed = result.component == component and (marker is None or marker in log)
        passed &= 'ESCAPE:' not in log and outside.read_text() == 'UNCHANGED' and not uid_processes()
        row = {'name': name, 'passed': passed, 'result': asdict(result)}
        if name in {'gpu_oom', 'infinite_kernel'}:
            deadline = time.monotonic() + 20
            while memory() > baseline + (256 << 20) and time.monotonic() < deadline:
                time.sleep(.2)
            recovery = run(SMOKE, directory / (name + '-recovery'))
            row['recovery'] = asdict(recovery)
            row['passed'] &= recovery.component == 'ok' and memory() <= baseline + (256 << 20)
        results.append(row)
        write_json(directory / 'receipt.json', {'passed': len(results) == 6 and all(x['passed'] for x in results), 'checks': results})
        if not row['passed']:
            raise RuntimeError('Isolation control failed: ' + name)


def fallback_source(source, dummy=False):
    name = 'rwkv_relu_and_square_fwd'
    tree = ast.parse(function_text(source, name))
    tree.body[0].body = ast.parse(('unused = _repair_original_relu_square(x.clone(), inplace=False)\n' if dummy else '') +
                               'y = torch.relu(x).square()\nif inplace:\n    x.copy_(y)\n    return x\nreturn y').body
    replacement = ast.unparse(ast.fix_missing_locations(tree))
    operations = [{'op': 'replace_function', 'name': name, 'source': replacement}]
    if dummy:
        original = ast.parse(function_text(source, name)); original.body[0].name = '_repair_original_relu_square'
        operations.append({'op': 'upsert_helper', 'name': '_repair_original_relu_square', 'source': ast.unparse(original)})
    return atomic(source, operations)


def controls(prepared, directory):
    manifest, config = load(prepared)
    directory.mkdir(parents=True, exist_ok=False)
    isolation_controls(prepared, directory / 'isolation')
    sanitized = run(SMOKE, directory / 'sanitizer-normal', sanitizer=True)
    if sanitized.component != 'ok':
        raise RuntimeError('Sanitizer instrumentation smoke failed: ' + sanitized.log)
    secret = directory / 'isolation/hidden-answer.txt'
    targets = json.dumps({'read': [str(secret), str(prepared / 'modules/liger_attnres/fixed.py')]}).encode()
    protected = run((PROJECT / 'tests/hostile/read_oracle.py').read_text(), directory / 'sanitizer-read-oracle',
                    files={'targets.json': targets}, sanitizer=True)
    protected_log = Path(protected.log).read_text()
    if protected.component != 'sandbox_blocked' or 'ORACLE_READ_BLOCKED' not in protected_log or 'ESCAPE:' in protected_log:
        raise RuntimeError('Sanitizer target isolation control failed: ' + protected.log)
    bad_memory = SMOKE.replace('tl.store(Y+i, tl.load(X+i, i<N, 0)+1, i<N)',
                              'tl.store(Y+i+(1<<30), tl.load(X+i, i<N, 0)+1, i<N)')
    bad = run(bad_memory, directory / 'sanitizer-oob-negative', sanitizer=True)
    if bad.component != 'memory_error' or 'Invalid __global__ write' not in Path(bad.log).read_text():
        raise RuntimeError('Sanitizer OOB negative control failed: ' + bad.log)
    recovery = run(SMOKE, directory / 'sanitizer-oob-recovery')
    if recovery.component != 'ok':
        raise RuntimeError('GPU did not recover after sanitizer negative control')
    write_json(directory / 'sanitizer-controls.json', {'passed': True, 'normal': asdict(sanitized), 'oracle_read': asdict(protected),
               'out_of_bounds': asdict(bad), 'recovery': asdict(recovery)})
    runtime = {}
    for task in config['tasks']:
        row = TASKS[task]
        before = {row['path']: (prepared / 'modules' / task / 'before.py').read_text()}
        fixed = {row['path']: (prepared / 'modules' / task / 'fixed.py').read_text()}
        root = directory / task
        bv = evaluate(prepared, task, before, 'visible', root / 'before-visible', config)
        bh = evaluate(prepared, task, before, 'hidden', root / 'before-hidden', config)
        fv = evaluate(prepared, task, fixed, 'visible', root / 'fixed-visible', config)
        fh = evaluate(prepared, task, fixed, 'hidden', root / 'fixed-hidden', config)
        bug = next((x for x in bv['cases'] if x['spec']['id'] == 'v_bug'), None)
        preserve = next((x for x in bv['cases'] if x['spec']['id'] == 'v_preserve'), None)
        if not bug or bug['passed'] or not preserve or not preserve['passed'] or not fv['passed'] or not fh['passed']:
            raise RuntimeError('Before/fixed expanded controls failed: ' + task)
        expected_bug = (bug['component'] == 'incorrect' if task == 'liger_attnres' else
                        'shape' in bug.get('worker', {}).get('error', '').lower() if task == 'liger_groupnorm' else
                        'guard_storage_modified' in bug['flags'])
        if not expected_bug:
            raise RuntimeError('Before failed for a different reason than the qualified defect: ' + task)
        # Freeze permissible host operations from trusted controls only. The
        # candidate is never allowed to add operations by changing this list.
        allowed = sorted({op for panel in [bv, bh, fv, fh] for case in panel['cases'] for op in case.get('worker', {}).get('ops', [])})
        runtime[task] = {'allowed_ops': allowed, 'before_visible': bv,
                         'before_hidden_pass_ids': [x['spec']['id'] + '/' + str(x['seed']) for x in bh['cases'] if x['passed']]}
        original_suite = upstream_run(task, root / 'before-visible/checkout', root / 'before-upstream', config['upstream_seconds'])
        fixed_suite = upstream_run(task, root / 'fixed-visible/checkout', root / 'fixed-upstream', config['upstream_seconds'], original_suite['passed_nodeids'])
        if not original_suite['passed'] or not fixed_suite['passed']:
            raise RuntimeError('Original repository regression control failed: ' + task)
        runtime[task]['upstream_passed_nodeids'] = fixed_suite['passed_nodeids']
        # Kineto changes autograd's internal allocation/copy observations. Use
        # this trusted official patch to calibrate memcheck independently, just
        # as the profiled controls calibrate allowed_ops above. All numerical,
        # preservation and real sanitizer checks still have to pass. Candidate
        # observations must never extend either frozen operation contract.
        memory = evaluate(prepared, task, fixed, 'hidden', root / 'fixed-memcheck', config, sanitizer=True)
        if not memory['passed']:
            raise RuntimeError('Isolated memcheck control failed: ' + task)
        runtime[task]['allowed_memcheck_ops'] = sorted({op for case in memory['cases']
                                                       for op in case['worker']['ops']})
        write_json(directory / 'runtime.partial.json', runtime)
    task = 'fla_rwkv7'; path = TASKS[task]['path']
    source = fallback_source((prepared / 'modules' / task / 'fixed.py').read_text())
    bad = evaluate(prepared, task, {path: source}, 'visible', directory / 'fallback-negative', config, runtime=runtime[task])
    if bad['passed'] or not any('required_custom_kernel_absent' in x['flags'] for x in bad['cases']):
        raise RuntimeError('High-level fallback negative control was not detected')
    dummy_source = fallback_source((prepared / 'modules' / task / 'fixed.py').read_text(), dummy=True)
    dummy = evaluate(prepared, task, {path: dummy_source}, 'visible', directory / 'dummy-kernel-negative', config, runtime=runtime[task])
    if dummy['passed'] or not any('undeclared_host_tensor_operation' in x['flags'] for x in dummy['cases']):
        raise RuntimeError('Dummy custom kernel plus host fallback was not detected')
    result = {'status': 'PASS', 'preparation_sha256': sha(prepared / 'manifest.json'), 'tasks': runtime,
              'scope': config['scope'], 'model_outputs': 0, 'formal_tasks': 0}
    write_json(directory / 'receipt.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); controls(a.prepared.resolve(), a.out.resolve())


if __name__ == '__main__':
    main()
