"""CPU preparation and single-GPU controls for four development candidates.

Does not evaluate models, train, alter historical results, or claim formal task
eligibility. Only reviewed upstream revisions run here; model-generated code
needs its own security boundary before this harness may be reused for it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .cases import TASKS
from .prepare import sha, write_json

PROJECT = Path(__file__).resolve().parents[2]
SEEDS = [1701, 1702, 1703]


def verify_files(root, files):
    for row in files:
        path = root / row['path']
        if 'symlink' in row:
            if not path.is_symlink() or str(path.readlink()) != row['symlink']:
                raise ValueError('Snapshot symlink drift: ' + str(path))
        elif not path.is_file() or sha(path) != row['sha256']:
            raise ValueError('Snapshot source drift: ' + str(path))


def tree_manifest(root):
    rows = []
    for p in sorted(root.rglob('*')):
        relative = str(p.relative_to(root))
        if '__pycache__' in p.parts or '.pytest_cache' in p.parts:
            continue
        if p.is_symlink():
            rows.append({'path': relative, 'symlink': str(p.readlink())})
        elif p.is_file():
            rows.append({'path': relative, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    return rows


def copy_snapshot(source, target, files):
    """Copy only receipted source files, excluding generated discovery output."""
    target.mkdir(parents=True, exist_ok=True)
    for row in files:
        dest = target / row['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if 'symlink' not in row:
            shutil.copy2(source / row['path'], dest)
    for row in files:
        if 'symlink' in row:
            (target / row['path']).symlink_to(row['symlink'])


def plan(root):
    executions = []
    for key, task in TASKS.items():
        before_source = root / 'sources' / task['slug'] / 'before'
        before_receipt = json.loads((root / 'source-receipts' / (task['slug'] + '-before.json')).read_text())
        before_test_files = [r for r in before_receipt['files'] if r['path'].startswith(task['test_tree'] + '/')]
        for revision in ['before', 'fixed']:
            source = root / 'sources' / task['slug'] / revision
            receipt = json.loads((root / 'source-receipts' / (task['slug'] + '-' + revision + '.json')).read_text())
            verify_files(source, receipt['files'])
            target = root / 'execution' / task['slug'] / revision
            manifest_path = root / 'execution-manifests' / (task['slug'] + '-' + revision + '.json')
            if target.exists():
                old = json.loads(manifest_path.read_text())
                verify_files(target, old['files'])
            else:
                copy_snapshot(source, target, receipt['files'])
                # Use the exact same pre-fix preservation fixtures in both arms.
                # Never touch source snapshots; new tests are in the external probes.
                shutil.rmtree(target / task['test_tree'])
                copy_snapshot(before_source, target, before_test_files)
                old = {'task': key, 'revision': revision, 'commit': receipt['commit'],
                       'source_snapshot': str(source), 'test_fixture_origin': 'before/' + task['test_tree'],
                       'files': tree_manifest(target)}
                write_json(manifest_path, old)
            executions.append({'task': key, 'revision': revision, 'checkout': str(target),
                               'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path)})
    code = [PROJECT / 'research/repair_v8' / x for x in ['__init__.py', 'prepare.py', 'cases.py', 'worker.py', 'qualify.py', 'pytest_receipt.py']]
    code.append(PROJECT / 'experiments/repair-v8-qualification.lock')
    result = {'schema': 'v8-development-qualification-batch1-v1',
              'status': 'prepared_for_GPU_controls_not_qualified',
              'tasks': TASKS, 'seeds': SEEDS, 'executions': executions,
              'files': [{'path': str(p.relative_to(PROJECT)), 'sha256': sha(p)} for p in code],
              'timeout_seconds': {'probe': 600, 'upstream_suite': 1800, 'memcheck': 1200},
              'environment': {'python': '3.12', 'torch': '2.8.0', 'triton': '3.4.0', 'cuda_runtime': '12.8'},
              'gpu_count': 1, 'training_enabled': False, 'model_evaluation_enabled': False,
              'skip_policy': 'record individually; zero passed tests or any failure/error is not a suite pass',
              'memcheck_policy': 'required for qualification; missing tool is a pending gate, never a pass',
              'qualification_limit': 'Controls validate this development suite only; formal hidden tests, edit-interface parity and generated-code security still not sealed.'}
    path = root / 'qualification-plan.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Plan drift: use a new preparation root or explicitly preserve/amend the old draft before GPU execution')
    write_json(path, result)
    return result


def environment(checkout, task, output):
    env = os.environ.copy()
    env.update({'PYTHONPATH': os.pathsep.join([str(checkout / task['source_subdir']), str(checkout), str(PROJECT)]),
                'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'REPAIR_V8_PYTEST_RECEIPT': str(output),
                'PYTORCH_NO_CUDA_MEMORY_CACHING': '1'})
    return env


def execute(command, cwd, env, log, timeout):
    start = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        timed_out = False
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
    return {'command': command, 'returncode': code, 'timeout': timed_out,
            'seconds': time.monotonic() - start, 'log': str(log), 'log_sha256': sha(log)}


def worker_command(key, case, checkout, seed, output, oracle=False, profile=False):
    command = [sys.executable, '-m', 'research.repair_v8.worker', '--task', key, '--case', case,
               '--checkout', str(checkout), '--seed', str(seed), '--out', str(output)]
    if oracle:
        command.append('--oracle-only')
    if profile:
        command.append('--profile')
    return command


def cpu_check(root, prepared):
    import torch
    output = root / 'cpu-check'
    rows = []
    # No GPU access, including accidental default CUDA initialization in imports.
    for key, task in TASKS.items():
        checkout = root / 'execution' / task['slug'] / 'fixed'
        for case in task['cases']:
            path = output / (key + '-' + case + '.json')
            env = environment(checkout, task, path); env['CUDA_VISIBLE_DEVICES'] = ''
            process = execute(worker_command(key, case, checkout, SEEDS[0], path, oracle=True),
                              checkout, env, path.with_suffix('.log'), 120)
            row = json.loads(path.read_text()) if path.exists() else {'status': 'missing_receipt'}
            rows.append({'task': key, 'case': case, 'status': row['status'], 'process': process})
    result = {'status': 'pass' if all(x['status'] == 'pass' and x['process']['returncode'] == 0 for x in rows) else 'fail',
              'scope': 'CPU_reference_arithmetic_and_subprocess_paths_only_not_GPU_import_or_kernel_success',
              'python': sys.version, 'torch': torch.__version__, 'runtime_cuda': torch.version.cuda,
              'packages': {p: importlib.metadata.version(p) for p in ['torch', 'triton', 'transformers', 'pytest']},
              'GPU_runs': 0, 'cases': rows}
    write_json(output / 'summary.json', result)
    print(json.dumps({'CPU_reference_cases': len(rows), 'status': result['status'], 'GPU_runs': 0}))
    return result


def bug_reproduced(task, row):
    if row.get('oracle_only') or row.get('stage') != 'probe':
        return False
    error = row.get('error', '').lower()
    if row.get('status') == 'test_failure':
        return ('numerical mismatch' in error if task in ['liger_attnres', 'flaggems_addmm']
                else 'out-of-bounds store changed guard' in error if task == 'fla_rwkv7' else False)
    if task == 'liger_groupnorm' and row.get('status') == 'probe_exception':
        return 'shape' in error and ('invalid' in error or 'size' in error) or 'illegal memory access' in error
    return False


def suite_pass(receipt):
    return (receipt.get('exitstatus') == 0 and any(x['outcome'] == 'passed' for x in receipt.get('rows', []))
            and all(x['outcome'] in ['passed', 'skipped'] for x in receipt.get('rows', [])))


def gpu_probe_pass(value):
    receipt = value.get('receipt', {})
    return (receipt.get('status') == 'pass' and receipt.get('oracle_only') is False
            and receipt.get('result', {}).get('mode') == 'GPU_original_repository'
            and value.get('process', {}).get('returncode') == 0
            and not value.get('process', {}).get('timeout', True))


def summarize_task(key, rows, sanitizer_available):
    task = TASKS[key]
    before_ok = fixed_ok = preserved_ok = traces_ok = memory_ok = True
    for seed in SEEDS:
        lookup = {(x['revision'], x['kind'], x['case']): x for x in rows if x['seed'] == seed}
        bug = lookup.get(('before', 'probe', task['bug_case']), {})
        before_ok &= bug_reproduced(key, bug.get('receipt', {})) and not bug.get('process', {}).get('timeout', True)
        for revision in ['before', 'fixed']:
            suite = lookup.get((revision, 'upstream', 'suite'), {})
            preserved_ok &= suite_pass(suite.get('receipt', {})) and suite.get('process', {}).get('returncode') == 0
        for case in task['cases']:
            value = lookup.get(('fixed', 'probe', case), {})
            fixed_ok &= gpu_probe_pass(value)
        for case in task['preserve_cases']:
            value = lookup.get(('before', 'probe', case), {})
            preserved_ok &= gpu_probe_pass(value)
        trace = lookup.get(('fixed', 'profile', task['profile_case']), {})
        traces_ok &= gpu_probe_pass(trace) and any(task['kernel_name'] in name for name in
                                                 trace.get('receipt', {}).get('cuda_kernel_names', []))
        mem = lookup.get(('fixed', 'memcheck', task['memcheck_case']), {})
        memory_ok &= gpu_probe_pass(mem)
    gates = {'before_same_bug_three_processes': bool(before_ok), 'fixed_all_probes_three_processes': bool(fixed_ok),
             'preservation_both_revisions_three_processes': bool(preserved_ok), 'custom_kernel_trace': bool(traces_ok),
             'memcheck': bool(memory_ok and sanitizer_available)}
    return {'task': key, 'gates': gates, 'status': 'development_controls_pass' if all(gates.values()) else 'not_qualified',
            'formal_eligible': False}


def gpu_run(root, prepared):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device; nothing launched')
    if sys.version_info[:2] != (3, 12) or torch.__version__.split('+')[0] != '2.8.0' or torch.version.cuda != '12.8':
        raise RuntimeError('Wrong qualification environment; use the separately locked venv')
    if importlib.metadata.version('triton') != '3.4.0':
        raise RuntimeError('Triton version mismatch')
    verify_files(PROJECT, prepared['files'])
    sanitizer = shutil.which('compute-sanitizer')
    run = root / 'gpu-controls' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'GPU': torch.cuda.get_device_name(0),
               'capability': torch.cuda.get_device_capability(0), 'compute_sanitizer': sanitizer,
               'plan_sha256': sha(root / 'qualification-plan.json'),
               'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))})
    rows = []; summaries = []
    for key, task in TASKS.items():
        for seed in SEEDS:
            for revision in ['before', 'fixed']:
                checkout = root / 'execution' / task['slug'] / revision
                manifest = json.loads((root / 'execution-manifests' / (task['slug'] + '-' + revision + '.json')).read_text())
                verify_files(checkout, manifest['files'])
                work = run / key / str(seed) / revision
                commands = [('upstream', 'suite')] + [('probe', case) for case in task['cases']]
                if revision == 'fixed':
                    commands += [('profile', task['profile_case'])]
                    if sanitizer:
                        commands += [('memcheck', task['memcheck_case'])]
                for kind, case in commands:
                    path = work / (kind + '-' + case + '.json')
                    env = environment(checkout, task, path)
                    # One device and one evaluator process at a time.
                    env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
                    if kind == 'upstream':
                        command = [sys.executable, '-m', 'pytest', '-p', 'research.repair_v8.pytest_receipt',
                                   '-q', *task['upstream_tests']]
                        timeout = prepared['timeout_seconds']['upstream_suite']
                    else:
                        command = worker_command(key, case, checkout, seed, path, profile=kind == 'profile')
                        timeout = prepared['timeout_seconds']['probe']
                        if kind == 'memcheck':
                            command = [sanitizer, '--tool', 'memcheck', '--error-exitcode', '86',
                                       '--log-file', str(path.with_suffix('.memcheck.log')), *command]
                            timeout = prepared['timeout_seconds']['memcheck']
                    process = execute(command, checkout, env, path.with_suffix('.log'), timeout)
                    receipt = json.loads(path.read_text()) if path.exists() else {'status': 'missing_receipt'}
                    row = {'task': key, 'seed': seed, 'revision': revision, 'kind': kind,
                           'case': case, 'process': process, 'receipt': receipt}
                    rows.append(row); write_json(run / 'results.json', rows)
                    print(json.dumps({k: row[k] for k in ['task', 'seed', 'revision', 'kind', 'case']} |
                                     {'returncode': process['returncode']}), flush=True)
                verify_files(checkout, manifest['files'])
        summaries.append(summarize_task(key, [x for x in rows if x['task'] == key], bool(sanitizer)))
        write_json(run / 'summary.json', summaries)
    verify_files(PROJECT, prepared['files'])
    print(json.dumps({'run': str(run), 'summary': summaries}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['plan', 'cpu-check', 'gpu-run'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    if args.phase == 'plan':
        result = plan(root)
        print(json.dumps({'tasks': len(result['tasks']), 'execution_snapshots': len(result['executions'])}))
    else:
        prepared = json.loads((root / 'qualification-plan.json').read_text())
        verify_files(PROJECT, prepared['files'])
        if args.phase == 'cpu-check':
            result = cpu_check(root, prepared)
            raise SystemExit(0 if result['status'] == 'pass' else 1)
        gpu_run(root, prepared)


if __name__ == '__main__':
    main()
