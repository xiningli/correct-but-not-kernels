"""K3/T1: six inherited C1 probes for the profiler-preloaded strict launcher; no old gate is overwritten."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from research.reliability_v2.strict_sandbox import Limits, run, uid_processes
from env.reward import failure_reward
from env.tasks import verified_catalog, validate_split
from tools.gatelib import ROOT, enforce_failure_stop, finish_gate, require_gate, timestamp
from tools.sandbox_smoke import SOURCE

def main():
    c0 = require_gate('C0')
    import pynvml
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    if pynvml.nvmlDeviceGetComputeRunningProcesses(gpu):
        raise RuntimeError('C1 OOM test requires an otherwise idle, exclusive GPU')
    baseline = pynvml.nvmlDeviceGetMemoryInfo(gpu).used
    directory = ROOT / 'runs/reliability-v2/strict-isolation' / timestamp()
    directory.mkdir(parents=True, mode=0o700)
    checks = []
    try:
        normal = run(SOURCE, directory / 'normal-before')
        if normal.component != 'ok':
            raise RuntimeError(f'Normal CUDA/Triton failed: {normal}; {Path(normal.log).read_text()}')
        tasks = verified_catalog()
        train, _ = validate_split(tasks)
        reference = next(t.path for t in tasks if t.task_id in train)
        harness = ROOT / 'env/timing.py'
        harness.chmod(0o600)
        oracle = directory / 'private-oracle.txt'
        oracle.write_text(os.urandom(32).hex())
        oracle.chmod(0o600)
        outside = directory / 'outside.txt'
        outside.write_text('UNCHANGED')
        outside.chmod(0o666)  # prove Landlock, not merely DAC, prevents write
        targets = json.dumps({'write': [str(outside), '/tmp/cuda-gemma4-c1-outside'],
                              'read': [str(reference), str(harness), str(oracle)]}).encode()
        probes = [('socket', 'sandbox_blocked', None),
                  ('write_outside', 'sandbox_blocked', 'OUTSIDE_WRITE_BLOCKED'),
                  ('fork_bomb', 'process_limit', 'PROCESS_LIMIT_REACHED'),
                  ('gpu_oom', 'gpu_oom', 'CUDA_OOM_REACHED'),
                  ('infinite_kernel', 'timeout', 'INFINITE_KERNEL_LAUNCHED'),
                  ('read_oracle', 'sandbox_blocked', 'ORACLE_READ_BLOCKED')]
        for name, expected, marker in probes:
            caps = Limits()
            if name == 'gpu_oom':
                caps = replace(caps, gpu_bytes=None)
            elif name == 'infinite_kernel':
                caps = replace(caps, wall_seconds=30)
            elif name == 'fork_bomb':
                # Summed RSS double-counts the shared, preloaded Torch pages
                # after fork; the kernel process cap is the target of this test.
                caps = replace(caps, processes=32, wall_seconds=15, rss_bytes=64 << 30)
            result = run((ROOT / f'tests/hostile/{name}.py').read_text(), directory/name,
                         caps, files={'targets.json': targets})
            log = Path(result.log).read_text()
            if 'ESCAPE:' in log or outside.read_text() != 'UNCHANGED':
                checks.append({'name': name, 'status': 'FAIL', 'result': asdict(result),
                               'error': 'Possible sandbox escape: STOP for operator'})
                break
            passed = result.component == expected and (marker is None or marker in log)
            check = {'name': name, 'status': 'PASS' if passed else 'FAIL',
                     'expected_component': expected, 'result': asdict(result),
                     'reward_C': failure_reward(result.component) if result.component != 'ok' else None}
            if name in ('gpu_oom', 'infinite_kernel'):
                deadline = time.monotonic() + 15
                while pynvml.nvmlDeviceGetMemoryInfo(gpu).used > baseline and time.monotonic() < deadline:
                    time.sleep(0.1)
                used = pynvml.nvmlDeviceGetMemoryInfo(gpu).used
                recovery = run(SOURCE, directory / (name + '-recovery'))
                check.update(memory_before_bytes=baseline, memory_after_bytes=used,
                             recovery=asdict(recovery))
                passed &= used <= baseline and recovery.component == 'ok'
                check['status'] = 'PASS' if passed else 'FAIL'
            checks.append(check)
            print(json.dumps(check), flush=True)
            if not passed:
                print(log[-8000:], flush=True)
                break
        if len(checks) != 6:
            checks.append({'name': 'complete_coverage', 'status': 'FAIL', 'error': 'Need all six probes'})
        if uid_processes():
            checks.append({'name': 'descendants_reaped', 'status': 'FAIL'})
    except Exception as error:
        checks.append({'name': 'blocking_failure', 'status': 'FAIL',
                       'error': f'{type(error).__name__}: {error}'})
        print(checks[-1]['error'], flush=True)
    passed = len(checks) == 6 and all(c['status'] == 'PASS' for c in checks)
    receipt = {'status': 'PASS' if passed else 'FAIL', 'scope': 'strict_launcher_C1_regression',
               'checks': checks, 'prerequisite': c0, 'artifacts': str(directory.relative_to(ROOT))}
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'strict_isolation': receipt['status'], 'path': str(directory)}), flush=True)
    raise SystemExit(0 if passed else 1)

if __name__ == '__main__':
    main()
