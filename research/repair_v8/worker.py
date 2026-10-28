"""One trusted qualification probe per process. Not a generated-code sandbox."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--oracle-only', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    row = {'task': args.task, 'case': args.case, 'seed': args.seed,
           'oracle_only': args.oracle_only, 'status': 'infrastructure_error', 'stage': 'environment'}
    code = 3
    try:
        import torch
        from .cases import TASKS, configure_torch, load_owner, run_case
        configure_torch()
        row['torch'] = torch.__version__
        row['runtime_cuda'] = torch.version.cuda
        if args.oracle_only:
            module, device = None, 'cpu'
        else:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; CPU/reference success is not GPU qualification')
            row['stage'] = 'original_package_import'
            module = load_owner(args.task, args.checkout)
            row['module_origin'] = module.__file__
            row['device'] = torch.cuda.get_device_name(0)
            device = 'cuda'
        row['stage'] = 'probe'
        profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) if args.profile else contextlib.nullcontext())
        with profiler as trace:
            row['result'] = run_case(args.task, args.case, args.seed, module, device)
            if device == 'cuda':
                torch.cuda.synchronize()
        if args.profile:
            row['stage'] = 'kernel_trace'
            row['cuda_kernel_names'] = sorted({e.name for e in trace.events()
                if str(e.device_type).endswith('CUDA')})
            wanted = TASKS[args.task]['kernel_name']
            if not any(wanted in name for name in row['cuda_kernel_names']):
                raise RuntimeError('Required custom GPU kernel absent from profiler trace: ' + wanted)
        row.update(status='pass', stage='complete')
        code = 0
    except AssertionError as error:
        row.update(status='test_failure', error=str(error), traceback=traceback.format_exc()[-16000:])
        code = 2
    except Exception as error:
        row['error'] = type(error).__name__ + ': ' + str(error)
        row['traceback'] = traceback.format_exc()[-16000:]
        # Runtime exceptions remain distinct: only a reviewed bug signature can
        # count as reproduction. Import, OOM and absent-device errors never do.
        if row['stage'] == 'probe' and not args.oracle_only:
            row['status'] = 'probe_exception'
            code = 2
    row['seconds'] = time.monotonic() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: row[k] for k in ['task', 'case', 'status', 'stage']}))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
