"""Resume a sealed local GPU replay; report terminal outcomes, never power off host."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from .common import ROOT, load, read, write, append, sha, utc, IMAGE
from .backend import run as container_run
from .controls import controls
from .judge import judge
from .report import report


def archive(out, prepared):
    target = out / 'evidence.tgz'
    with tarfile.open(target, 'w:gz', compresslevel=1) as tar:
        for root, prefix in [(out, 'run'), (prepared, 'prepared')]:
            for p in sorted(root.rglob('*')):
                rel = p.relative_to(root)
                if not p.is_file() or p.is_symlink() or p == target or 'work' in rel.parts or 'scratch' in rel.parts:
                    continue
                tar.add(p, arcname=str(Path(prefix) / rel), recursive=False)
    write(out / 'archive.json', {'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target)})


def reference(task, prepared, directory, plan):
    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / 'inputs'; inputs.mkdir()
    source = prepared / task['file']
    if sha(source) != task['sha256']:
        raise ValueError('Reference changed')
    shutil.copyfile(source, inputs / 'reference.py')
    write(inputs / 'request.json', {'seeds': task['seeds'], 'plan': plan})
    caps = {**plan, 'worker_disk_limit_bytes': plan['reference_max_task_storage_bytes'] + (1 << 20)}
    execution = container_run(Path(__file__).with_name('reference_worker.py'), inputs,
                              directory / 'execution', caps, seconds=plan['reference_seconds'])
    output = Path(execution['work'])
    if execution['reason'] in {'timeout', 'cpu_memory_limit', 'gpu_memory_limit', 'disk_limit'}:
        result = {'eligible': False, 'reason': 'reference_resource_exclusion',
                  'error': execution['reason'], 'samples': []}
    elif execution['returncode'] or not (output / 'reference-result.json').exists():
        raise RuntimeError('Reference infrastructure failed: ' + execution['log'])
    else:
        result = read(output / 'reference-result.json')
    result['execution'] = execution
    write(directory / 'reference.json', result)
    if result.get('reason') == 'reference_unavailable':
        raise RuntimeError('Trusted reference requires investigation: ' + str(result['error']))
    return result, output


def execute(prepared, out):
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    manifest, plan = load(prepared)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = {'preparation_sha256': sha(prepared / 'manifest.json'), 'image': IMAGE}
        if (out / 'binding.json').exists() and read(out / 'binding.json') != binding:
            raise ValueError('Resume identity mismatch')
        write(out / 'binding.json', binding)
        if (out / 'outcome.json').exists() and read(out / 'outcome.json')['status'] == 'complete':
            return report(out, plan)
        session = utc().replace(':', '').replace('+', '_')
        write(out / 'process.json', {'pid': os.getpid(), 'utc': utc(), 'host_shutdown': False})
        write(out / 'outcome.json', {'status': 'running', 'utc': utc()})
        started = time.monotonic()
        records = [json.loads(s) for s in (out / 'metrics.jsonl').read_text().splitlines()] if (out / 'metrics.jsonl').exists() else []
        done = {(r['policy'], r['task'], r['sample']) for r in records}
        if len(done) != len(records):
            raise ValueError('Duplicate saved outcomes')
        rows = [json.loads(s) for s in (prepared / 'programs.jsonl').read_text().splitlines()]
        groups = defaultdict(list)
        for r in rows:
            groups[r['task']].append(r)
        eligibility = read(out / 'eligibility.json') if (out / 'eligibility.json').exists() else {}
        outcome = None
        try:
            write(out / 'progress.json', {'stage': 'controls', 'utc': utc(), 'recorded': len(done)})
            controls(out / 'sessions' / session / 'controls', plan)
            for index, task in enumerate(read(prepared / 'tasks.json')):
                load(prepared)
                task_id = task['task']
                grid = groups[task_id]
                if all((r['policy'], task_id, r['sample']) in done for r in grid):
                    continue
                if task_id in eligibility and not eligibility[task_id]['eligible']:
                    continue
                if time.monotonic() - started > plan['total_wall_seconds']:
                    raise RuntimeError('Controller wall-time budget reached')
                directory = out / 'sessions' / session / f'task-{index:03d}'
                write(out / 'progress.json', {'stage': 'reference_qualification', 'task': task_id,
                    'task_index': index, 'recorded': len(done), 'total': manifest['attempts'], 'utc': utc()})
                qualified, ref_work = reference(task, prepared, directory / 'reference', plan)
                eligibility[task_id] = {k: v for k, v in qualified.items() if k != 'samples'}
                eligibility[task_id]['sample_manifest'] = qualified['samples']
                write(out / 'eligibility.json', eligibility)
                if not qualified['eligible']:
                    shutil.rmtree(ref_work, ignore_errors=True)
                    report(out, plan)
                    continue
                # Cache only identical source + exact original input seed within
                # this qualified task. Every logical attempt retains its row.
                cache = {}
                for row in grid:
                    key = (row['policy'], task_id, row['sample'])
                    if key in done:
                        continue
                    if time.monotonic() - started > plan['total_wall_seconds']:
                        raise RuntimeError('Controller wall-time budget reached')
                    write(out / 'progress.json', {'stage': 'candidate_replay', 'task': task_id,
                        'policy': row['policy'], 'sample': row['sample'], 'recorded': len(done),
                        'total': manifest['attempts'], 'utc': utc()})
                    digest = row['candidate_sha256']
                    source = (prepared / 'programs' / (digest + '.py')).read_text() if digest else None
                    pair = (digest, row['input_seed'])
                    physical_reuse = pair in cache
                    if physical_reuse:
                        result = cache[pair]
                    else:
                        sample = ref_work / f"sample-{row['sample']}"
                        result = judge(source, sample / 'inputs', sample / 'oracle',
                            directory / f"{row['sample']}-{row['policy']}", plan)
                        cache[pair] = result
                    saved = {**row, 'result': result, 'physical_execution_reused': physical_reuse, 'utc': utc()}
                    append(out / 'metrics.jsonl', saved)
                    done.add(key)
                    if len(done) % 56 == 0:
                        report(out, plan)
                shutil.rmtree(ref_work, ignore_errors=True)
                report(out, plan)
            outcome = {'status': 'complete', 'utc': utc(), 'attempts': len(done),
                'excluded_tasks': sum(not r['eligible'] for r in eligibility.values()),
                'model_outputs': 0, 'training': False}
        except Exception as error:
            outcome = {'status': 'blocked', 'utc': utc(), 'attempts': len(done),
                       'error': f'{type(error).__name__}: {error}', 'model_outputs': 0}
        finally:
            if outcome is not None:
                write(out / 'outcome.json', outcome)
                report(out, plan)
                archive(out, prepared)
                write(out / 'progress.json', {'stage': outcome['status'], 'recorded': len(done),
                                              'utc': utc(), 'host_shutdown': False})
        print(json.dumps(outcome), flush=True)
        return outcome


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    execute(args.prepared, args.out)
