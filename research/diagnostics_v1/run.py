"""D1/D2: sealed one-turn diagnostic repair grid, then independent scoring."""
import argparse
import difflib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from research.diagnostics_v1.common import ROOT, digest, save, source_seal, utc
from research.diagnostics_v1.controls import validate_controls
from research.diagnostics_v1.prepare import load_prepared
from research.diagnostics_v1.protocol import candidate_source, entries
from research.diagnostics_v1.report import report
from research.diagnostics_v1.tasks import visible_suite, withheld_suite
from research.reliability_v2.strict_evaluator import evaluate
from tools.gatelib import git_state
from tools.replication import verify_frozen


def execute(prepared, controls, out):
    import torch
    torch.set_num_threads(4)
    verify_frozen(ROOT / 'runs/replication-v1.1')
    plan, tasks, policies = load_prepared(prepared)
    diagnostics = validate_controls(controls, prepared)
    grid = entries(tasks, diagnostics, plan)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    (out / 'config.yaml').write_bytes(Path(prepared, 'config.yaml').read_bytes())
    save(out / 'git_sha', git_state())
    (out / 'seed').write_text(str(plan['seeds']['generation']) + '\n')
    save(out / 'versions.json', {p: importlib.metadata.version(p) for p in
         ['torch', 'triton', 'vllm', 'transformers', 'safetensors']})
    save(out / 'source-seal.json', source_seal())
    save(out / 'bindings.json', {'prepared_sha256': digest(Path(prepared) / 'prepared.json'),
         'controls_sha256': digest(Path(controls) / 'receipt.json')})
    metrics = out / 'metrics.jsonl'; metrics.touch()
    start, scored = time.monotonic(), 0
    outcome = {'status': 'running', 'utc': utc()}
    try:
        for label in plan['development']['policies']:
            if time.monotonic() - start > plan['limits']['run_wall_seconds']:
                raise RuntimeError('Development run exceeded its declared wall budget')
            if json.loads((out / 'source-seal.json').read_text()) != source_seal():
                raise ValueError('Source changed during experiment')
            directory = out / label; directory.mkdir()
            diagnostic_path = Path(controls).resolve() / 'visible-diagnostics.json'
            request = {'scope': 'two_development_defects_only', 'prepared': str(Path(prepared).resolve()),
                       'visible_diagnostics': str(diagnostic_path),
                       'visible_diagnostics_sha256': digest(diagnostic_path), 'policy': label, 'entries': grid}
            save(directory / 'request.json', request)
            save(out / 'progress.json', {'stage': 'generation', 'policy': label, 'scored': scored,
                                        'total': plan['development']['expected_attempts'], 'utc': utc()})
            begin = time.monotonic()
            with (directory / 'generation.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'research.diagnostics_v1.sample',
                    '--request', str(directory / 'request.json'), '--out', str(directory / 'generations.jsonl')],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                    timeout=plan['limits']['generation_policy_seconds'])
            save(directory / 'generation-cost.json', {'seconds': time.monotonic() - begin})
            rows = [json.loads(s) for s in (directory / 'generations.jsonl').read_text().splitlines()]
            if len(rows) != len(grid):
                raise ValueError('Incomplete generation grid')
            for index, (row, entry) in enumerate(zip(rows, grid, strict=True)):
                if row['policy'] != label or any(row[k] != v for k, v in entry.items() if k != 'messages'):
                    raise ValueError('Generation identity/provenance mismatch')
                begin = time.monotonic()
                candidate = directory / f'candidate-{index:03d}'; candidate.mkdir()
                source = candidate_source(row['text'], row['finish_reason'])
                result = {'component': 'format', 'resolved': False, 'suites': {}}
                if source is not None:
                    (candidate / 'program.py').write_text(source)
                    (candidate / 'patch.diff').write_text(''.join(difflib.unified_diff(
                        tasks[row['task']]['source'].splitlines(True), source.splitlines(True),
                        fromfile='before.py', tofile='candidate.py')))
                    for suite, factory in [('visible', visible_suite), ('withheld', withheld_suite)]:
                        inputs, expected = factory(row['task'])
                        verdict = evaluate(source, inputs, expected, candidate / suite, seed=plan['seeds']['model_init'])
                        result['suites'][suite] = verdict
                        if verdict['component'] == 'harness_error':
                            break
                    if any(v['component'] == 'harness_error' for v in result['suites'].values()):
                        result.update(component='harness_error', resolved=None)
                    else:
                        resolved = len(result['suites']) == 2 and all(v['correct'] for v in result['suites'].values())
                        failure = next((v['component'] for v in result['suites'].values() if not v['correct']), 'correct')
                        result.update(component=failure, resolved=resolved)
                saved = {k: v for k, v in row.items() if k not in {'text', 'token_ids'}}
                saved.update(**result, evaluation_seconds=time.monotonic() - begin,
                             candidate_path=str(candidate.relative_to(out)))
                with metrics.open('a') as stream:
                    stream.write(json.dumps(saved, allow_nan=False) + '\n'); stream.flush()
                scored += 1
                save(out / 'progress.json', {'stage': 'evaluation', 'policy': label, 'task': row['task'],
                    'condition': row['condition'], 'scored': scored,
                    'total': plan['development']['expected_attempts'], 'utc': utc()})
                report(out)
                if result['component'] == 'harness_error':
                    raise RuntimeError('Infrastructure failure; preserve evidence and stop')
        if scored != plan['development']['expected_attempts']:
            raise ValueError('Final attempt denominator mismatch')
        outcome['status'] = 'complete'
    except Exception as error:
        outcome.update(status='stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        outcome.update(scored=scored, seconds=time.monotonic() - start, finished_utc=utc())
        save(out / 'outcome.json', outcome)
        report(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    execute(a.prepared, a.controls, a.out)


if __name__ == '__main__':
    main()
