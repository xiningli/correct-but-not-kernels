"""K1/K3/K4: retrospective, read-only reliability audit of the sealed v1.1 grids.

This separate namespace preserves the original source seal. Programs are parsed
as text/AST only; no candidate is imported, compiled, or executed. Source markers
and traceback hints are diagnostic observations, never verified mechanisms.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics

from env.reward import extract_program
from env.tasks import verified_catalog
from eval.replication import read_rows, validate_rows
from eval.replication_report import load_complete
from tools.replication import sha256, verify_frozen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / 'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/local-analysis-verification.json'
LOG_RULES = {
    'storage_io': r'No space left on device|Disk quota exceeded|File too large',
    'api_import_or_name': r'AttributeError:|NameError:|ImportError:|ModuleNotFoundError:',
    'illegal_memory_or_assert': r'illegal memory access|misaligned address|device-side assert',
    'shape_or_broadcast': r'shape mismatch|cannot be broadcast|size mismatch|sizes? of tensors? must match|invalid shape',
    'dtype_or_device': r'expected scalar type|same device|dtype mismatch|unsupported dtype',
    'type_or_call': r'TypeError:',
    'compiler_diagnostic': r'CompilationError:|UnsupportedLanguageConstruct:|ptxas fatal|(?m:^\s*(?:[\w./-]+:\d+(?::\d+)?:\s*)?(?:fatal )?error:)|Error building extension',
    'resource_or_launch': r'OutOfResources|out of memory|too many resources|invalid configuration argument',
}


def forward_ast(source: str, class_name: str):
    """Exact structural match only; absence/mismatch says nothing about equivalence."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == 'forward':
                    if (method.body and isinstance(method.body[0], ast.Expr)
                            and isinstance(method.body[0].value, ast.Constant)
                            and isinstance(method.body[0].value.value, str)):
                        method.body = method.body[1:]
                    return ast.dump(method, include_attributes=False)
    return None


def source_observation(text: str, reference: str):
    source = extract_program(text)
    if source is None:
        return {'source_markers': 'unextractable', 'exact_forward_ast_match': False,
                'source_sha256': None}
    tree = ast.parse(source)
    imports = []
    literals = []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
    triton = any(x == 'triton' or x.startswith('triton.') for x in imports)
    cuda = (any(x in {'load_inline', 'CUDAExtension'} for x in names)
            or any('__global__' in x for x in literals))
    kind = ('mixed_custom_markers' if triton and cuda else 'cuda_markers' if cuda
            else 'triton_markers' if triton else 'no_custom_markers')
    actual = forward_ast(source, 'ModelNew')
    expected = forward_ast(reference, 'Model')
    return {'source_markers': kind,
            'exact_forward_ast_match': actual is not None and actual == expected,
            'source_sha256': hashlib.sha256(source.encode()).hexdigest()}


def log_observation(text: str | None):
    if text is None:
        return {'log_available': False, 'log_hint': 'missing_log', 'log_signals': [], 'exception_tail': '',
                'output_serialization_enospc_hint': False}
    signals = [name for name, pattern in LOG_RULES.items() if re.search(pattern, text, re.I)]
    exceptions = [line.strip() for line in text.splitlines()
                  if re.match(r'^[\w.]*(?:Error|Exception|UnsupportedLanguageConstruct|OutOfResources):', line)]
    return {'log_available': True, 'log_hint': signals[0] if signals else 'unclassified_log',
            'log_signals': signals, 'exception_tail': exceptions[-1][-500:] if exceptions else '',
            'output_serialization_enospc_hint': ('No space left on device' in text
                and 'safetensors/torch.py' in text and 'in save_file' in text
                and 'in execute_worker' in text)}


def archive_file(extracted: Path, relative: str):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe archived relative path')
    path = extracted / rel
    if not path.resolve().is_relative_to(extracted.resolve()):
        raise ValueError('Archived path escapes extraction root')
    return path


def summarize(records):
    by_label = defaultdict(list)
    for row in records:
        by_label[row['label']].append(row)
    policies = {}
    for label, rows in by_label.items():
        original = Counter(r['original_component'] for r in rows)
        by_source = {}
        for category in sorted({r['source_markers'] for r in rows}):
            part = [r for r in rows if r['source_markers'] == category]
            by_source[category] = {'n': len(part), 'correct': sum(r['correct'] for r in part),
                                   'components': dict(Counter(r['original_component'] for r in part))}
        policies[label] = {'arm': rows[0]['arm'], 'n': len(rows), 'correct': sum(r['correct'] for r in rows),
                           'components': dict(original), 'source_markers': by_source,
                           'exact_forward_ast_matches': sum(r['exact_forward_ast_match'] for r in rows),
                           'crash_log_hints': dict(Counter(r['log_hint'] for r in rows if r['original_component'] == 'crash'))}
    arms = {}
    for arm in ['base', 'C', 'S', 'SFT']:
        cells = [v for v in policies.values() if v['arm'] == arm]
        if not cells:
            continue
        keys = set().union(*(v['components'] for v in cells))
        hint_keys = set().union(*(v['crash_log_hints'] for v in cells))
        arms[arm] = {'policies': len(cells), 'mean_n': statistics.fmean(v['n'] for v in cells),
                     'mean_correct': statistics.fmean(v['correct'] for v in cells),
                     'mean_components': {k: statistics.fmean(v['components'].get(k, 0) for v in cells) for k in sorted(keys)},
                     'mean_exact_forward_ast_matches': statistics.fmean(v['exact_forward_ast_matches'] for v in cells),
                     'mean_crash_log_hints': {k: statistics.fmean(v['crash_log_hints'].get(k, 0) for v in cells)
                                              for k in sorted(hint_keys)}}
    base = {(r['task'], r['sample']): r for r in by_label.get('base', [])}
    transitions = {}
    for label, rows in by_label.items():
        if label == 'base' or not base:
            continue
        assert set(base) == {(r['task'], r['sample']) for r in rows}
        counts = Counter((base[r['task'], r['sample']]['original_component'], r['original_component']) for r in rows)
        transitions[label] = [{'base': left, 'policy': right, 'n': n} for (left, right), n in sorted(counts.items())]
    affected = [r for r in records if r.get('output_serialization_enospc_hint')]
    return {'policies': policies, 'arms': arms, 'paired_outcome_transitions': transitions,
            'output_serialization_enospc': {
                'n': len(affected), 'by_policy': dict(Counter(r['label'] for r in affected)),
                'by_task': dict(Counter(r['task'] for r in affected)),
                'scope': 'Archived traceback hints; failure is at harness output serialization. '
                         'Historical filesystem cause and candidate correctness remain unresolved. '
                         'Original verdicts are unchanged.',
                'cases': [{k: r[k] for k in ('label', 'task', 'sample', 'log_path', 'log_sha256')}
                          for r in affected]}}


def review_sample(records, per_arm=24, seed=20261012):
    """Diagnostic diversity sample; deliberately NOT a prevalence estimator."""
    selected = []
    for arm in ['base', 'C', 'S', 'SFT']:
        cells = defaultdict(list)
        for row in records:
            if row['arm'] == arm:
                key = (row['original_component'], row['source_markers'], row['log_hint'])
                cells[key].append(row)
        rank = lambda r: hashlib.sha256(f"{seed}:{r['label']}:{r['task']}:{r['sample']}".encode()).hexdigest()
        for rows in cells.values():
            rows.sort(key=rank)
        count = 0
        while count < per_arm and any(cells.values()):
            for key in sorted(cells):
                if cells[key] and count < per_arm:
                    selected.append(cells[key].pop(0)); count += 1
    return selected


def run(receipt_path: Path, out: Path):
    verify_frozen(ROOT / 'runs/replication-v1.1')
    receipt = json.loads(receipt_path.read_text())
    expected = {'base'} | {f'{a}-{s}' for a in ['C', 'S', 'SFT'] for s in [20261021, 20261022, 20261023]}
    if set(receipt['policies']) != expected:
        raise ValueError('Need one shared base and all nine completed policies')
    tasks = {t.task_id: t for t in verified_catalog()}
    references = {key: task.source() for key, task in tasks.items()}
    records = []; provenance = []
    for label, saved in receipt['policies'].items():
        ev = ROOT / saved['path']
        extracted = next(p for p in ev.parents if p.name == 'extracted')
        rows = load_complete(ev / 'scores.jsonl')
        req = json.loads((ev / 'request.json').read_text())
        done = json.loads((ev / 'complete.json').read_text())
        if sha256(ev / 'scores.jsonl') != saved['scores_sha256'] or sha256(ev / 'generations.jsonl') != done['generations_sha256']:
            raise ValueError('Previously verified grid changed')
        validate_rows(read_rows(ev / 'generations.jsonl'), req)
        if len(rows) != 960 or sum(r['result']['correct'] for r in rows) != saved['correct']:
            raise ValueError('Mismatch with validated final result')
        provenance.append({'label': label, 'path': str(ev.relative_to(ROOT)),
                           'scores_sha256': done['scores_sha256'], 'generations_sha256': done['generations_sha256']})
        for row in rows:
            if row['source_sha256'] != tasks[row['task']].sha256:
                raise ValueError('Task source drift')
            result = row['result']; component = result['component']
            program = row.get('program_directory')
            logfile = archive_file(extracted, program + '/candidate/worker.log') if program else None
            log = logfile.read_text(errors='replace') if logfile is not None and logfile.exists() else None
            observed = log_observation(log) if component == 'crash' else {
                'log_available': logfile is not None and logfile.exists(), 'log_hint': 'not_a_crash',
                'log_signals': [], 'exception_tail': '', 'output_serialization_enospc_hint': False}
            record = {'label': label, 'arm': label.split('-')[0], 'task': row['task'], 'sample': row['sample'],
                      'original_component': component, 'correct': result['correct'],
                      'original_detector_flags': result['hacks'],
                      'numerical_check_components': [c['component'] for c in result['details'].get('checks', [])],
                      'scores_path': str((ev / 'scores.jsonl').relative_to(ROOT)),
                      'program_path': str(archive_file(extracted, program + '/program.py').relative_to(ROOT)) if program else '',
                      'log_path': str(logfile.relative_to(ROOT)) if logfile is not None else '',
                      'log_sha256': sha256(logfile) if logfile is not None and logfile.exists() else None,
                      **source_observation(row['text'], references[row['task']]), **observed}
            records.append(record)
    result = summarize(records)
    result.update({'scope': 'Post-hoc descriptive audit, not a new independent evaluation or causal attribution.',
                   'limitations': ['AST/import/string markers do not prove which code executed or contract compliance.',
                                  'Exact forward AST matches do not prove full-program equivalence; nonmatches do not rule out copying.',
                                  'Traceback hints may overlap; the original evaluator verdict is never replaced.',
                                  'Paired outcomes compare independently generated programs, not before/after repairs of the same program.',
                                  'Backend groups are selected after generation; their conditional success rates are not causal effects.'],
                   'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'audit_source_sha256': sha256(Path(__file__)), 'input_receipt_sha256': sha256(receipt_path),
                   'inputs': provenance, 'records': len(records)})
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'programs.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row) + '\n')
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    fields = ['label', 'task', 'sample', 'original_component', 'source_markers', 'exact_forward_ast_match',
              'log_hint', 'exception_tail', 'program_path', 'scores_path', 'log_path',
              'reviewed_source_type', 'reviewed_failure_type', 'runtime_evidence_path', 'reviewer', 'notes']
    with (out / 'manual_review.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        writer.writerows(review_sample(records))
    lines = ['# Retrospective reliability audit', '', result['scope'], '', *result['limitations'], '',
             '| Arm | Policies | Mean correct / 960 | Mean crash | Mean format | Mean incorrect | Exact forward AST matches / 960 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm, v in result['arms'].items():
        c = v['mean_components']
        lines.append(f"| {arm} | {v['policies']} | {v['mean_correct']:.2f} | {c.get('crash',0):.2f} | {c.get('format',0):.2f} | {c.get('incorrect',0):.2f} | {v['mean_exact_forward_ast_matches']:.2f} |")
    lines += ['', '| Arm | API/import/name crash hints, mean per 960 | Storage I/O crash hints, mean per 960 |',
              '|---|---:|---:|']
    for arm, v in result['arms'].items():
        hints = v['mean_crash_log_hints']
        lines.append(f"| {arm} | {hints.get('api_import_or_name', 0):.2f} | {hints.get('storage_io', 0):.2f} |")
    io = result['output_serialization_enospc']
    lines += ['', f"Output serialization ENOSPC tracebacks: {io['n']}; per policy: {io['by_policy']}; per task: {io['by_task']}.",
              io['scope'],
              'This measurement issue must be resolved in a separately versioned replay before clean per-task correctness claims.']
    lines += ['', 'The 96-case review queue is a deterministic diversity sample (24 per arm), not a prevalence estimate.',
              'Original verdicts, source observations and log hashes are retained in programs.jsonl.',
              'All source-category counts and paired outcome transitions are in summary.json.',
              'No candidate was executed. GPU runtime attribution is a separate pending study.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return {'output': str(out), 'programs': len(records), 'arms': result['arms']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.receipt, args.out), indent=2))


if __name__ == '__main__':
    main()
