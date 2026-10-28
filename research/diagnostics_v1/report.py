"""D1/D2: descriptive development denominators; never pool repeats as defects."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from research.diagnostics_v1.common import save


def summarize(rows):
    seen, groups = set(), defaultdict(list)
    for r in rows:
        key = (r['policy'], r['task'], r['condition'], r['replicate'])
        if key in seen:
            raise ValueError('Duplicate repair attempt')
        seen.add(key)
        groups[(r['policy'], r['condition'])].append(r)
    cells = []
    for (policy, condition), records in sorted(groups.items()):
        cells.append({'policy': policy, 'condition': condition, 'attempts': len(records),
            'independent_defects': len({r['task'] for r in records}),
            'resolved': sum(r['resolved'] is True for r in records),
            'unresolved': sum(r['resolved'] is False for r in records),
            'unknown': sum(r['resolved'] is None for r in records),
            'components': dict(Counter(r['component'] for r in records)),
            'prompt_tokens': sum(r['prompt_tokens'] for r in records),
            'generated_tokens': sum(r['new_tokens'] for r in records),
            'evaluation_seconds': sum(r['evaluation_seconds'] for r in records)})
    return {'attempts': len(rows), 'independent_defects': len({r['task'] for r in rows}),
            'cells': cells, 'limits': [
                'Two previously seen reduced defects from one repository; no full-repository or generalization claim.',
                'Repeated generations and saved training seeds are not independent defects.',
                'Flat/grouped conditions contain identical measured facts; they are not unfiltered raw stderr.',
                'A failure after diagnostic feedback does not establish that the evidence was sufficient.',
                'No new training, strong-model comparison, significance test, or model-size claim.',
                'Actual prompt/output tokens and GPU evaluation time are reported; profiling is not kernel timing.']}


def report(directory):
    directory = Path(directory)
    metrics = directory / 'metrics.jsonl'
    rows = [json.loads(s) for s in metrics.read_text().splitlines()] if metrics.exists() else []
    data = summarize(rows)
    outcome = directory / 'outcome.json'
    data['outcome'] = json.loads(outcome.read_text()) if outcome.exists() else {'status': 'partial'}
    save(directory / 'REPORT.json', data)
    lines = ['# Execution evidence for Gemma GPU repair: development', '',
             '| Policy | Feedback | Attempts | Independent defects | Resolved | Unknown |',
             '|---|---|---:|---:|---:|---:|']
    for c in data['cells']:
        lines.append(f"| {c['policy']} | {c['condition']} | {c['attempts']} | {c['independent_defects']} | {c['resolved']} | {c['unknown']} |")
    lines += ['', *data['limits'], '', 'Outcome: ' + data['outcome']['status']]
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return data
