"""Descriptive complete-episode endpoints and distinct physical/logical costs."""
from collections import defaultdict, Counter
import json
from pathlib import Path
from research.diagnostics_v2.common import save


def summarize(episodes, nodes):
    groups = defaultdict(list); seen = set(); index = {}
    for n in nodes:
        key = (n['model'], n['id'])
        if key in index:
            raise ValueError('Duplicate physical generation')
        index[key] = n
    for e in episodes:
        key = (e['model'], e['task'], e['replicate'], e['condition'])
        if key in seen:
            raise ValueError('Duplicate episode')
        seen.add(key); groups[(e['model'], e['condition'])].append(e)
        if e['selected'] != e['trace'][-1] or any((e['model'], n) not in index for n in e['trace']):
            raise ValueError('Invalid selected candidate or trace')
    cells = []
    for (model, condition), values in sorted(groups.items()):
        logical = [index[(model, n)] for e in values for n in e['trace']]
        cells.append({'model': model, 'condition': condition, 'episodes': len(values),
            'independent_defects': len({e['task'] for e in values}),
            'resolved': sum(e['resolved'] for e in values),
            'visible_pass': sum(e['visible_pass'] for e in values),
            'logical_attempts': len(logical), 'prompt_tokens': sum(n['prompt_tokens'] for n in logical),
            'output_tokens': sum(n['new_tokens'] for n in logical),
            'visible_evaluation_seconds': sum(n['evaluation_seconds'] for n in logical),
            'selected_components': dict(Counter(e['selected_component'] for e in values))})
    return {'episodes': len(episodes), 'unique_generations': len(nodes),
        'physical_prompt_tokens': sum(n['prompt_tokens'] for n in nodes),
        'physical_output_tokens': sum(n['new_tokens'] for n in nodes), 'cells': cells,
        'limits': ['Two previously seen reduced defects, not an independent test set.',
                   'First candidate/evaluation is shared across the three branches; repetitions are not new defects.',
                   'First visible pass is selected; withheld scores never drive generation or selection.',
                   'Matched attempt/output caps are not matched actual compute, FLOPs or latency.',
                   '12B/31B checkpoints differ beyond parameter count; this is not a causal scaling-law test.',
                   'Rich feedback may omit the root cause; failure does not establish intrinsic inability.',
                   'Descriptive development results only; no p-values, generalization, or prize claim.']}


def report(directory):
    directory = Path(directory)
    def rows(name):
        p = directory / name
        return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
    data = summarize(rows('episodes.jsonl'), rows('nodes.jsonl'))
    outcome = directory / 'outcome.json'
    data['outcome'] = json.loads(outcome.read_text()) if outcome.exists() else {'status': 'partial'}
    data['generation_batches'] = [json.loads(p.read_text()) for p in sorted(directory.glob('*/round-*/cost.json'))]
    save(directory / 'REPORT.json', data)
    lines = ['# Larger Gemma and iterative GPU repair: development', '',
        '| Model | Condition | Episodes | Resolved | Visible pass | Logical attempts |',
        '|---|---|---:|---:|---:|---:|']
    lines += [f"| {c['model']} | {c['condition']} | {c['episodes']} | {c['resolved']} | {c['visible_pass']} | {c['logical_attempts']} |" for c in data['cells']]
    lines += ['', 'Unique generated candidates: ' + str(data['unique_generations']), '', *data['limits'],
              '', 'Outcome: ' + data['outcome']['status']]
    (directory / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return data
