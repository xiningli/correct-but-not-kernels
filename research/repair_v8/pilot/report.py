"""Descriptive development report; no p-values from three developer defects."""
from __future__ import annotations
import json
from pathlib import Path
from research.repair_v8.prepare import sha
from .artifacts import write_json


def report(root, config):
    selected = json.loads((root / 'selected-patches.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((root / 'judging').glob('*/final.json'))]
    if {x['episode'] for x in rows} != set(selected):
        raise ValueError('Report requires every predeclared episode')
    arms = {}
    for arm in config['arms']:
        subset = [r for r in rows if r['arm'] == arm]
        sequences = [s['failure_signatures'] for s in selected.values() if s['arm'] == arm]
        arms[arm] = {'episodes': len(subset), 'reliable_repairs': sum(r['success'] for r in subset),
                     'visible_pass': sum(r['visible_pass'] for r in subset),
                     'visible_pass_hidden_fail': sum(r['visible_pass'] and not r['success'] for r in subset),
                     'tasks_repaired_at_least_once': len({r['task'] for r in subset if r['success']}),
                     'logical_prompt_tokens': sum(s['logical_prompt_tokens'] for s in selected.values() if s['arm'] == arm),
                     'logical_output_tokens': sum(s['logical_output_tokens'] for s in selected.values() if s['arm'] == arm),
                     'logical_generation_seconds': sum(s['logical_generation_seconds'] for s in selected.values() if s['arm'] == arm),
                     'logical_visible_seconds': sum(s['logical_visible_seconds'] for s in selected.values() if s['arm'] == arm),
                     'consecutive_failure_signature_repeats': sum(a is not None and a == b for seq in sequences for a, b in zip(seq, seq[1:])),
                     'consecutive_attempt_pairs': sum(max(0, len(s) - 1) for s in sequences),
                     'episodes_with_observed_semantic_regression': sum(bool(r['observed_semantic_regressions']) for r in subset)}
    physical = sum(len(p.read_text().splitlines()) for p in root.glob('round-*/generations.jsonl'))
    result = {'status': 'complete', 'scope': config['scope'], 'independent_development_defects': len(config['tasks']),
              'formal_tasks': 0, 'training_runs': 0, 'arms': arms, 'physical_generations': physical,
              'selected_manifest_sha256': sha(root / 'selected-patches.json'), 'episodes': rows,
              'interpretation': config['interpretation'], 'cost_note': 'First draws shared A/B and C/D. Logical costs are fully charged to each episode; physical work is reported separately.'}
    write_json(root / 'REPORT.json', result)
    lines = ['# Gemma CUDA repair: three-task development pilot', '', config['interpretation'], '',
             '| Arm | Episodes | Reliable repairs | Visible pass | Visible pass but final failure |',
             '|---|---:|---:|---:|---:|']
    for arm, r in arms.items():
        lines.append(f'| {arm} | {r["episodes"]} | {r["reliable_repairs"]} | {r["visible_pass"]} | {r["visible_pass_hidden_fail"]} |')
    lines += ['', f'Independent defects: {len(config["tasks"])}. Physical generations: {physical}.', '',
              'All patches were selected from visible feedback and sealed before hidden judging.',
              'Success requires visible, hidden, original-repository preservation and memcheck panels.',
              'Runtime observations are bounded checks, not a proof against arbitrary native tampering.',
              'Different samples and input seeds are not additional independent defects.',
              'No significance, generalization, speedup or RL benefit claim is made.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    return result
