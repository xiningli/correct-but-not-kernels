"""D1/D2: identical initial repairs with counterbalanced diagnostic presentation."""
import ast
import hashlib
import random

from research.diagnostics_v1.common import object_hash
from research.diagnostics_v1.evidence import CONDITIONS, render, validate_diagnostic
from research.diagnostics_v1.tasks import SYSTEM
from research.reliability_v2.repair_protocol import sampling_seed
from research.reliability_v2.fixed_interface import extract_code


def messages(task, diagnostic, condition):
    validate_diagnostic(diagnostic, task)
    evidence = render(diagnostic['facts'], condition)
    prompt = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content':
        task['issue'] + '\n\nCurrent implementation:\n```python\n' + task['source']
        + '\n```\n\n' + evidence['text']}]
    return prompt, {k: v for k, v in evidence.items() if k != 'text'}


def entries(tasks, diagnostics, plan):
    if (set(tasks) != set(plan['development']['task_ids']) or
            set(diagnostics) != set(tasks) or plan['development']['conditions'] != list(CONDITIONS)
            or plan['formal_test_enabled'] or plan['training_enabled']):
        raise ValueError('Development scope changed')
    rows = []
    for task_id, task in sorted(tasks.items()):
        for replicate in range(plan['development']['replicates']):
            for condition in CONDITIONS:
                prompt, meta = messages(task, diagnostics[task_id], condition)
                rows.append({'task': task_id, 'replicate': replicate, 'condition': condition,
                    'messages': prompt, 'prompt_sha256': object_hash(prompt), **meta,
                    # Common seeds across conditions; they are not extra tasks.
                    'sampling_seed': sampling_seed(plan['seeds']['generation'], task_id, replicate, 0)})
    random.Random(plan['seeds']['order']).shuffle(rows)
    return rows


def validate_entries(rows, expected):
    if rows != expected:
        raise ValueError('Prompt/grid/evidence drift')


def candidate_source(text, finish_reason):
    if finish_reason != 'stop':
        return None
    source = extract_code(text)
    if source is None:
        return None
    if sum(isinstance(n, ast.ClassDef) and n.name == 'ModelNew' for n in ast.parse(source).body) != 1:
        return None
    return source
