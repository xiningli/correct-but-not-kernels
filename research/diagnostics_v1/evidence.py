"""D1/D2: allowlisted visible measurements, lossless presentation interventions.

No root-cause inference, source localization from a gold patch, raw process logs,
or withheld result is accepted here. The controller supplies only visible runs.
"""
import hashlib
import json
import math
from pathlib import Path
import re

from env.correctness import ATOL, RTOL, read_output
from research.diagnostics_v1.common import canonical, object_hash

CONDITIONS = ('verdict_only', 'facts_flat', 'facts_grouped')
KEY = re.compile(r'(result\.(component|passed|total)|tests\.\d+\.(passed|component|mismatch_count|max_abs_error|first_index|actual|expected)|tests\.\d+\.inputs\.\d+\.(shape|stride|dtype|storage_offset))\Z')
SUMMARY = {'result.component', 'result.passed', 'result.total'}
COMPONENTS = {'correct', 'incorrect', 'numerical', 'nonfinite', 'output_type', 'shape_dtype',
              'format', 'crash', 'compile', 'timeout', 'contract_violation', 'sandbox_blocked',
              'gpu_oom', 'gpu_memory_limit', 'cpu_memory_limit', 'cpu_limit', 'disk_limit',
              'process_limit'}


def validate_facts(facts):
    if type(facts) is not dict or not SUMMARY <= facts.keys() or any(not KEY.fullmatch(k) for k in facts):
        raise ValueError('Unrecognized or missing diagnostic fields')
    for key, value in facts.items():
        field = key.rsplit('.', 1)[-1]
        if field == 'component':
            ok = value in COMPONENTS if type(value) is str else False
        elif field == 'passed' and key.startswith('tests.'):
            ok = type(value) is bool
        elif field in {'passed', 'total', 'mismatch_count', 'storage_offset'}:
            ok = type(value) is int and value >= 0
        elif field in {'shape', 'stride', 'first_index'}:
            ok = type(value) is list and all(type(x) is int and x >= 0 for x in value)
        elif field == 'dtype':
            ok = value in {'torch.float32', 'torch.float64', 'torch.float16', 'torch.bfloat16',
                           'torch.int64', 'torch.int32', 'torch.int16', 'torch.int8', 'torch.uint8', 'torch.bool'}
        else:
            ok = ((type(value) in {int, float} and math.isfinite(value)) or
                  (type(value) is str and value in {'NaN', '+Infinity', '-Infinity'}))
        if not ok:
            raise ValueError('Invalid diagnostic value: ' + key)
    if not 0 < facts['result.total'] or facts['result.passed'] > facts['result.total']:
        raise ValueError('Invalid visible test counts')
    return facts


def group(facts):
    tree = {}
    for key, value in sorted(facts.items()):
        cursor = tree
        parts = key.split('.')
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return tree


def flatten(tree, prefix=''):
    result = {}
    for key, value in tree.items():
        name = prefix + key
        if type(value) is dict:
            result.update(flatten(value, name + '.'))
        else:
            result[name] = value
    return result


def render(facts, condition):
    validate_facts(facts)
    if condition not in CONDITIONS:
        raise ValueError('Undeclared feedback condition')
    exposed = {k: v for k, v in facts.items() if condition != 'verdict_only' or k in SUMMARY}
    payload = group(exposed) if condition == 'facts_grouped' else exposed
    # Both rich conditions retain *all* measured facts; do not summarize/truncate.
    text = 'Observed visible-test facts:\n```json\n' + json.dumps(payload, sort_keys=True, indent=2) + '\n```'
    return {'text': text, 'exposed_facts_sha256': object_hash(exposed),
            'all_facts_sha256': object_hash(facts)}


def number(value):
    value = float(value)
    return value if math.isfinite(value) else 'NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity'


def build_visible_diagnostic(task_id, source, inputs, expected, verdict):
    import torch
    if verdict['component'] == 'harness_error':
        raise RuntimeError('Infrastructure failure cannot become model feedback')
    checks = verdict.get('checks', [])
    if len(checks) != len(inputs) or len(expected) != len(inputs):
        raise ValueError('Initial development diagnostic requires all visible outputs')
    facts = {'result.component': verdict['component'], 'result.total': len(inputs),
             'result.passed': sum(c['correct'] for c in checks)}
    work = Path(verdict['sandbox']['workdir'])
    for i, (values, wanted, check) in enumerate(zip(inputs, expected, checks, strict=True)):
        prefix = f'tests.{i}.'
        facts.update({prefix + 'passed': check['correct'], prefix + 'component': check['component']})
        for j, x in enumerate(values):
            if not isinstance(x, torch.Tensor):
                raise ValueError('Development suite must contain tensors')
            facts.update({prefix + f'inputs.{j}.shape': list(x.shape),
                          prefix + f'inputs.{j}.stride': list(x.stride()),
                          prefix + f'inputs.{j}.dtype': str(x.dtype),
                          prefix + f'inputs.{j}.storage_offset': x.storage_offset()})
        actual = read_output(work / f'output-{i}.safetensors', wanted)
        close = torch.isclose(actual, wanted, rtol=RTOL, atol=ATOL) & torch.isfinite(actual)
        bad = (~close).nonzero()
        facts[prefix + 'mismatch_count'] = int((~close).sum())
        facts[prefix + 'max_abs_error'] = number((actual.double() - wanted.double()).abs().max())
        if bad.shape[0]:
            index = tuple(int(x) for x in bad[0])
            facts[prefix + 'first_index'] = list(index)
            facts[prefix + 'actual'] = number(actual[index])
            facts[prefix + 'expected'] = number(wanted[index])
    validate_facts(facts)
    return {'schema': 1, 'scope': 'development_visible_only', 'task': task_id,
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'facts': facts, 'facts_sha256': object_hash(facts)}


def validate_diagnostic(diagnostic, task):
    if set(diagnostic) != {'schema', 'scope', 'task', 'source_sha256', 'facts', 'facts_sha256'}:
        raise ValueError('Unrecognized diagnostic envelope')
    if (diagnostic['schema'] != 1 or diagnostic['scope'] != 'development_visible_only'
            or diagnostic['task'] != task['id'] or diagnostic['source_sha256'] != task['source_sha256']
            or diagnostic['facts_sha256'] != object_hash(validate_facts(diagnostic['facts']))):
        raise ValueError('Diagnostic provenance mismatch')
