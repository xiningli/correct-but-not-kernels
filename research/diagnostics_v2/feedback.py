"""Only visible measurements and closed compiler-error patterns enter feedback."""
import re
from pathlib import Path
from research.diagnostics_v1.evidence import build_visible_diagnostic, group, flatten, validate_facts
from research.diagnostics_v1.common import object_hash

ERRORS = [
    ('missing_attribute', re.compile(r"module '(triton(?:\.[A-Za-z_][A-Za-z_0-9]*)*|torch(?:\.[A-Za-z_][A-Za-z_0-9]*)*)' has no attribute '([A-Za-z_][A-Za-z_0-9]{0,79})'")),
    ('unexpected_keyword', re.compile(r"got an unexpected keyword argument '([A-Za-z_][A-Za-z_0-9]{0,79})'")),
    ('undefined_name', re.compile(r"NameError: name '([A-Za-z_][A-Za-z_0-9]{0,79})' is not defined")),
]


def compiler_facts(text):
    facts = []
    for label, regex in ERRORS:
        for match in regex.finditer(text):
            row = {'kind': label, 'identifiers': list(match.groups())}
            if row not in facts:
                facts.append(row)
    return facts[:4]


def visible_feedback(task_id, source, inputs, expected, verdict, *, format_reason=None):
    if verdict['component'] == 'harness_error':
        raise RuntimeError('Infrastructure failure cannot be model feedback')
    checks = verdict.get('checks', [])
    summary = {'component': verdict['component'], 'strict_pass': verdict['correct'] is True,
               'checks_observed': len(checks), 'checks_total': len(inputs),
               'checks_passed': sum(x['correct'] for x in checks) if checks else None}
    detail = {}
    if format_reason:
        if format_reason not in {'truncated', 'invalid_program'}:
            raise ValueError('Unrecognized format reason')
        detail['format_reason'] = format_reason
    if source is not None and len(checks) == len(inputs):
        measured = build_visible_diagnostic(task_id, source, inputs, expected, verdict)
        detail['measured'] = group(measured['facts'])
    # Logs are untrusted. Never forward raw logs, harness paths, or arbitrary exceptions.
    if verdict.get('sandbox', {}).get('log'):
        p = Path(verdict['sandbox']['log'])
        with p.open('rb') as stream:
            size = p.stat().st_size
            stream.seek(max(0, size - 32768))
            detail['compiler_errors'] = compiler_facts(stream.read(32768).decode(errors='replace'))
    ops = set()
    for check in verdict.get('contract_checks', []):
        for op in check.get('ops', []):
            if re.fullmatch(r'aten\.[A-Za-z0-9_.]{1,100}', op):
                ops.add(op)
    if ops:
        detail['forbidden_compute_ops'] = sorted(ops)
    return {'scope': 'visible_only', 'summary': summary, 'detail': detail}


def project(feedback, condition):
    if set(feedback) != {'scope', 'summary', 'detail'} or feedback['scope'] != 'visible_only':
        raise ValueError('Feedback contains hidden or unrecognized fields')
    summary = feedback['summary']
    if set(summary) != {'component', 'strict_pass', 'checks_observed', 'checks_total', 'checks_passed'}:
        raise ValueError('Unrecognized summary fields')
    if set(feedback['detail']) - {'measured', 'compiler_errors', 'format_reason', 'forbidden_compute_ops'}:
        raise ValueError('Unrecognized diagnostic fields')
    if 'measured' in feedback['detail']:
        validate_facts(flatten(feedback['detail']['measured']))
    if condition == 'revision_verdict':
        return {'summary': summary}
    if condition in {'revision_diagnostic', 'initial', 'independent'}:
        return {'summary': summary, 'detail': feedback['detail']}
    raise ValueError('Unknown diagnostic condition')
