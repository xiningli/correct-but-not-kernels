"""D1/D2: public problems separated from upstream fixes and withheld inputs."""
import ast
import hashlib
import io
import re
import tokenize

from research.reliability_v2 import repair_cases

CASES = {'development/case-001': 'softcap_tail',
         'development/case-002': 'independent_strides'}
ISSUES = {
    'development/case-001': (
        'Repair the supplied GPU cross-entropy implementation. For each row x, '
        'compute z=tanh(x), then logsumexp(z)-z[label]. Return one float32 loss '
        'per row. Labels are valid class indices; class counts may be any positive '
        'integer supported by this implementation. Inputs may be float32, float16 '
        'or bfloat16. Some supported inputs currently produce incorrect values.'),
    'development/case-002': (
        'Repair the supplied selective log-softmax implementation. Logits have '
        'shape (B,L+1,V); use the first L positions and temperature 0.7. For each '
        '(b,l), return logits[b,l,ids[b,l]]/0.7 minus logsumexp(logits[b,l,:]/0.7), '
        'or zero when mask[b,l] is false. Return float32 shape (B,L). IDs are '
        'valid indices. Supported tensor views may have independent strides. '
        'Some supported inputs currently produce incorrect values.')}
SYSTEM = '''Repair the existing program using custom CUDA C++ or Triton kernels.
Return the complete revised Python program in one Python code block, including
ModelNew(torch.nn.Module) and its existing forward interface. No explanation.
Preserve inputs and return fresh output storage. PyTorch may only allocate empty
outputs or access shape/stride metadata and noncomputational views. Do not replace
the mathematical computation with PyTorch, vendor libraries, CPU computation,
reference replay, or runtime inspection. Keep the upstream license notice.
The diagnostic contains observations, not a suggested fix. Implement the full
specified domain, including inputs beyond the visible observations.
'''


def public_task(task_id):
    case = CASES[task_id]
    source, meta = repair_cases.build(case, 'before')
    marker = '# Reduced development reproduction of ' + meta['upstream'] + '\n'
    if source.count(marker) != 1:
        raise ValueError('Cannot remove the exact upstream answer locator')
    source = source.replace(marker, '')
    before = ast.dump(ast.parse(source), include_attributes=False)
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    source = tokenize.untokenize([t._replace(string=re.sub(
        r'https?://\S*/pull/\S+', '[upstream PR locator removed]', t.string))
        if t.type == tokenize.COMMENT else t for t in tokens])
    if ast.dump(ast.parse(source), include_attributes=False) != before:
        raise ValueError('Public comment redaction changed executable code')
    return {'id': task_id, 'split': 'development', 'scope': meta['scope'],
            'issue': ISSUES[task_id], 'source': source,
            'public_adaptations': ['Remove PR locators in comments; executable AST unchanged; license retained.'],
            'source_sha256': hashlib.sha256(source.encode()).hexdigest()}


def visible_suite(task_id):
    return repair_cases.regressions(CASES[task_id], hidden=False)


def withheld_suite(task_id):
    # Called only by the evaluator, never by prompt construction or sampling.
    return repair_cases.regressions(CASES[task_id], hidden=True)


def fixed_control(task_id):
    return repair_cases.build(CASES[task_id], 'after')
