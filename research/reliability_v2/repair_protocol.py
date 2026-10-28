"""K1/K4, T2/T3: bounded repair episodes without hidden-test selection leakage.

Pure orchestration: callbacks execute no code here. GPU execution must use C1.
The test callback's public result has a closed schema; hidden verdicts belong
to a separate, post-episode process. Input-token cost is retained, not assumed
equal merely because output-token caps match.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class Budget:
    attempts: int = 3
    tokens_per_attempt: int = 4096
    total_new_tokens: int = 12288
    visible_test_calls: int = 3

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('Budgets must be positive integers')
        if self.visible_test_calls < self.attempts:
            raise ValueError('Every attempt must receive the same visible evaluation')


def visible_result(value):
    """Do not accidentally feed hidden verdicts, oracle outputs or paths back."""
    required = {'passed', 'total', 'diagnostic', 'infrastructure_error'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('Visible feedback must use the closed public schema')
    if (type(value['passed']) is not int or type(value['total']) is not int
            or not 0 <= value['passed'] <= value['total'] or value['total'] < 1
            or type(value['infrastructure_error']) is not bool
            or type(value['diagnostic']) is not str):
        raise ValueError('Invalid visible-test feedback')
    if value['infrastructure_error']:
        raise RuntimeError('Harness failure; stop episode, do not score model failure')
    # Callback must produce diagnostics from visible tests only. Schema validation
    # cannot determine the provenance of arbitrary text; isolation supplies that.
    return {**value, 'diagnostic': value['diagnostic'][:6000]}


def sampling_seed(namespace, task, replicate, attempt):
    raw = f'{namespace}:{task}:{replicate}:{attempt}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big') % (2**31 - 1)


def select_patch(attempts):
    """Visible test count, then earliest. Never receive hidden outcomes here."""
    if not attempts:
        raise ValueError('No completed attempts')
    totals = {x['visible']['total'] for x in attempts}
    if len(totals) != 1:
        raise ValueError('Visible test suite changed within episode')
    return max(range(len(attempts)), key=lambda i: (attempts[i]['visible']['passed'], -i))


def run_episode(*, task, issue, initial_code, system_prompt, mode, generate,
                test_visible, namespace, replicate=0, budget=Budget()):
    """generate(messages, max_tokens, seed) returns text and actual token counts.

    No test-result-based early stop: all policies see the same maximum number
    of opportunities. Candidate selection uses visible results in both arms;
    only the feedback arm supplies those results to the subsequent generation.
    """
    if mode not in {'test_feedback', 'independent_attempts'}:
        raise ValueError('Undeclared repair condition')
    if any(type(x) is not str for x in (task, issue, initial_code, system_prompt)):
        raise ValueError('Only public task text belongs in an episode')
    initial = [{'role': 'system', 'content': system_prompt},
               {'role': 'user', 'content': issue + '\n\n```python\n' + initial_code + '\n```'}]
    history, attempts, spent = list(initial), [], 0
    for index in range(budget.attempts):
        allowance = min(budget.tokens_per_attempt, budget.total_new_tokens - spent)
        if allowance <= 0:
            break
        messages = history if mode == 'test_feedback' else initial
        # Copy through JSON so a backend cannot mutate stored prompt history.
        messages = json.loads(json.dumps(messages))
        seed = sampling_seed(namespace, task, replicate, index)
        output = generate(messages, allowance, seed)
        if set(output) != {'text', 'new_tokens', 'prompt_tokens', 'finish_reason'}:
            raise ValueError('Generation must retain exact usage')
        if (type(output['new_tokens']) is not int or not 0 <= output['new_tokens'] <= allowance
                or type(output['prompt_tokens']) is not int or output['prompt_tokens'] < 1
                or type(output['text']) is not str or type(output['finish_reason']) is not str):
            raise ValueError('Invalid generation or exceeded token allowance')
        feedback = visible_result(test_visible(output['text'], index))
        spent += output['new_tokens']
        record = {'attempt': index, 'seed': seed, **output, 'visible': feedback,
                  'cumulative_new_tokens': spent,
                  'prompt_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()}
        attempts.append(record)
        record['selected_at_this_budget'] = select_patch(attempts)
        if mode == 'test_feedback':
            history += [{'role': 'assistant', 'content': output['text']},
                        {'role': 'user', 'content': 'Visible test feedback:\n' + json.dumps(feedback)
                         + '\nReturn the full revised Python implementation.'}]
    return {'task': task, 'mode': mode, 'replicate': replicate, 'budget': asdict(budget),
            'attempts': attempts, 'selected': select_patch(attempts),
            'total_new_tokens': spent,
            'total_prompt_tokens': sum(x['prompt_tokens'] for x in attempts),
            'visible_test_calls': len(attempts)}
