"""K1–K3: extract source without executing it. Scoring awaits C1–C3.

Adapted from glomergence/env/reward.py, source HEAD e602f63207fe4e67eeda4dff7b53d49e92cc318b.
Keeps Gemma reasoning removal; rejects truncated thought channels rather than
mistaking a speculative code block for a final answer. No non-Gemma sampling.
"""
import ast
import re

# K1/K3: all containment/runtime failures retain a distinct logged component,
# but share the preregistered Arm C execution penalty. No reward is inferred
# from worker-provided correctness or speed claims.
FAILURE_COMPONENTS = frozenset({'format', 'compile', 'crash', 'timeout',
    'sandbox_blocked', 'process_limit', 'gpu_oom', 'gpu_memory_limit',
    'cpu_memory_limit', 'cpu_limit', 'disk_limit'})

def failure_reward(component: str, arm: str = 'C') -> float:
    if component not in FAILURE_COMPONENTS or arm not in ('C', 'S'):
        raise ValueError('Unknown failure component or reward arm')
    return -0.1 if arm == 'C' else 0.0

_THINK = re.compile(r"<\|channel>thought.*?<channel\|>", re.S)
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def extract_program(completion: str) -> str | None:
    text = _THINK.sub("", completion)
    if "<|channel>thought" in text:
        return None
    blocks = _FENCE.findall(text)
    if not blocks:
        return None
    src = blocks[-1]
    try:
        tree = ast.parse(src, filename="candidate.py")
    except (SyntaxError, ValueError):
        return None
    if not any(isinstance(n, ast.ClassDef) and n.name == "ModelNew" for n in tree.body):
        return None
    return src


def score_result(result, arm='C'):
    """K1/K2: bounded speed term s/(1+s), with diminishing returns, in (0,1).

    It is monotone, equals 0.5 at eager parity and cannot let a timing outlier
    dominate a GRPO group. Correctness/failure/detector components stay separate.
    """
    if arm not in ('C','S'):
        raise ValueError('Unknown reward arm')
    if result.component == 'harness_error':
        return None
    if result.component in FAILURE_COMPONENTS:
        return failure_reward(result.component, arm)
    if not result.correct:
        return 0.0
    if arm == 'C':
        return 1.0
    if result.speedup is None or result.speedup <= 0:
        raise ValueError('Correct sample needs a measured speedup')
    return result.speedup/(1+result.speedup)

def score_completion(completion, task=None, directory=None, *, seed=None, trials=1, split='train'):
    if task is None or directory is None or seed is None:
        raise NotImplementedError('Explicit gated task/run/seed context required; no code executed')
    from env.run_kernel import RunResult, evaluate_task
    source = extract_program(completion)
    return evaluate_task(source, task, directory, seed=seed, trials=trials, split=split) if source else RunResult('format')
