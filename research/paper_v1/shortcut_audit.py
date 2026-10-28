"""K1/K3: decompose the saved Level-2 correctness gain by observed custom-kernel use.

Read-only analysis of the completed v4 same-program replay. The original endpoint
(`original.correct`) is unchanged; the replay's profiler events only classify
whether a program's forward pass actually launched a kernel the program declared.
Classes: no declared kernel; declared but never launched; launched (possibly
alongside ATen compute). Static reference checks are a heuristic description.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

REPLAY = Path('runs/diagnostics-v5/terminal-audit-20261016/replay/metrics.jsonl')
PROGRAMS = Path('runs/reliability-v2/capability-v4/saved-programs.json')
CLASSES = ['no_declared_kernel', 'declared_not_launched', 'launched']
ARMS = ['base', 'C', 'SFT']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launched(names, kernels):
    return any(re.search(r'(?<![A-Za-z0-9_])' + re.escape(n) + r'(?![A-Za-z0-9_])', k)
               for n in names for k in kernels)


def _names(node):
    used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    return used | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def referenced_by_model(source, names):
    """True when ModelNew reaches a declared kernel name through module-level definitions."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    defs = {}  # in source order, so a later definition or `ModelNew = X` alias wins
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defs[n.name] = n
        elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id in defs:
            defs.update({t.id: defs[n.value.id] for t in n.targets if isinstance(t, ast.Name)})
    if 'ModelNew' not in defs:
        return None
    seen, todo = set(), ['ModelNew']
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        used = _names(defs[name])
        if used & set(names) - {name}:
            return True
        todo.extend(u for u in used if u in defs)
    return False


def classify(row):
    res = row['result']
    names = res.get('declared_kernel_names') or []
    obs = (res.get('runtime') or {}).get('observations') or []
    kernels = [k for o in obs for k in o.get('cuda_kernels', [])]
    if not names:
        return 'no_declared_kernel'
    return 'launched' if launched(names, kernels) else 'declared_not_launched'


def arm(policy):
    return policy.split('-')[0]


def task_rates(rows):
    """Per arm, per task: mean over attempts (seeds pooled within arm) of each indicator."""
    acc = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        c = acc[arm(r['policy'])][r['task']]
        c['n'] += 1
        if r['original']['correct']:
            c['correct'] += 1
            c['correct_' + r['class']] += 1
        c['all_' + r['class']] += 1
    keys = ['correct'] + ['correct_' + k for k in CLASSES] + ['all_' + k for k in CLASSES]
    return {a: {t: {k: c[k] / c['n'] for k in keys} for t, c in tasks.items()} for a, tasks in acc.items()}, keys


def paired(rates, keys, a, b, draws, seed):
    tasks = sorted(rates[a])
    assert tasks == sorted(rates[b])
    diff = {k: [rates[a][t][k] - rates[b][t][k] for t in tasks] for k in keys}
    rng = random.Random(seed)
    boot = {k: [] for k in keys}
    for _ in range(draws):
        idx = [rng.randrange(len(tasks)) for _ in tasks]
        for k in keys:
            boot[k].append(sum(diff[k][i] for i in idx) / len(idx))
    lo, hi = 0.025, 0.975
    out = {}
    for k in keys:
        s = sorted(boot[k])
        out[k] = {'estimate': sum(diff[k]) / len(tasks),
                  'ci95': [s[int(lo * draws)], s[int(hi * draws) - 1]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--root', type=Path, default=Path('.'), help='checkout or release-data root holding runs/')
    ap.add_argument('--draws', type=int, default=10000)
    args = ap.parse_args()
    rows = [json.loads(s) for s in (args.root / REPLAY).read_text().splitlines()]
    sources = {(r['policy'], r['task'], r['sample']): r['source']
               for r in json.loads((args.root / PROGRAMS).read_text())['rows']}
    assert len(rows) == len(sources) == 5600
    for r in rows:
        r['class'] = classify(r)
    counts = {}
    for a in ARMS:
        members = [r for r in rows if arm(r['policy']) == a]
        correct = [r for r in members if r['original']['correct']]
        dnl = [r for r in correct if r['class'] == 'declared_not_launched']
        refs = Counter(referenced_by_model(sources[(r['policy'], r['task'], r['sample'])],
                                           r['result']['declared_kernel_names']) for r in dnl)
        defined_only = sum(all(len(re.findall(r'\b' + re.escape(n) + r'\b', sources[(r['policy'], r['task'], r['sample'])])) == 1
                               for n in r['result']['declared_kernel_names']) for r in dnl)
        counts[a] = {
            'correct_declared_not_launched_name_appears_only_at_definition': defined_only,
            'attempts': len(members), 'correct': len(correct),
            'all_by_class': dict(Counter(r['class'] for r in members)),
            'correct_by_class': dict(Counter(r['class'] for r in correct)),
            'correct_without_runtime_observation': sum(
                not (r['result'].get('runtime') or {}).get('observations') for r in correct),
            'correct_declared_not_launched_referenced_by_ModelNew': {str(k): v for k, v in refs.items()},
            'strict_success': sum(r['result']['correct'] is True for r in members),
            'contract_reasons': dict(Counter(ch['reason'] for r in members
                                             for ch in (r['result'].get('contract_checks') or [])[:1])),
        }
    rates, keys = task_rates(rows)
    effects = {f'{a}-{b}': paired(rates, keys, a, b, args.draws, 20261025)
               for a, b in [('C', 'base'), ('SFT', 'base'), ('C', 'SFT')]}
    receipt = {
        'inputs': {str(REPLAY): sha(args.root / REPLAY), str(PROGRAMS): sha(args.root / PROGRAMS)},
        'tasks': len(rates['base']), 'bootstrap': {'draws': args.draws, 'unit': 'task', 'seed': 20261025,
                                                   'interval': 'percentile 95%, unadjusted'},
        'counts': counts, 'paired_task_effects': effects,
        'limits': [
            'Retrospective analysis of saved programs on tasks the researcher had already observed.',
            '"launched" means a profiler kernel name matched a declared Triton/CUDA kernel; ATen compute may also run.',
            'Seeds are pooled within arm per task; intervals are conditional on these checkpoints.',
            'ModelNew-reference check is static and name-based; CUDA wrappers may hide kernel names.',
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'shortcut-audit.json').write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    for a in ARMS:
        c = counts[a]
        print(a, c['correct'], '/', c['attempts'], c['correct_by_class'], 'strict', c['strict_success'])
    for name, e in effects.items():
        print(name, {k: (round(100 * v['estimate'], 2), [round(100 * x, 2) for x in v['ci95']])
                     for k, v in e.items() if k.startswith('correct')})


if __name__ == '__main__':
    main()
