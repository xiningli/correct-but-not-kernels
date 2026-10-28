"""K1/K3: trace decorative kernels back into training data, statically.

Training rollouts and SFT data have no runtime kernel traces, so this uses a
source-only classifier. It is first validated against the v4 replay, where
profiler traces exist, then applied to the R3 GRPO rollouts (step 0 is the base
policy), the base-policy SFT candidates, the accepted SFT datasets and the intermediate checkpoint evaluations.
Inputs are files extracted from the verified R3 archive; nothing is re-executed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from env.reward import extract_program
from research.paper_v1.shortcut_audit import PROGRAMS, REPLAY, classify, referenced_by_model, sha
from research.reliability_v2.strict_contract import declared_kernel_names

STATIC = ['no_declared_kernel', 'definition_only', 'reachable', 'unreachable_other']
R3 = 'runs/replication-v1.1'


def static_class(source):
    if source is None:
        return 'not_extractable'
    try:
        names = declared_kernel_names(source)
    except SyntaxError:
        return 'not_extractable'
    if not names:
        return 'no_declared_kernel'
    if all(len(re.findall(r'\b' + re.escape(n) + r'\b', source)) == 1 for n in names):
        return 'definition_only'
    return 'reachable' if referenced_by_model(source, names) else 'unreachable_other'


def validate(root):
    """Static class versus profiler-observed class on the 5,600 replayed programs."""
    rows = [json.loads(s) for s in (root / REPLAY).read_text().splitlines()]
    sources = {(r['policy'], r['task'], r['sample']): r['source']
               for r in json.loads((root / PROGRAMS).read_text())['rows']}
    table = defaultdict(Counter)
    for r in rows:
        if (r['result'].get('runtime') or {}).get('observations'):
            table[static_class(sources[(r['policy'], r['task'], r['sample'])])][classify(r)] += 1
    return {k: dict(v) for k, v in table.items()}


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def summarize(programs):
    """programs: iterable of (source, correct)."""
    c = Counter()
    for source, correct in programs:
        k = static_class(source)
        c['n'] += 1
        c[k] += 1
        if correct:
            c['correct'] += 1
            c['correct_' + k] += 1
    return dict(c)


def rollouts(root):
    out = {}
    for path in sorted((root / R3 / 'training').glob('*/rollouts.jsonl')):
        rows = jsonl(path)
        by_step = defaultdict(list)
        for r in rows:
            bucket = 'step_0' if int(r['step']) == 0 else f'steps_{(int(r["step"]) - 1) // 10 * 10 + 1}-{(int(r["step"]) - 1) // 10 * 10 + 10}'
            by_step[bucket].append((extract_program(r['text']), r['result']['correct'] is True))
        out[path.parent.name] = {'sha256': sha(path), 'rows': len(rows),
                                 'buckets': {b: summarize(v) for b, v in sorted(by_step.items())}}
    return out


def sft(root):
    out = {}
    for path in sorted((root / R3 / 'confirmation').glob('SFT-*/data/dataset/train.jsonl')):
        rows = jsonl(path)
        assert all(r['result']['correct'] is True for r in rows)
        out[path.parts[-4]] = {'sha256': sha(path), **summarize((extract_program(r['text']), True) for r in rows)}
    return out


def base_candidates(root):
    """All 320 base-policy LayerNorm samples per SFT seed, accepted or not."""
    out = {}
    for path in sorted((root / R3 / 'confirmation').glob('SFT-*/data/candidates/scores.jsonl')):
        rows = jsonl(path)
        out[path.parts[-4]] = {'sha256': sha(path), **summarize(
            (extract_program(r['text']), r['result']['correct'] is True) for r in rows)}
    return out


def crash_causes(root, logs):
    """Worker-log crash causes for one SFT seed's base candidates (logs extracted separately)."""
    path = root / R3 / 'confirmation/SFT-20261022/data/candidates/scores.jsonl'
    counts = Counter()
    for r in jsonl(path):
        if r['result']['component'] != 'crash':
            continue
        log = logs / r['program_directory'] / 'candidate/worker.log'
        text = log.read_text() if log.exists() else None
        cause = ('no_log' if text is None else 'nonexistent_triton_attribute'
                 if re.search(r"module 'triton(\.language)?' has no attribute", text) else 'other')
        counts[static_class(extract_program(r['text'])) + '/' + cause] += 1
    return {'scores_sha256': sha(path), 'counts': dict(counts)}


def checkpoints(root):
    out = {}
    for path in sorted((root / R3 / 'confirmation').glob('*/eval-step-*/scores.jsonl')):
        rows = jsonl(path)
        out[f'{path.parts[-3]}/{path.parts[-2]}'] = {
            'sha256': sha(path),
            **summarize((extract_program(r['text']), r['result']['correct'] is True) for r in rows)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extracted', type=Path, required=True,
                    help='directory holding runs/replication-v1.1/... members of the verified R3 archive')
    ap.add_argument('--root', type=Path, default=Path('.'), help='checkout or release-data root holding runs/')
    ap.add_argument('--logs', type=Path, help='directory holding SFT-20261022 candidate worker logs')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    receipt = {
        'archive_sha256': '8cc3de5ecadc11e6ce5339dacba4bafc702e0f79d4892f57f927cb07b9f01e74',
        'static_vs_runtime_on_v4_replay': validate(args.root),
        'grpo_rollouts_layernorm': rollouts(args.extracted),
        'base_candidates_layernorm': base_candidates(args.extracted),
        'sft_datasets_layernorm': sft(args.extracted),
        'checkpoint_evaluations': checkpoints(args.extracted),
        'base_crash_causes_sft_20261022': crash_causes(args.extracted, args.logs) if args.logs else None,
        'limits': [
            'Static source classes only; training programs were not re-executed under the kernel observer.',
            'definition_only = every declared kernel name occurs once in the file; validated against profiler classes.',
            'Intermediate checkpoint evaluations are 1 program per task (120); step 40 has 8 per task.',
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'training-shortcut-audit.json').write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    print(json.dumps(receipt['static_vs_runtime_on_v4_replay'], indent=1))


if __name__ == '__main__':
    main()
