"""D1: GPU boundary regression for the saved 31B case-001 repair (tanh via exp).

Executes the saved candidate unchanged in the portability-v1 container backend on
large-magnitude inputs outside the visible/withheld suites, and compares with an
FP64 formula. Development evidence only; the historical success score is unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from research.portability_v1.backend import run

SOURCE_SHA = '35c80af2bdc1d47c0a750b010f167e396928b974656383f7940387ef7b02cae6'
NODES = Path('runs/full-audit-20261018/reconstructed/iterative_31b_v6/nodes.jsonl')
PLAN = Path('runs/portability-v1-20261020/prepared-r3/plan.json')


def cases():
    g = torch.Generator().manual_seed(2026102501)
    out = []
    for magnitude in [1.0, 10.0, 80.0, 88.0, 89.0, 90.0, 100.0]:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(4, 33, generator=g)
            x[:, 0] = magnitude  # one large positive logit per row
            x[1:, 1] = -magnitude
            labels = torch.tensor([0, 1, 2, 3])
            out.append({'magnitude': magnitude, 'dtype': str(dtype), 'x': x.to(dtype), 'labels': labels})
    return out


def expected(x, labels):
    z = x.double().tanh()
    return (torch.logsumexp(z, -1) - z.gather(1, labels[:, None]).squeeze(1)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    node, = [r for r in map(json.loads, NODES.read_text().splitlines())
             if r['task'] == 'development/case-001' and r['component'] == 'correct']
    assert hashlib.sha256(node['source'].encode()).hexdigest() == SOURCE_SHA
    rows = cases()
    inputs = args.out / 'input'
    inputs.mkdir(parents=True, exist_ok=False)
    (inputs / 'program.py').write_text(node['source'])
    torch.save([(r['x'], r['labels']) for r in rows], inputs / 'cases.pt')
    plan = json.loads(PLAN.read_text())
    execution = run(Path(__file__).with_name('tanh_boundary_runner.py'), inputs, args.out / 'container',
                    plan, seconds=600)
    assert execution['returncode'] == 0, execution
    got = torch.load(Path(execution['work']) / 'outputs.pt')
    results = []
    for r, y in zip(rows, got):
        want = expected(r['x'], r['labels'])
        finite = torch.isfinite(y)
        ok = bool(finite.all() and torch.allclose(y, want, rtol=1e-3, atol=1e-4))
        results.append({'magnitude': r['magnitude'], 'dtype': r['dtype'], 'pass': ok,
                        'nonfinite_rows': int((~finite).sum()),
                        'max_abs_error_finite': float((y[finite] - want[finite]).abs().max()) if finite.any() else None})
    receipt = {'candidate_sha256': SOURCE_SHA, 'execution': {k: execution[k] for k in ['returncode', 'reason', 'seconds']},
               'runner': json.loads((Path(execution['work']) / 'runner.json').read_text()),
               'tolerance': {'rtol': 1e-3, 'atol': 1e-4}, 'results': results,
               'scope': 'development boundary regression; not part of the sealed v6 suites; historical score unchanged'}
    (args.out / 'tanh-boundary.json').write_text(json.dumps(receipt, indent=1) + '\n')
    for x in results:
        print(x)


if __name__ == '__main__':
    main()
