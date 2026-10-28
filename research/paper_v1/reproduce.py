"""Recompute every Paper Track number from the release bundle and compare with receipts.

CPU only. Run from a checkout with PyTorch installed:
  python -m research.paper_v1.reproduce --data release-data/data --out <new dir>
The bundle must sit inside the checkout: the frozen R3 loader rejects outside paths.
Each check must reproduce the committed receipt exactly (parsed JSON equality).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from eval.replication_report import report
from tools.replication import load_plan

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'evidence/paper-shortcut-audit-20261025'
R3_SCORES = 'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/local-analysis-verification.json'
R3_REPORT = 'runs/full-audit-20261018/replication-archive/selected/runs/replication-v1.1/confirmation/REPORT.json'


def r3(data):
    """Main table: rebuild the R3 confirmation report from the ten per-policy score files."""
    old = json.loads((data / R3_REPORT).read_text())
    receipt = json.loads((data / R3_SCORES).read_text())
    # The frozen report loader accepts only checkout-relative paths.
    rel = data.relative_to(ROOT)
    manifest = [{'label': label, 'arm': cell['arm'], 'training_seed': cell['training_seed'],
                 'scores': str(rel / receipt['policies'][label]['path'] / 'scores.jsonl')}
                for label, cell in old['policies'].items()]
    # The frozen-source check is skipped: it compares the whole checkout, not these data files.
    return report(load_plan(data / 'runs/replication-v1.1/plan.yaml'), manifest) == old


def module(name, *extra):
    subprocess.run([sys.executable, '-W', 'ignore', '-m', name, *extra], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    if not data.is_relative_to(ROOT):
        sys.exit('Extract the bundle inside the checkout (e.g. ./release-data); the frozen R3 loader requires it.')
    out.mkdir(parents=True, exist_ok=False)
    checks = {'r3_confirmation_report': r3(data)}
    module('research.paper_v1.shortcut_audit', '--root', str(data), '--out', str(out))
    module('research.paper_v1.training_shortcut_audit', '--root', str(data), '--extracted', str(data),
           '--logs', str(data), '--out', str(out))
    for name in ['shortcut-audit.json', 'training-shortcut-audit.json']:
        checks[name] = json.loads((out / name).read_text()) == json.loads((EVIDENCE / name).read_text())
    (out / 'reproduce.json').write_text(json.dumps(checks, indent=1) + '\n')
    for k, v in checks.items():
        print('PASS' if v else 'FAIL', k)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
