"""Build the public data bundle behind the Paper Track writeup.

Files keep their repository-relative `runs/...` paths, so a checkout plus the
extracted bundle runs `research.paper_v1.reproduce --data <bundle>/data` unchanged.
Model weights, adapters, upstream KernelBench sources and compile workdirs are
excluded; adapter and archive hashes are recorded instead.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

R3_SCORES = 'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/local-analysis-verification.json'
R3_ARCHIVE = 'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/evidence-and-checkpoints.tgz'
FIXED = [
    'runs/reliability-v2/capability-v4/saved-programs.json',
    'runs/diagnostics-v5/terminal-audit-20261016/replay/metrics.jsonl',
    'runs/diagnostics-v5/terminal-audit-20261016/replay/REPORT.json',
    'runs/full-audit-20261018/replication-archive/selected/runs/replication-v1.1/confirmation/REPORT.json',
    'runs/replication-v1.1/plan.yaml',
    'runs/replication-v1.1/frozen.json',
    'runs/portability-v1-20261020/run-r3/metrics.jsonl',
    'runs/portability-v1-20261020/run-r3/REPORT.json',
    R3_SCORES,
    # Issue-style repair records behind Section 4.4.
    'runs/diagnostics-v5/terminal-audit-20261016/diagnostics/metrics.jsonl',
    'runs/diagnostics-v5/terminal-audit-20261016/diagnostics/REPORT.json',
    *[f'runs/diagnostics-v6-resume31b/terminal-audit-20261017/{m}/{f}'
      for m in ['gemma-4-12B-it', 'gemma-4-31B-it'] for f in ['episodes.jsonl', 'nodes.jsonl', 'REPORT.json']],
]
# Members of the verified R3 archive extracted for the training-dynamics analysis.
R3_MEMBERS = ['runs/replication-v1.1/training/*/rollouts.jsonl',
              'runs/replication-v1.1/confirmation/*/eval-step-*/scores.jsonl',
              'runs/replication-v1.1/confirmation/SFT-*/data/dataset/*',
              'runs/replication-v1.1/confirmation/SFT-*/data/candidates/scores.jsonl',
              'runs/replication-v1.1/confirmation/SFT-20261022/data/candidates/rollouts/*/candidate/worker.log']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extracted', type=Path, required=True, help='root holding extracted R3 archive members')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    data = args.out / 'data'
    data.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(Path(R3_SCORES).read_text())
    files = {p: Path(p) for p in FIXED}
    for v in receipt['policies'].values():  # R3 held-out scores plus the files load_complete checks
        files.update({v['path'] + '/' + n: Path(v['path'] + '/' + n) for n in ['scores.jsonl', 'request.json', 'complete.json']})
    for pattern in R3_MEMBERS:
        for p in sorted(args.extracted.glob(pattern)):
            files[str(p.relative_to(args.extracted))] = p
    manifest = []
    for rel, src in sorted(files.items()):
        dst = data / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        manifest.append({'path': rel, 'bytes': dst.stat().st_size, 'sha256': sha(dst),
                         'origin': 'verified R3 archive member' if src.is_relative_to(args.extracted) else 'local run record'})
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout)
    meta = {'built_utc': datetime.now(timezone.utc).isoformat(), 'git_head': head, 'working_tree_dirty': dirty,
            'r3_archive_sha256': '8cc3de5ecadc11e6ce5339dacba4bafc702e0f79d4892f57f927cb07b9f01e74',
            'files': len(manifest), 'bytes': sum(m['bytes'] for m in manifest), 'manifest': manifest}
    (args.out / 'MANIFEST.json').write_text(json.dumps(meta, indent=1) + '\n')
    (args.out / 'SHA256SUMS').write_text(''.join(f"{m['sha256']}  data/{m['path']}\n" for m in manifest))
    shutil.copyfile(Path(__file__).with_name('DATA_CARD.md'), args.out / 'README.md')
    print(args.out, meta['files'], 'files', meta['bytes'], 'bytes', 'dirty' if dirty else 'clean', head)


if __name__ == '__main__':
    main()
