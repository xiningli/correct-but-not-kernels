"""Artifact identity and atomic status for the separate portability study."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'experiments/cuda-portability-v1.json'
RECEIPT = ROOT / 'runs/autodl-westb-reboot-20261006/terminal-recovery-20261012/local-analysis-verification.json'
REMOTE = ROOT / 'runs/full-audit-20261018/reconstructed/strict_replay_v4/metrics.jsonl'
IMAGE = 'sha256:dacdca4039c98270005f9c543eeb638b547d09e484d072526e98b4d18f66074c'
PYTHON = '/mnt/fastdata/glomergence/.venv/bin/python'
VENV = '/mnt/fastdata/glomergence/.venv'
BASE_PYTHON = '/home/noname/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def append(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def code_files():
    files = list((ROOT / 'research/portability_v1').glob('*.py'))
    files += [ROOT / p for p in [
        'experiments/cuda-portability-v1.json', 'env/reward.py', 'env/correctness.py',
        'env/hacks/detectors.py', 'env/tasks.py', 'env/kernelbench.lock.json',
        'research/reliability_v2/tensor_checks.py',
        'research/reliability_v2/decomposition_worker.py',
        'research/reliability_v2/strict_contract.py',
        'tests/test_portability_v1.py']]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def load(prepared):
    prepared = Path(prepared)
    manifest = read(prepared / 'manifest.json')
    if manifest['code'] != code_files():
        raise ValueError('Portability code changed; prepare a new revision')
    for p, h in manifest['files'].items():
        if sha(prepared / p) != h:
            raise ValueError('Frozen portability input changed: ' + p)
    return manifest, read(prepared / 'plan.json')
