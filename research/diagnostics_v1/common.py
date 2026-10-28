"""D1/D2: immutable source identity and atomic evidence records."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'experiments/cuda-diagnostics-v5.yaml'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':'))


def object_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_seal():
    # Binding the unchanged dependencies avoids changing the in-flight v4 seal.
    # These two local-only post-hoc analysis files are not runtime dependencies.
    # Uploading them would change v4's glob-based seal while its replay is live.
    local_analysis = {'task_inference.py', 'test_task_inference.py'}
    from tools.replication import source_hashes
    files = [PLAN, *sorted((ROOT / 'research/diagnostics_v1').glob('*.py')),
             *sorted(p for p in (ROOT / 'research/reliability_v2').glob('*.py')
                     if p.name not in local_analysis)]
    return {**source_hashes(), **{str(p.relative_to(ROOT)): digest(p) for p in files}}
