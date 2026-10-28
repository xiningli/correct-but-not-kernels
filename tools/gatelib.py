"""K1–K4: immutable gate evidence and the twice-failed stop rule (kickoff §6)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def init_run(directory: Path, config: dict, seed: int):
    """K1–K4: provenance before executing any experimental workload."""
    import yaml
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    (directory/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (directory/'git_sha').write_text(json.dumps(git_state())+'\n')
    (directory/'seed').write_text(str(seed)+'\n')
    versions = {}
    for package in ['torch','triton','vllm','transformers','trl','peft','safetensors']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(directory/'versions.json', versions)
    (directory/'metrics.jsonl').touch(exist_ok=False)


def git_state():
    def git(*args):
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"sha": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def enforce_failure_stop(gate: str, root: Path = ROOT):
    consecutive = 0
    for path in sorted((root / "runs/gates").glob(f"{gate.lower()}-*.json"), reverse=True):
        record = json.loads(path.read_text())
        if record["status"] == "PASS":
            break
        consecutive += record["status"] == "FAIL"
    if consecutive >= 2:
        raise RuntimeError(f"{gate} failed twice. Stop for the operator per kickoff §6; preserve evidence.")


def require_gate(gate: str, root: Path = ROOT):
    """K1–K4: a recorded PASS, not a comment, unlocks the next workload."""
    records = sorted((root / 'runs/gates').glob(f'{gate.lower()}-*.json'))
    if not records or json.loads(records[-1].read_text())['status'] != 'PASS':
        raise RuntimeError(f'{gate} PASS required before this workload')
    return str(records[-1].relative_to(root))


def finish_gate(gate: str, checks: list[dict], **details):
    passed = bool(checks) and all(c["status"] == "PASS" for c in checks)
    record = {"gate": gate, "status": "PASS" if passed else "FAIL",
              "time_utc": timestamp(), "git": git_state(), "checks": checks, **details}
    path = ROOT / "runs/gates" / f"{gate.lower()}-{record['time_utc']}-{uuid4().hex[:8]}.json"
    write_json(path, record)
    if passed:
        # Resolve kickoff's conflict: commit a byte-identical copy, never raw runs/.
        dest = ROOT / "evidence/gates" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as f:
            f.write(path.read_bytes())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"Evidence for commit: {dest.relative_to(ROOT)} sha256={digest}")
    print(f"{gate}: {record['status']} — {path.relative_to(ROOT)}")
    return 0 if passed else 1


def deferred_gate(gate: str, prerequisite: str):
    enforce_failure_stop(gate)
    return finish_gate(gate, [{"name": "implementation", "status": "FAIL",
                               "error": f"Not implemented; requires {prerequisite}. No workload executed."}])
