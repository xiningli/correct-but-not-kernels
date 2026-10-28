"""K1–K4/C0: prepare a committed source bundle and pinned KernelBench checkout for offline transfer.

Only code, Git history and the pinned public benchmark are included. Run data,
credentials, environments and model weights never enter the deployment artifacts.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

from env.tasks import DEFAULT_BENCH, LOCK, verified_catalog
from tools.gatelib import ROOT, timestamp, write_json


def main():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("Commit the reviewed source first; refusing to package a dirty checkout")
    verified_catalog()
    if subprocess.check_output(["git", "-C", str(DEFAULT_BENCH), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Refusing to package a changed KernelBench checkout")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out = ROOT / "runs/deploy" / f"{timestamp()}-{sha[:8]}"
    out.mkdir(parents=True)
    bundle = out / "source.bundle"
    subprocess.run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=ROOT, check=True)
    benchmark = out / "kernelbench.tar.gz"
    def include(member):
        if "__pycache__" in Path(member.name).parts or member.name.endswith(".pyc"):
            return None
        # Root extraction must not recreate the developer workstation's UID/GID.
        member.uid = member.gid = 0
        member.uname = member.gname = "root"
        return member
    # Keep upstream .git so content checks can verify the actual revision offline.
    with tarfile.open(benchmark, "w:gz") as tf:
        tf.add(DEFAULT_BENCH, arcname="third_party/KernelBench", filter=include)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (bundle, benchmark)}
    write_json(out / "manifest.json", {"source_sha": sha, "kernelbench_revision": json.loads(LOCK.read_text())["revision"],
                                      "sha256": files, "excludes": ["runs", "models", "credentials", ".venv", ".cache"]})
    with (out / "SHA256SUMS").open("x") as f:
        for name, digest in files.items():
            f.write(f"{digest}  {name}\n")
        f.write(f"{hashlib.sha256((out / 'manifest.json').read_bytes()).hexdigest()}  manifest.json\n")
    print(out)


if __name__ == "__main__":
    main()
