"""K1–K4: local preparation only. Never passes a research gate or downloads models.

--gpu-python (K1–K4) optionally reuses an existing GPU interpreter read-only for
official-reference smoke. It installs nothing and executes no policy programs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from env.tasks import validate_split, verified_catalog
from tools.gatelib import ROOT, git_state, timestamp, write_json
from tools.inventory import collect
from tools.weights import inspect_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-python", type=Path, help="K1–K4: optional existing GPU interpreter for trusted smoke")
    a = p.parse_args()
    directory = ROOT / "runs/preflight" / timestamp()
    directory.mkdir(parents=True)
    record = {"scope": "local_preparation_not_C0", "time_utc": timestamp(), "git": git_state(), "checks": []}
    write_json(directory / "inventory.json", collect())
    try:
        tasks = verified_catalog()
        train, heldout = validate_split(tasks)
        record["tasks"] = {"total": len(tasks), "train_level1": len(train),
                           "heldout_level1": sum(x.startswith("level1/") for x in heldout),
                           "heldout_level2": sum(x.startswith("level2/") for x in heldout)}
        record["checks"].append({"name": "locked_tasks_and_split", "status": "PASS"})
        subprocess.run(["bash", "-n", "scripts/setup_node.sh", "scripts/gpu_env.sh"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={directory / 'tests.xml'}"], cwd=ROOT, check=True)
        record["checks"].append({"name": "tests_and_shell_syntax", "status": "PASS", "record": "tests.xml"})
        if a.gpu_python:
            # Preserve the venv symlink: resolve() runs base Python without its site-packages.
            gpu_python = a.gpu_python.absolute()
            subprocess.run([str(ROOT / "scripts/gpu_env.sh"), str(gpu_python), "-m", "tools.c0_worker", "reference",
                            "--out", str(directory / "reference.json")], cwd=ROOT, check=True, timeout=120)
            subprocess.run([str(ROOT / "scripts/gpu_env.sh"), str(gpu_python), "-m", "tools.inventory", "--offline",
                            "--out", str(directory / "gpu-inventory.json")], cwd=ROOT, check=True, timeout=60,
                           stdout=subprocess.DEVNULL)
            record["checks"].append({"name": "trusted_reference_cuda", "status": "PASS", "record": "reference.json",
                                     "gpu_python": str(gpu_python)})
        else:
            record["checks"].append({"name": "trusted_reference_cuda", "status": "not run"})
        record["local_checks_passed"] = True
    except Exception as e:
        record["checks"].append({"name": "preparation_failure", "status": "FAIL", "error": f"{type(e).__name__}: {e}"})
        record["local_checks_passed"] = False
    record["weights"] = {}
    for model in ("gemma-4-12B-it", "gemma-4-E4B-it"):
        try:
            record["weights"][model] = inspect_weights(ROOT / "models" / model)
        except (ValueError, OSError) as e:
            record["weights"][model] = {"status": "unavailable", "reason": str(e)}
    record["gates"] = {f"C{i}": "not run by preflight" for i in range(7)}
    write_json(directory / "preflight.json", record)
    print(f"Local preparation: {'PASS' if record['local_checks_passed'] else 'FAIL'} (not a gate result)")
    print(f"Record: {directory.relative_to(ROOT) / 'preflight.json'}")
    raise SystemExit(0 if record["local_checks_passed"] else 1)


if __name__ == "__main__":
    main()
