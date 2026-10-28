"""K1–K4: full C0 on the approved target node; local smoke cannot produce PASS.

Requires the pinned stack, >=80 GB VRAM, two complete loaded snapshots, a 12B
vLLM answer, and pinned reference modules on CUDA. No candidate execution,
training, provisioning or implicit downloads.
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from env.tasks import validate_split, verified_catalog
from tools.gatelib import ROOT, enforce_failure_stop, finish_gate, timestamp
from tools.inventory import collect
from tools.weights import inspect_weights

STACK = {"torch": "2.13.0", "vllm": "0.30.0", "transformers": "5.17.0", "trl": "1.14.0", "peft": "0.21.0"}

def probe(mode, directory, model=None):
    stem = f"{mode}-{model.name if model else 'cuda'}"
    output, log = directory / f"{stem}.json", directory / f"{stem}.log"
    args = [sys.executable, "-m", "tools.c0_worker", mode, "--out", str(output)]
    if model is not None:
        args.extend(["--model", str(model)])
    with log.open("x") as f:
        child = subprocess.Popen(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise RuntimeError(f"{mode} timed out; see {log}") from None
    if code != 0 or not output.exists():
        error = json.loads(output.read_text()).get("error") if output.exists() else "No worker result"
        raise RuntimeError(f"{mode}: exit {code}: {error}; see {log}")
    result = json.loads(output.read_text())
    if result.get("status") != "PASS":
        raise RuntimeError(f"{mode} did not pass: {result}")
    return result

def prerequisites(inventory, models_dir):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use uv-managed Python 3.12")
    for name, version in STACK.items():
        observed = inventory["packages"].get(name)
        if observed is None or observed.split("+")[0] != version:
            raise RuntimeError(f"Stack mismatch: {name}={observed}, expected {version}")
    gpu = inventory["commands"]["gpu"]
    if gpu["returncode"] != 0:
        raise RuntimeError(f"nvidia-smi failed: {gpu}")
    rows = [r.split(",") for r in gpu["stdout"].splitlines()]
    # Hardware-reported capacities are slightly below nominal, e.g. an 80 GB H800.
    if len(rows) != 1 or float(rows[0][2]) < 79 * 1024:
        raise RuntimeError("Full C0 requires one >=80 GB target GPU; use local preflight on the dev box")
    metadata = [inspect_weights(models_dir / name) for name in ("gemma-4-12B-it", "gemma-4-E4B-it")]
    if [r["model_type"] for r in metadata] != ["gemma4_unified", "gemma4"]:
        raise RuntimeError("Unexpected model families for 12B / E4B snapshots")
    tasks = verified_catalog()
    validate_split(tasks)
    return {"weights": metadata, "task_count": len(tasks)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", type=Path, default=ROOT / "models", help="K1–K4: both explicit ModelScope snapshots")
    a = p.parse_args()
    enforce_failure_stop("C0")
    inventory = collect()
    directory = ROOT / "runs/c0" / timestamp()
    directory.mkdir(parents=True)
    checks = []
    try:
        result = prerequisites(inventory, a.models_dir.resolve())
        checks.append({"name": "stack_weights_tasks", "status": "PASS", "result": result})
        for name in ("gemma-4-12B-it", "gemma-4-E4B-it"):
            result = probe("weights", directory, a.models_dir.resolve() / name)
            checks.append({"name": f"load_{name}", "status": "PASS", "result": result})
        checks.append({"name": "vllm_12b_answer", "status": "PASS",
                       "result": probe("generate", directory, a.models_dir.resolve() / "gemma-4-12B-it")})
        checks.append({"name": "reference_cuda", "status": "PASS", "result": probe("reference", directory)})
    except Exception as e:
        checks.append({"name": "blocking_failure", "status": "FAIL", "error": f"{type(e).__name__}: {e}"})
        print(checks[-1]["error"], file=sys.stderr)
    raise SystemExit(finish_gate("C0", checks, inventory=inventory, artifacts=str(directory.relative_to(ROOT))))

if __name__ == "__main__":
    main()
