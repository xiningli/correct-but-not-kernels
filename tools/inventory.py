"""K1–K4: record machine, stack and network facts without changing the machine."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.gatelib import ROOT, timestamp, write_json

PACKAGES = ("torch", "triton", "vllm", "transformers", "trl", "peft", "safetensors", "modelscope", "pytest")


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"argv": args, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"argv": args, "returncode": None, "error": str(e)}


def reachable(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cuda-gemma4-c0/0.1"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return {"url": url, "reachable": True, "http_status": r.status}
    except urllib.error.HTTPError as e:
        return {"url": url, "reachable": True, "http_status": e.code, "error": str(e)}
    except (OSError, urllib.error.URLError) as e:
        return {"url": url, "reachable": False, "error": str(e)}


def collect(network=True):
    # C0/K1–K4: lscpu/free may describe the host; cgroups govern this container.
    limits = {}
    for name in ("cpu.max", "memory.max", "pids.max"):
        try:
            limits[name] = (Path("/sys/fs/cgroup") / name).read_text().strip()
        except OSError as e:
            limits[name] = {"error": str(e)}
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commands = {
        "gpu": ["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used,driver_version,compute_cap", "--format=csv,noheader,nounits"],
        "driver": ["nvidia-smi"], "nvcc": ["nvcc", "--version"], "cpu": ["lscpu"],
        "memory": ["free", "-b"], "kernel": ["uname", "-sr"],
    }
    with ThreadPoolExecutor(max_workers=6) as pool:
        probes = dict(zip(commands, pool.map(command, commands.values())))
        net = list(pool.map(reachable, ["https://huggingface.co", "https://modelscope.cn"])) if network else []
    disk = shutil.disk_usage(ROOT)
    return {"time_utc": timestamp(), "hostname": platform.node(), "python": sys.version,
            "python_executable": sys.executable, "logical_cpus": os.cpu_count(),
            "cpu_affinity_count": len(os.sched_getaffinity(0)), "cgroup_v2_limits": limits,
            "disk_bytes": dict(zip(("total", "used", "free"), disk)), "packages": packages,
            "tools": {name: shutil.which(name) for name in ("uv", "nvcc", "docker", "bwrap", "git", "g++", "ninja")},
            "commands": probes, "network": net}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, help="K1–K4: immutable inventory output path")
    p.add_argument("--offline", action="store_true", help="K1–K4: omit reachability probes")
    a = p.parse_args()
    record = collect(network=not a.offline)
    path = a.out or ROOT / "runs/inventory" / f"{timestamp()}.json"
    write_json(path, record)
    print(path)
    print(json.dumps({"packages": record["packages"], "gpu": record["commands"]["gpu"], "network": record["network"]}, indent=2))


if __name__ == "__main__":
    main()
