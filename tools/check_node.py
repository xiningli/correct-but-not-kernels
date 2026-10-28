"""K1–K4/C0: resource readiness before large downloads; not a research gate.

Capacity requirements are operational headroom: target GPU >=80 GB, RAM >=32 GiB,
and free data-disk space >=80 GiB. These do not change task/reward definitions.
"""
from __future__ import annotations

import json
import argparse
import re
from tools.gatelib import ROOT, timestamp, write_json
from tools.inventory import collect


def readiness(info, download_only=False):
    checks = []
    def add(name, passed, detail):
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    gpu = info["commands"]["gpu"]
    rows = [row.split(",") for row in gpu.get("stdout", "").splitlines() if row.strip()]
    try:
        gpu_ok = gpu.get("returncode") == 0 and len(rows) == 1 and float(rows[0][2]) >= 79 * 1024
    except (ValueError, IndexError):
        gpu_ok = False
    add("target_gpu", gpu_ok, gpu)
    driver = info["commands"]["driver"]
    cuda = re.search(r"CUDA Version:\s*(\d+)", driver.get("stdout", ""))
    add("driver_cuda13", driver.get("returncode") == 0 and cuda is not None and int(cuda[1]) >= 13, driver)
    nvcc = info["commands"]["nvcc"]
    add("toolkit_cuda13", nvcc.get("returncode") == 0 and bool(re.search(r"release 13\.", nvcc.get("stdout", ""))), nvcc)
    add("host_compiler", bool(info["tools"].get("g++")), info["tools"].get("g++"))
    free = info["disk_bytes"]["free"]
    add("data_disk_headroom", free >= 80 * 2**30, {"free_bytes": free, "minimum_bytes": 80 * 2**30})
    host_mem = re.search(r"Mem:\s*(\d+)", info["commands"]["memory"].get("stdout", ""))
    host_bytes = int(host_mem[1]) if host_mem else 0
    limit = info.get("cgroup_v2_limits", {}).get("memory.max")
    # Missing quota evidence is not permission to rely on host totals inside a container.
    if limit == "max":
        memory = host_bytes
    elif isinstance(limit, str) and limit.isdigit():
        memory = min(host_bytes, int(limit))
    else:
        memory = 0
    minimum_memory = (1 if download_only else 32) * 2**30
    add("container_memory", memory >= minimum_memory, {"effective_bytes": memory, "memory.max": limit,
                                                       "minimum_bytes": minimum_memory})
    if download_only:
        checks = [c for c in checks if c["name"] in {"data_disk_headroom", "container_memory"}]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true",
                        help="K1–K4: require disk and bounded CPU memory only; never certifies GPU readiness")
    args = parser.parse_args()
    info = collect(network=False)
    checks = readiness(info, download_only=args.download_only)
    ready = all(c["status"] == "PASS" for c in checks)
    path = ROOT / "runs/preflight" / f"node-readiness-{timestamp()}.json"
    write_json(path, {"scope": "download_readiness_not_C0" if args.download_only else "infrastructure_readiness_not_C0",
                     "ready": ready, "checks": checks, "inventory": info})
    for c in checks:
        print(f"{c['name']}: {c['status']} — {json.dumps(c['detail'])}")
    print(f"Readiness: {'READY' if ready else 'NOT READY'}; record: {path.relative_to(ROOT)}")
    raise SystemExit(0 if ready else 2)


if __name__ == "__main__":
    main()
