"""K1–K4/C0: no-GPU resource limits must block installs before a formal gate is attempted."""
from tools.check_node import readiness


def snapshot():
    return {
        "commands": {
            "gpu": {"returncode": 0, "stdout": "RTX PRO 6000, GPU-1, 97887, 0, 595.58.03, 12.0"},
            "driver": {"returncode": 0, "stdout": "CUDA Version: 13.0"},
            "nvcc": {"returncode": 0, "stdout": "release 13.0, V13.0.88"},
            "memory": {"returncode": 0, "stdout": f"Mem: {1024 * 2**30} 0 0"},
        },
        "tools": {"g++": "/usr/bin/g++"},
        "disk_bytes": {"free": 180 * 2**30},
        "cgroup_v2_limits": {"memory.max": str(128 * 2**30)},
    }


def failures(info):
    return {c["name"] for c in readiness(info) if c["status"] != "PASS"}


def test_target_node_is_ready():
    assert failures(snapshot()) == set()


def test_no_gpu_boot_cannot_use_host_ram_as_its_quota():
    info = snapshot()
    info["commands"]["gpu"] = {"returncode": None, "error": "Permission denied"}
    info["cgroup_v2_limits"]["memory.max"] = str(2 * 2**30)
    assert failures(info) == {"target_gpu", "container_memory"}


def test_unexpanded_disk_blocks_large_downloads():
    info = snapshot()
    info["disk_bytes"]["free"] = 50 * 2**30
    assert failures(info) == {"data_disk_headroom"}


def test_unknown_memory_quota_is_not_assumed_unlimited():
    info = snapshot()
    info["cgroup_v2_limits"] = {}
    assert failures(info) == {"container_memory"}


def test_driver_and_toolkit_are_independent():
    info = snapshot()
    info["commands"]["nvcc"] = {"returncode": None, "error": "not installed"}
    assert failures(info) == {"toolkit_cuda13"}


def test_prefetch_can_run_without_gpu_but_does_not_certify_c0():
    info = snapshot()
    info["commands"]["gpu"] = {"returncode": None, "error": "No GPU"}
    info["commands"]["driver"] = {"returncode": None, "error": "No GPU"}
    info["cgroup_v2_limits"]["memory.max"] = str(2 * 2**30)
    assert all(c["status"] == "PASS" for c in readiness(info, download_only=True))
    assert "target_gpu" in failures(info)
    assert "container_memory" in failures(info)


def test_prefetch_still_requires_expanded_disk():
    info = snapshot()
    info["disk_bytes"]["free"] = 50 * 2**30
    assert any(c["name"] == "data_disk_headroom" and c["status"] == "FAIL"
               for c in readiness(info, download_only=True))
