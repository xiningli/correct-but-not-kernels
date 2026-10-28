"""K1–K4: scaffold imports must neither load models nor execute generated code."""
import importlib
import pytest

@pytest.mark.parametrize("name", [
    "env.tasks", "env.sandbox", "env.run_kernel", "env.timing",
    "env.correctness", "env.reward", "env.hacks.detectors",
    "train.launch", "eval.heldout", "eval.taxonomy", "eval.report",
    "train.sft", "train.sft_data", "eval.replication", "eval.replication_report", "eval.pilot_audit", "eval.terminal_report",
    "tools.replication", "tools.replication_worker", "tools.run_replication", "tools.gate_r0", "tools.kl_probe",
    *[f"tools.gate_c{i}" for i in range(7)],
])
def test_import(name):
    importlib.import_module(name)
