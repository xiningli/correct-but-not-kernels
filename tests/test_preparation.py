"""K1–K4: prevent false C0 passes, split leakage and accidental candidate execution."""
import json
import struct
from pathlib import Path

import pytest

from env.reward import extract_program, score_completion
from env.tasks import ROOT, validate_split, verified_catalog
from tools.gatelib import enforce_failure_stop
from tools.weights import inspect_weights


def snapshot(tmp_path, payload=True):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    if payload:
        (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))
    return tmp_path


def test_empty_download_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Cannot find any model weights"):
        inspect_weights(snapshot(tmp_path, payload=False))


def test_truncated_payload_is_rejected(tmp_path):
    snapshot(tmp_path)
    path = tmp_path / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated tensor payload"):
        inspect_weights(tmp_path)


def test_missing_shard_is_rejected(tmp_path):
    snapshot(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="missing shards"):
        inspect_weights(tmp_path)


def test_structural_check_is_not_a_load(tmp_path):
    result = inspect_weights(snapshot(tmp_path))
    assert result["tensor_count"] == 1
    assert result["validation"] == "structure_only_not_a_model_load"


def test_index_cannot_hide_an_extra_tensor(tmp_path):
    snapshot(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    with pytest.raises(ValueError, match="Index does not match"):
        inspect_weights(tmp_path)


def test_locked_split_keeps_all_level2_heldout():
    tasks = verified_catalog()
    train, heldout = validate_split(tasks)
    assert not train & heldout
    assert {t.task_id for t in tasks if t.level == 2} <= heldout


def test_split_leakage_is_rejected(tmp_path):
    split = json.loads((ROOT / "env/heldout.json").read_text())
    split["train_level1"].append(split["heldout_level1"][0])
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="overlapping"):
        validate_split(verified_catalog(), path)


def test_thought_code_is_not_the_answer():
    thought = "<|channel>thought\n```python\nclass ModelNew: pass\n```<channel|>"
    assert extract_program(thought + "No final program") is None
    assert extract_program(thought[:-10]) is None
    assert extract_program(thought + "```python\nclass ModelNew: pass\n```") == "class ModelNew: pass\n"


def test_extraction_never_executes(tmp_path):
    marker = tmp_path / "should-not-exist"
    code = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nclass ModelNew: pass\n"
    assert extract_program(f"```python\n{code}```") == code
    assert not marker.exists()
    with pytest.raises(NotImplementedError):
        score_completion(code)
    assert not marker.exists()


def test_two_failed_gates_require_operator(tmp_path):
    directory = tmp_path / "runs/gates"
    directory.mkdir(parents=True)
    for i in range(2):
        (directory / f"c0-{i}.json").write_text(json.dumps({"status": "FAIL"}))
    with pytest.raises(RuntimeError, match="failed twice"):
        enforce_failure_stop("C0", tmp_path)
