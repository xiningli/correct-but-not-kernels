"""K1/K3: verify storage planning and NaN/persistence distinction without GPU candidates."""
from pathlib import Path
from types import SimpleNamespace
import pytest

from research.reliability_v2.storage import require_free, storage_budget


def test_large_equal_cases_fit_without_duplicate_inputs():
    gib = 1 << 30
    result = storage_budget([8 * gib] * 4, [8 * gib] * 4)
    assert result['required_before_staging_bytes'] == 48 * gib
    assert result['required_after_staging_bytes'] == 16 * gib


def test_expanding_outputs_are_included_in_peak():
    result = storage_budget([1, 1], [10, 10], headroom=3)
    assert result['streaming_peak_bytes'] == 20
    assert result['required_after_staging_bytes'] == 31


def test_no_space_fails_before_worker(monkeypatch, tmp_path):
    monkeypatch.setattr('research.reliability_v2.storage.shutil.disk_usage', lambda p: SimpleNamespace(free=99))
    with pytest.raises(RuntimeError, match='Insufficient evaluator storage'):
        require_free(tmp_path, 100)


@pytest.mark.parametrize('inputs,outputs', [([], []), ([1], [1, 2]), ([-1], [2])])
def test_bad_storage_manifest_rejected(inputs, outputs):
    with pytest.raises(ValueError):
        storage_budget(inputs, outputs)


def test_nan_persistence_and_numerical_correctness_are_distinct():
    torch = pytest.importorskip('torch')
    from research.reliability_v2.tensor_checks import same_values
    from env.correctness import compare
    a = torch.tensor([float('nan'), 1.0, float('inf')])
    b = a.clone()
    assert same_values(a, b)
    assert compare(a, b)['component'] == 'nonfinite'
    b[1] = 2.0
    assert not same_values(a, b)
    assert not same_values(a, a.double())
