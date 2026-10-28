import copy
import json

import pytest
import yaml

from research.diagnostics_resume_v6 import prepare as recovery

ARCHIVE = recovery.ROOT / 'runs/diagnostics-v6/recovery/2026-10-16T222119.856430_0000/evidence.tgz'


def test_only_memory_and_unstarted_arm_change():
    parent = yaml.safe_load(recovery.original.PLAN.read_text())
    protocol = yaml.safe_load(recovery.PROTOCOL.read_text())
    actual = recovery.resumed_plan(parent, protocol)
    expected = copy.deepcopy(parent)
    expected.update(id='cuda-diagnostics-v6-resume31b', models=['gemma-4-31B-it'])
    expected['inference']['gpu_memory_utilization'] = 0.90
    expected['development'].update(episodes=12, logical_attempt_cap=48, unique_generation_cap=40)
    assert actual == expected
    assert parent == yaml.safe_load(recovery.original.PLAN.read_text())
    parent['sampling']['max_tokens'] = 4096
    with pytest.raises(ValueError, match='Parent protocol changed'):
        recovery.resumed_plan(parent, protocol)


def test_wrong_archive_cannot_create_preparation(tmp_path):
    wrong = tmp_path / 'wrong.tgz'
    wrong.write_bytes(b'not the frozen evidence')
    out = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='Parent archive identity'):
        recovery.prepare(wrong, out)
    assert not out.exists()


@pytest.mark.skipif(not ARCHIVE.exists(), reason='Verified evidence archive is local, not in git')
def test_real_archive_preparation_and_rerun_refusal(tmp_path):
    out = tmp_path / 'prepared'
    receipt = recovery.prepare(ARCHIVE, out)
    assert receipt['source_sha256'] == recovery.original.source_seal()
    assert json.loads((out / 'continuation.json').read_text())['parent_31B_candidates'] == 0
    for name, sha in receipt['files'].items():
        assert recovery.original.digest(out / name) == sha
    with pytest.raises(FileExistsError):
        recovery.prepare(ARCHIVE, out)


def test_supervisor_overrides_only_new_process_globals(monkeypatch):
    from research.diagnostics_resume_v6 import supervise
    old_base = supervise.original.BASE
    old_loader = supervise.original.load_prepared
    old_seal = supervise.original.source_seal
    seen = []
    def mock_main():
        seen.append((supervise.original.BASE, supervise.original.load_prepared, supervise.original.source_seal))
    monkeypatch.setattr(supervise.original, 'main', mock_main)
    try:
        supervise.main()
        assert seen == [(recovery.BASE, recovery.load_prepared, recovery.source_seal)]
    finally:
        supervise.original.BASE = old_base
        supervise.original.load_prepared = old_loader
        supervise.original.source_seal = old_seal


def test_monitor_targets_continuation_without_launching(monkeypatch):
    from research.diagnostics_resume_v6 import monitor
    old_base, old_probe = monitor.original.BASE, monitor.original.PROBE
    old_remote = monitor.original.remote
    calls = []

    def remote(code, timeout=50):
        calls.append(code)
        return {'preparation': {'cpu-tests.json': {'status': 'PASS'}}}

    def run():
        assert monitor.original.BASE == recovery.BASE
        assert "b=root/'runs/diagnostics-v6-resume31b'" in monitor.original.PROBE
        assert monitor.original.remote(monitor.original.PROBE)['preparation'] == {}
        assert monitor.original.remote('ack')['preparation']

    monkeypatch.setattr(monitor.original, 'main', run)
    monkeypatch.setattr(monitor.original, 'remote', remote)
    try:
        monitor.main()
        assert len(calls) == 2
    finally:
        monitor.original.BASE, monitor.original.PROBE = old_base, old_probe
        monitor.original.remote = old_remote
