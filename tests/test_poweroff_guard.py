"""K1–K4: the unattended supervisor must never power off the development host."""
import sys
from pathlib import Path
import pytest
from tools import autonomous_pilot
from tools import run_replication

def test_refuses_non_autodl_host_before_any_side_effect(monkeypatch):
    monkeypatch.setattr(autonomous_pilot,'ROOT',Path('/tmp/not-the-autodl-project'))
    monkeypatch.setattr(autonomous_pilot.os,'geteuid',lambda:0)
    monkeypatch.setattr(sys,'argv',['supervisor','--operator','test','--power-off'])
    def unexpected(*args,**kwargs): raise AssertionError('Unexpected process launch')
    monkeypatch.setattr(autonomous_pilot.subprocess,'run',unexpected)
    with pytest.raises(RuntimeError,match='restricted to the prepared AutoDL'):
        autonomous_pilot.main()

def test_replication_refuses_non_autodl_host_before_any_side_effect(monkeypatch):
    monkeypatch.setattr(run_replication,'ROOT',Path('/tmp/not-the-autodl-project'))
    monkeypatch.setattr(run_replication.os,'geteuid',lambda:0)
    monkeypatch.setattr(sys,'argv',['supervisor','--study','runs/test','--stage','all','--power-off'])
    def unexpected(*args,**kwargs):raise AssertionError('Unexpected process launch')
    monkeypatch.setattr(run_replication.subprocess,'run',unexpected)
    with pytest.raises(RuntimeError,match='restricted to the prepared AutoDL'):
        run_replication.main()
