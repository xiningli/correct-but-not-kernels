"""The rental watchdog must stop workers that start their own process sessions."""
import json
import os
from pathlib import Path
import signal
import sys

import pytest

from research.repair_v8 import supervise


def test_stage_timeout_stops_detached_descendants(tmp_path, monkeypatch):
    monkeypatch.setattr(supervise, 'PROJECT', tmp_path)
    control = tmp_path / 'control'; control.mkdir()
    pidfile = tmp_path / 'grandchild.pid'
    statement = (
        'import subprocess,sys,time; from pathlib import Path; '
        'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True); '
        'Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)'
    )
    try:
        with pytest.raises(RuntimeError, match='exceeded wall-time budget'):
            supervise.stage('watchdog-test', [sys.executable, '-c', statement, str(pidfile)],
                            os.environ.copy(), control, 2)
        parent = json.loads((control / 'child.json').read_text())['pid']
        for pid in [parent, int(pidfile.read_text())]:
            status = Path('/proc') / str(pid) / 'status'
            if status.exists():
                state = next(line for line in status.read_text().splitlines() if line.startswith('State:'))
                assert 'Z (zombie)' in state, state  # exited but awaiting init's reap
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
