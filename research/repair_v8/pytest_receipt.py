"""Record original repository test outcomes; skipped tests are not passes."""
import json
import os
from pathlib import Path

ROWS = []


def pytest_runtest_logreport(report):
    if report.when == 'call' or (report.when in ['setup', 'teardown'] and report.outcome != 'passed'):
        ROWS.append({'nodeid': report.nodeid, 'phase': report.when, 'outcome': report.outcome,
                     'duration': report.duration, 'detail': str(report.longrepr)[:4000] if report.longrepr else None})


def pytest_sessionfinish(session, exitstatus):
    path = Path(os.environ['REPAIR_V8_PYTEST_RECEIPT'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'exitstatus': int(exitstatus), 'collected': session.testscollected,
                              'rows': ROWS}, indent=2) + '\n')
