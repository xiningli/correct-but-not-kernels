"""K1–K4: transient SSH outages must not silently abandon checkpoint backup."""
import hashlib
import json
import subprocess
import sys

from tools import watch_pilot


def test_timeout_then_verified_backup(monkeypatch, tmp_path):
    remote='/root/autodl-tmp/project/runs/autonomous/test'
    payload=b'checkpoint archive bytes'
    meta={'path':remote+'/evidence-and-checkpoints.tgz',
          'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(sys,'argv',['watch_pilot','--socket','/tmp/test-control',
        '--host','test-host','--port','22','--remote-run',remote,'--out',str(tmp_path)])
    monkeypatch.setattr(watch_pilot,'init_run',lambda *args:None)
    monkeypatch.setattr(watch_pilot.time,'sleep',lambda _:None)
    reads=0
    ack=[]
    def run(command,**kwargs):
        nonlocal reads
        stdout=''
        if command[-1].endswith('/archive.json'):
            reads+=1
            if reads==1:
                raise subprocess.TimeoutExpired(command,30)
            stdout=json.dumps(meta)
        elif command[0]=='scp':
            (tmp_path/'evidence-and-checkpoints.tgz').write_bytes(payload)
        elif command[-1].endswith('/backup-local-ack'):
            assert json.loads((tmp_path/'verified-archive.json').read_text())==meta
            ack.append(True)
        elif command[-1].endswith('/shutdown-return.json'):
            stdout='{"returncode": 0}'
        return subprocess.CompletedProcess(command,0,stdout=stdout,stderr='')
    monkeypatch.setattr(watch_pilot.subprocess,'run',run)
    watch_pilot.main()
    assert reads==2
    assert ack==[True]
    assert json.loads((tmp_path/'shutdown-return.json').read_text())=={'returncode':0}
