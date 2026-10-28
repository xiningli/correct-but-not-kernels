"""K1–K4: workstation archive verification using an existing SSH master, never saved passwords."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from tools.gatelib import init_run,write_json

def read_status(ssh, command):
    """K1–K4: treat a stalled transport as retryable, not a lost backup worker."""
    try:
        return subprocess.run([*ssh,command],capture_output=True,text=True,timeout=30)
    except subprocess.TimeoutExpired:
        error='SSH status read timed out after 30 seconds'
        print(error+'; treating as unreachable',flush=True)
        return subprocess.CompletedProcess([*ssh,command],255,stdout='',stderr=error)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',required=True,help='K1–K4: existing authenticated SSH control socket')
    p.add_argument('--host',required=True,help='K1–K4: prepared AutoDL SSH destination')
    p.add_argument('--port',type=int,required=True,help='K1–K4: prepared AutoDL SSH port')
    p.add_argument('--remote-run',required=True,help='K1–K4: root-owned remote supervisor run directory')
    p.add_argument('--out',type=Path,required=True,help='K1–K4: new local backup directory')
    a=p.parse_args()
    init_run(a.out,{'hypotheses':['K1','K2','K3','K4'],'host':a.host,'port':a.port,
                   'remote_run':a.remote_run,'authentication':'existing control master'},20261006)
    ssh=['ssh','-S',a.socket,'-p',str(a.port),'-o','BatchMode=yes','-o','ConnectTimeout=10',
         '-o','ServerAliveInterval=5','-o','ServerAliveCountMax=3',a.host]
    started=time.monotonic();archived=False;unreachable=0
    while time.monotonic()-started<73000:
        result=read_status(ssh,'cat -- '+shlex.quote(a.remote_run+'/archive.json'))
        if result.returncode==0:
            unreachable=0
            meta=json.loads(result.stdout)
            remote=Path(meta['path'])
            if remote.parent!=Path(a.remote_run) or remote.name!='evidence-and-checkpoints.tgz':
                raise RuntimeError('Unexpected archive path')
            dest=a.out/remote.name
            subprocess.run([*ssh,'touch -- '+shlex.quote(a.remote_run+'/backup-local-started')],
                           check=True,timeout=30)
            transfer=subprocess.run(['scp','-o','ControlPath='+a.socket,'-o','BatchMode=yes',
                '-P',str(a.port),a.host+':'+str(remote),str(dest)],timeout=1800)
            if transfer.returncode: raise RuntimeError('Archive transfer failed')
            digest=hashlib.sha256()
            with dest.open('rb') as f:
                for block in iter(lambda:f.read(8<<20),b''):digest.update(block)
            if digest.hexdigest()!=meta['sha256'] or dest.stat().st_size!=meta['bytes']:
                raise RuntimeError('Archive checksum mismatch')
            write_json(a.out/'verified-archive.json',meta)
            subprocess.run([*ssh,'touch -- '+shlex.quote(a.remote_run+'/backup-local-ack')],
                           check=True,timeout=30)
            archived=True;break
        unreachable=unreachable+1 if result.returncode==255 else 0
        if unreachable>=10:
            write_json(a.out/'ssh-unavailable-before-archive.json',{'stderr':result.stderr})
            raise RuntimeError('SSH unavailable before archive could be verified')
        time.sleep(30)
    if not archived:raise RuntimeError('No completed archive within supervisor wall limit')
    for _ in range(24):
        result=read_status(ssh,'cat -- '+shlex.quote(a.remote_run+'/shutdown-return.json'))
        if result.returncode==0:
            write_json(a.out/'shutdown-return.json',json.loads(result.stdout))
            break
        if result.returncode==255:
            write_json(a.out/'ssh-unavailable-after-shutdown-request.json',
                       {'ssh_returncode':255,'stderr':result.stderr})
            break
        time.sleep(5)

if __name__=='__main__':main()
