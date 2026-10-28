"""K1–K4: CPU-only protocol and archived-artifact gate; makes no GPU-validation claim."""
import argparse
import json
from pathlib import Path
from tools.gatelib import ROOT,finish_gate,require_gate,enforce_failure_stop
from tools.replication import verify_frozen,sha256,local_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--archive-meta',type=Path,required=True)
    a=p.parse_args();enforce_failure_stop('R0');require_gate('C6')
    checks=[]
    try:
        plan,_=verify_frozen(a.study)
        meta=json.loads(a.archive_meta.read_text())
        if a.archive.stat().st_size!=meta['bytes'] or sha256(a.archive)!=meta['sha256']:
            raise ValueError('Archive hash/size mismatch')
        checks.append({'name':'pilot_archive','status':'PASS','sha256':meta['sha256'],'bytes':meta['bytes']})
        for arm,adapter in plan['pilot_adapters'].items():
            if sha256(local_path(adapter['path'])/'adapter_model.safetensors')!=adapter['sha256']:
                raise ValueError(f'{arm} final adapter hash mismatch')
        checks.append({'name':'frozen_protocol_and_final_adapters','status':'PASS'})
    except Exception as e:
        checks.append({'name':'preparation','status':'FAIL','error':f'{type(e).__name__}: {e}'})
    raise SystemExit(finish_gate('R0',checks,protocol_sha256=sha256(a.study/'frozen.json'),
        artifacts=str(a.study.relative_to(ROOT)),scope='CPU validation only; R1 GPU audit and SFT smoke remain mandatory'))

if __name__=='__main__':main()
