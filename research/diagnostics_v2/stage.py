"""CPU staging waits for verified weights, then seals a GPU-ready preparation."""
import argparse
import json
import os
from pathlib import Path
import time
from research.diagnostics_v2.common import BASE, ROOT, save, digest, source_seal, utc, load_prepared
from research.diagnostics_v2.prepare import prepare


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();directory=BASE/'preparation';initial=source_seal()
    save(directory/'stage-source.json',initial)
    save(directory/'stage-progress.json',{'stage':'waiting_for_download','pid':os.getpid(),'utc':utc()})
    try:
        deadline=time.monotonic()+24*3600
        while time.monotonic()<deadline:
            p=directory/'download-progress.json'
            if p.exists():
                state=json.loads(p.read_text())
                if state['status']=='stopped':raise RuntimeError('Weight download stopped: '+state.get('error','unknown'))
                if state['status']=='complete':break
            time.sleep(20)
        else:raise RuntimeError('Download wait exceeded 24 hours')
        if source_seal()!=initial:raise ValueError('Source changed during CPU staging')
        save(directory/'stage-progress.json',{'stage':'hashing_and_preparing','pid':os.getpid(),'utc':utc()})
        if (a.out/'receipt.json').exists():
            load_prepared(a.out)
        else:
            if a.out.exists():
                a.out.rename(a.out.with_name(a.out.name+'-interrupted-'+str(time.time_ns())))
            prepare(a.out)
        save(directory/'stage-progress.json',{'stage':'ready_for_GPU','prepared':str(a.out.resolve()),
             'receipt_sha256':digest(a.out/'receipt.json'),'pid':os.getpid(),'utc':utc()})
    except Exception as error:
        save(directory/'stage-progress.json',{'stage':'stopped','error':f'{type(error).__name__}: {error}',
             'pid':os.getpid(),'utc':utc()});raise


if __name__=='__main__':main()
