"""K1–K4: reproduce operational facts from archived pilot records without selecting on held-out items."""
import argparse
import json
from pathlib import Path
from tools.gatelib import write_json

def audit(pilot):
    result={'arms':{},'kl_cause':'unresolved; GPU reference-invariance and diagnostic replay required'}
    for arm in ['C','S']:
        directory=Path(pilot)/f'training-{arm}'
        completions=sorted(directory.glob('completion-step-*.json'),key=lambda p:int(p.stem.split('-')[-1]))
        final=json.loads(completions[-1].read_text())
        metrics=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
        loss_rows=[r for r in metrics if 'kl' in r]
        resumes=[json.loads(p.read_text()) for p in directory.glob('resume-checkpoint-*.json')]
        by_step={json.loads(p.read_text())['step']:json.loads(p.read_text()) for p in completions}
        matches=all(by_step[int(Path(r['checkpoint']).name.split('-')[-1])]['lora_hash']==r['lora_hash'] for r in resumes)
        result['arms'][arm]={'final_step':final['step'],'resume_hashes_match':matches,
            'max_logged_kl':max(r['kl'] for r in loss_rows),'last_kl':loss_rows[-1]['kl'],
            'gpu_reclamation_max_delta_bytes':max(r.get('env/gpu_delta_bytes',0) for r in loss_rows),
            'stop':json.loads((directory/'kill.json').read_text())['reason'] if (directory/'kill.json').exists() else 'completed',
            'kl_last_steps':[{'step':r['step'],'kl':r['kl']} for r in loss_rows[-6:]]}
        if not matches:raise ValueError('Pilot resume hashes do not match saved policy')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();write_json(a.out,audit(a.pilot))

if __name__=='__main__':main()
