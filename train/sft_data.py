"""K1/K4: verified TRAIN-only rejection-sampled SFT; no candidate code executes here."""
import argparse
import json
from pathlib import Path
from env.tasks import split_ids, verified_catalog
from env.reward import extract_program
from eval.replication import read_rows, validate_rows
from tools.gatelib import write_json
from tools.replication import sha256

def select_successes(rows, allowed_tasks):
    train,_=split_ids()
    if not set(allowed_tasks)<=train:raise ValueError('SFT requested held-out data')
    selected=[]
    for row in rows:
        if row['task'] not in allowed_tasks:raise ValueError('Foreign task in SFT source')
        result=row['result']
        if result['component']=='harness_error':raise ValueError('Broken verifier cannot select SFT data')
        if result['correct'] and not result.get('hacks') and row.get('finish_reason')=='stop':
            if extract_program(row['text']) is None:raise ValueError('Successful record has no extractable program')
            if not row.get('token_ids'):raise ValueError('SFT requires original generation token IDs')
            selected.append(row)
    if not selected:raise ValueError('No verified untruncated TRAIN successes; SFT is blocked')
    return selected

def prepare(source,out,allowed_tasks,expected_candidates):
    source=Path(source);out=Path(out)
    req=json.loads((source/'request.json').read_text())
    if req.get('split')!='train' or set(req['tasks'])!=set(allowed_tasks):
        raise ValueError('SFT accepts only explicitly declared TRAIN evaluations')
    done=json.loads((source/'complete.json').read_text())
    if done['scores_sha256']!=sha256(source/'scores.jsonl'):
        raise ValueError('SFT source checksum mismatch')
    if done['generations_sha256']!=sha256(source/'generations.jsonl'):
        raise ValueError('SFT generation checksum mismatch')
    rows=read_rows(source/'scores.jsonl');validate_rows(rows,req,scores=True)
    if len(rows)!=expected_candidates:raise ValueError('Changed SFT candidate budget')
    hashes={t.task_id:t.sha256 for t in verified_catalog()}
    if any(r['source_sha256']!=hashes[r['task']] for r in rows):raise ValueError('Stale SFT task source')
    selected=select_successes(rows,allowed_tasks)
    out.mkdir(parents=True,exist_ok=False)
    with (out/'train.jsonl').open('x') as f:
        for row in selected:f.write(json.dumps(row)+'\n')
    write_json(out/'manifest.json',{'source_scores_sha256':sha256(source/'scores.jsonl'),
        'data_sha256':sha256(out/'train.jsonl'),'candidates':len(rows),'accepted':len(selected),
        'tasks':allowed_tasks,'selection':'correct and no flags and finish_reason=stop; retain duplicate frequency'})
    return out/'train.jsonl'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--expected-candidates',type=int,required=True)
    a=p.parse_args();prepare(a.source,a.out,a.tasks,a.expected_candidates)

if __name__=='__main__':main()
