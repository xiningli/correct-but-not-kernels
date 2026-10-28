"""Bounded repair branches: identical first attempt, visible-only feedback/selection."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from research.diagnostics_v2.common import ROOT, load_prepared, source_seal, save, digest, utc, verify_model
from research.diagnostics_v1.controls import validate_controls
from research.diagnostics_v1.protocol import candidate_source
from research.diagnostics_v1.tasks import visible_suite, withheld_suite
from research.diagnostics_v1.evidence import group
from research.diagnostics_v2.feedback import visible_feedback
from research.diagnostics_v2.protocol import CONDITIONS, prompt, seed, branch, next_request, advance, final_result
from research.diagnostics_v2.report import report
from research.reliability_v2.strict_evaluator import evaluate


def append(path,row):
    with Path(path).open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()


def generate(prepared,model,entries,out,plan):
    import pynvml
    pynvml.nvmlInit(); gpu=pynvml.nvmlDeviceGetHandleByIndex(0)
    baseline=pynvml.nvmlDeviceGetMemoryInfo(gpu).used
    out.mkdir(parents=True,exist_ok=False)
    request={'scope':'two_development_defects_only','prepared':str(prepared.resolve()),'model':model,'entries':entries}
    save(out/'request.json',request)
    start=time.monotonic()
    with (out/'generation.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-m','research.diagnostics_v2.sample','--request',str(out/'request.json'),
            '--out',str(out/'generations.jsonl')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=p.wait(timeout=plan['limits']['generation_batch_seconds'])
        finally:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
    if code:raise RuntimeError('Generation failed; see '+str(out/'generation.log'))
    deadline=time.monotonic()+30
    while pynvml.nvmlDeviceGetMemoryInfo(gpu).used > baseline+(256<<20):
        if time.monotonic()>deadline:raise RuntimeError('Generator did not release GPU; no candidate execution')
        time.sleep(.5)
    save(out/'cost.json',{'model':model,'round':entries[0]['round'],'seconds':time.monotonic()-start,'candidates':len(entries)})
    rows=[json.loads(s) for s in (out/'generations.jsonl').read_text().splitlines()]
    if len(rows)!=len(entries):raise ValueError('Incomplete generation batch')
    for row,entry in zip(rows,entries,strict=True):
        if any(row[k]!=v for k,v in entry.items() if k!='messages'):raise ValueError('Candidate identity drift')
    return rows


def execute(prepared,controls,feedback_controls,out):
    import torch
    torch.set_num_threads(4)
    plan,tasks,models=load_prepared(prepared)
    initial=validate_controls(controls,ROOT/plan['controls']['prepared_v5'])
    fc=json.loads((feedback_controls/'receipt.json').read_text())
    if (fc['status']!='PASS' or fc['source_sha256']!=source_seal()
        or fc['prepared_sha256']!=digest(prepared/'receipt.json')
        or fc['base_controls_sha256']!=digest(controls/'receipt.json')
        or fc['feedback_sha256']!=digest(feedback_controls/'visible-feedback.json')):
        raise ValueError('Fresh API feedback controls required')
    out.mkdir(parents=True,exist_ok=False);initial_seal=source_seal()
    save(out/'source-seal.json',initial_seal);(out/'config.yaml').write_bytes((prepared/'config.yaml').read_bytes())
    save(out/'bindings.json',{'prepared_sha256':digest(prepared/'receipt.json'),'controls_sha256':digest(controls/'receipt.json'),
         'feedback_controls_sha256':digest(feedback_controls/'receipt.json')})
    outcome={'status':'running','utc':utc()};start=time.monotonic();episode_count=0;node_count=0
    feedback0={}
    for task,d in initial.items():
        f=d['facts'];feedback0[task]={'scope':'visible_only','summary':{'component':f['result.component'],
            'strict_pass':False,'checks_observed':f['result.total'],'checks_total':f['result.total'],'checks_passed':f['result.passed']},
            'detail':{'measured':group(f)}}
    try:
        for model in plan['models']:
            verify_model(models[model],full=True)
            states={};nodes={}
            for round_index in range(plan['development']['rounds']):
                if time.monotonic()-start>plan['limits']['run_wall_seconds']:raise RuntimeError('Run wall budget exhausted')
                if source_seal()!=initial_seal:raise ValueError('Source changed mid-run')
                entries=[]
                if round_index==0:
                    for task_id,task in tasks.items():
                        for rep in range(plan['development']['replicates']):
                            entries.append({**prompt(task,task['source'],feedback0[task_id],'initial'),
                                'task':task_id,'replicate':rep,'mode':'shared_initial','round':0,
                                'sampling_seed':seed(plan,task_id,rep,0)})
                else:
                    for state in states.values():
                        if not state['done']:entries.append(next_request(tasks[state['task']],state,plan,round_index))
                if not entries:break
                for e in entries:e['id']=e['task'].split('/')[-1]+f"-rep{e['replicate']}-{e['mode']}-r{round_index}"
                random.Random(plan['seeds']['order']+round_index).shuffle(entries)
                directory=out/model/f'round-{round_index}'
                save(out/'progress.json',{'stage':'generation','model':model,'round':round_index,'nodes':node_count,
                    'episodes_completed':episode_count,'episodes_total':plan['development']['episodes'],'utc':utc()})
                rows=generate(prepared,model,entries,directory,plan)
                for row in rows:
                    begin=time.monotonic();candidate=directory/row['id'];candidate.mkdir()
                    source=candidate_source(row['text'],row['finish_reason'])
                    inputs,expected=visible_suite(row['task'])
                    if source is None:
                        verdict={'component':'format','correct':False}
                        feedback=visible_feedback(row['task'],None,inputs,expected,verdict,
                            format_reason='truncated' if row['finish_reason']=='length' else 'invalid_program')
                    else:
                        (candidate/'program.py').write_text(source)
                        verdict=evaluate(source,inputs,expected,candidate/'visible',seed=plan['seeds']['evaluator'])
                        feedback=visible_feedback(row['task'],source,inputs,expected,verdict)
                    node={k:v for k,v in row.items() if k not in ['text','token_ids']}
                    node.update(model=model,source=source,feedback=feedback,visible_pass=verdict['correct'] is True,
                        component=verdict['component'],visible=verdict,evaluation_seconds=time.monotonic()-begin,
                        candidate_path=str(candidate.relative_to(out)))
                    nodes[node['id']]=node;append(out/'nodes.jsonl',node);node_count+=1
                    if round_index==0:
                        for condition in CONDITIONS:
                            states[(row['task'],row['replicate'],condition)]=branch(tasks[row['task']],row['replicate'],node,feedback0[row['task']],condition)
                    else:
                        advance(states[(row['task'],row['replicate'],row['mode'])],node,plan['development']['rounds'])
                    save(out/'progress.json',{'stage':'visible_evaluation','model':model,'round':round_index,
                        'nodes':node_count,'episodes_completed':episode_count,'episodes_total':plan['development']['episodes'],'utc':utc()})
                    report(out)
            # All visible-generation decisions are over before any withheld score is computed.
            hidden_cache={}
            for state in states.values():
                selected=nodes[state['selected']];hidden=None
                if selected['visible_pass']:
                    if selected['id'] not in hidden_cache:
                        inputs,expected=withheld_suite(state['task'])
                        hidden_cache[selected['id']]=evaluate(selected['source'],inputs,expected,
                            out/selected['candidate_path']/'withheld',seed=plan['seeds']['evaluator'])
                    hidden=hidden_cache[selected['id']]
                endpoint={'model':model,**final_result(state,nodes,hidden)}
                append(out/'episodes.jsonl',endpoint);episode_count+=1;report(out)
            save(out/model/'terminal-states.json',list(states.values()))
        if episode_count!=plan['development']['episodes'] or node_count>plan['development']['unique_generation_cap']:
            raise ValueError('Terminal grid/candidate budget mismatch')
        outcome['status']='complete'
    except Exception as error:
        outcome.update(status='stopped',error=f'{type(error).__name__}: {error}');raise
    finally:
        outcome.update(episodes=episode_count,nodes=node_count,finished_utc=utc(),seconds=time.monotonic()-start)
        save(out/'outcome.json',outcome);report(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','controls','feedback-controls','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();execute(a.prepared,a.controls,a.feedback_controls,a.out)


if __name__=='__main__':main()
