"""K1–K4: paired task-cluster intervals, distinct training seeds and descriptive source taxonomy."""
import argparse
from collections import Counter,defaultdict
import json
import math
import random
import statistics
from pathlib import Path
from eval.replication import read_rows,validate_rows
from eval.report import bootstrap,measurements,format_metric
from env.reward import extract_program
from tools.gatelib import write_json
from tools.replication import local_path,sha256,load_plan
from env.tasks import split_ids

def paired_interval(left,right,*,seed,repeats,alpha=.05):
    if set(left)!=set(right) or not left:raise ValueError('Unpaired or empty task grids')
    differences=[left[t]-right[t] for t in sorted(left)]
    rng=random.Random(seed)
    draws=sorted(statistics.fmean(rng.choices(differences,k=len(differences))) for _ in range(repeats))
    return {'value':statistics.fmean(differences),
            'ci':[draws[int(alpha/2*(repeats-1))],draws[int((1-alpha/2)*(repeats-1))]],
            'confidence':1-alpha,'unit':'task; conditional on these trained checkpoints'}

def task_rates(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['task']].append(float(r['result']['correct']))
    return {t:statistics.fmean(values) for t,values in grouped.items()}

def source_category(text):
    src=extract_program(text)
    if src is None:return 'not_extractable'
    # Indicators are explicitly NOT claims about what executed on the GPU.
    if any(t in src for t in ['load_inline','CUDAExtension','__global__']):return 'mentions_custom_cuda'
    if any(t in src for t in ['triton','@triton.jit']):return 'mentions_triton'
    if 'torch' in src:return 'pytorch_without_custom_kernel_markers'
    return 'unclassified'

def load_complete(path):
    path=Path(path);req=json.loads((path.parent/'request.json').read_text())
    if req.get('split')!='heldout':raise ValueError('Held-out report cannot ingest TRAIN data')
    complete=json.loads((path.parent/'complete.json').read_text())
    if complete['request_sha256']!=sha256(path.parent/'request.json'):
        raise ValueError('Evaluation request checksum mismatch')
    _,expected=split_ids()
    if set(req['tasks'])!=expected:raise ValueError('Incomplete held-out task grid')
    if complete['scores_sha256']!=sha256(path):raise ValueError('Score checksum mismatch')
    rows=read_rows(path);validate_rows(rows,req,scores=True)
    return rows

def common_correct_speed(left,right):
    """K2: descriptive paired ratios on common successful draws, not a population speed claim."""
    maps=[{(r['task'],r['sample']):r for r in rows} for rows in [left,right]]
    if set(maps[0])!=set(maps[1]):raise ValueError('Speed comparison needs identical task/sample IDs')
    ratios=[]
    for key in maps[0]:
        a,b=[m[key]['result'] for m in maps]
        if a['correct'] and b['correct']:
            ratios.append(a['speedup']/b['speedup'])
    return {'common_successful_draws':len(ratios),'all_draws':len(maps[0]),
        'geomean_speedup_ratio':math.exp(statistics.fmean(map(math.log,ratios))) if ratios else None,
        'scope':'conditional on both policies succeeding on the same task/input draw; descriptive'}

def report(plan,manifest):
    cells={};loaded={};by_arm=defaultdict(list)
    labels=set()
    for entry in manifest:
        label=entry['label']
        if label in labels:raise ValueError('Duplicate policy label')
        labels.add(label)
        rows=load_complete(local_path(entry['scores']))
        loaded[label]=rows;by_arm[entry['arm']].append(entry)
        cells[label]={'arm':entry['arm'],'training_seed':entry.get('training_seed'),
            'failure_components':dict(Counter(r['result']['component'] for r in rows)),
            'source_indicators':dict(Counter(source_category(r['text']) for r in rows)),
            'levels':{}}
        for level in ['level1','level2']:
            part=[r for r in rows if r['task'].startswith(level+'/')]
            stats=bootstrap(part,seed=plan['analysis']['seed'],repeats=plan['analysis']['bootstrap_repeats'])
            stats['fast_thresholds']={str(t):statistics.fmean(bool(r['result']['correct'] and r['result']['speedup']>t)
                for r in part) for t in plan['analysis']['speed_thresholds']}
            cells[label]['levels'][level]=stats
    if len(by_arm['base'])!=1:raise ValueError('Exactly one independently sampled base grid required')
    comparisons={}
    family=plan['analysis']['primary_comparisons']
    for left,right in family:
        if left not in by_arm or right not in by_arm:continue
        # Never promote a subset of completed seeds to the confirmatory aggregate.
        for arm in [left,right]:
            seeds=[e.get('training_seed') for e in by_arm[arm]]
            if any(s is not None for s in seeds) and sorted(seeds)!=sorted(plan['train_seeds']):
                raise ValueError('Incomplete/duplicate training seeds in confirmatory report')
        for level in ['level1','level2']:
            per_arm={}
            per_seed={}
            for arm in [left,right]:
                grids=[]
                for entry in by_arm[arm]:
                    grid=task_rates([r for r in loaded[entry['label']] if r['task'].startswith(level+'/')])
                    grids.append(grid);per_seed[entry['label']]=statistics.fmean(grid.values())
                if any(set(g)!=set(grids[0]) for g in grids):raise ValueError('Task grids differ across seeds')
                per_arm[arm]={t:statistics.fmean(g[t] for g in grids) for t in grids[0]}
            comparisons[f'{left}-{right}/{level}']={
                'paired':paired_interval(per_arm[left],per_arm[right],seed=plan['analysis']['seed'],
                    repeats=plan['analysis']['bootstrap_repeats'],
                    alpha=plan['analysis']['familywise_alpha']/len(family) if level=='level2' else .05),
                'policy_correctness':per_seed}
    speeds={}
    for entry in manifest:
        if entry['arm']=='base':continue
        for level in ['level1','level2']:
            base=loaded[by_arm['base'][0]['label']]
            speeds[f'{entry["label"]}-base/{level}']=common_correct_speed(
                [r for r in loaded[entry['label']] if r['task'].startswith(level+'/')],
                [r for r in base if r['task'].startswith(level+'/')])
    for c in by_arm.get('C',[]):
        for s in by_arm.get('S',[]):
            if c.get('training_seed')==s.get('training_seed'):
                for level in ['level1','level2']:
                    speeds[f'{s["label"]}-{c["label"]}/{level}']=common_correct_speed(
                        [r for r in loaded[s['label']] if r['task'].startswith(level+'/')],
                        [r for r in loaded[c['label']] if r['task'].startswith(level+'/')])
    return {'policies':cells,'comparisons':comparisons,'common_correct_speed':speeds,'limitations':[
        'Previously observed tasks; independent generation draws, not an unseen-task confirmation.',
        'Intervals resample task clusters conditional on saved policies; they do not estimate population training-seed uncertainty.',
        'Training seeds are reported separately; base samples are shared, never counted as extra independent baselines.',
        'Source indicators and detector flags require manual/runtime review; they are not verified mechanisms.',
        'Conditional speedup changes with the successful task set; no equivalence claim from a nonsignificant difference.']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=report(load_plan(a.plan),json.loads(a.manifest.read_text()))
    write_json(a.out.with_suffix('.json'),result)
    lines=['# Phase-2 replication report','',*result['limitations'],'',
        '| Policy | Level | Correctness [95% CI] | fast1 [95% CI] | Conditional speed [95% CI] |',
        '|---|---|---|---|---|']
    for label,cell in result['policies'].items():
        for level,m in cell['levels'].items():
            lines.append(f'| {label} | {level} | '+ ' | '.join(format_metric(m[k]) for k in
                ['correctness','fast1','correct_geomean_speedup'])+' |')
    lines+=['','Primary Level-2 comparison intervals use the preregistered Bonferroni family adjustment.',
            'Detailed comparisons, seed-specific rates and source/failure taxonomy are in the JSON companion.']
    a.out.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
