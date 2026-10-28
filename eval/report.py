"""K1–K4: records to tables; task-cluster bootstrap, never hand-entered metrics.

Report/CI pattern adapted from glomergence/eval/report.py at
e602f63207fe4e67eeda4dff7b53d49e92cc318b. KernelBench tasks, rather than fresh
input checks, are independent sampling units here. Conditional speed is labelled.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
from datetime import datetime
from eval.gates import render
from eval.taxonomy import summarize as taxonomy
from tools.gatelib import ROOT, write_json

METRICS=('correctness','fast1','correct_geomean_speedup','detector_fraction')

def measurements(rows):
    if not rows:
        raise ValueError('No programs')
    if any(r['result']['component']=='harness_error' for r in rows):
        raise ValueError('Infrastructure failures cannot be policy scores')
    results=[r['result'] for r in rows]
    speeds=[r['speedup'] for r in results if r['correct']]
    return dict(correctness=statistics.fmean(r['correct'] for r in results),
        fast1=statistics.fmean(r['correct'] and r['speedup']>1 for r in results),
        correct_geomean_speedup=math.exp(statistics.fmean(math.log(s) for s in speeds)) if speeds else None,
        detector_fraction=statistics.fmean(bool(r.get('hacks',[])) for r in results))

def interval(values):
    finite=sorted(x for x in values if x is not None)
    # Missing conditional-speed resamples must not be silently discarded.
    if len(finite)!=len(values):
        return None
    return [finite[int(.025*(len(finite)-1))],finite[int(.975*(len(finite)-1))]]

def bootstrap(rows, seed=20261005, repeats=2000):
    groups=defaultdict(list)
    for row in rows:
        groups[row['task']].append(row)
    tasks=sorted(groups); rng=random.Random(seed); draws={k:[] for k in METRICS}
    for _ in range(repeats):
        sample=[row for task in rng.choices(tasks,k=len(tasks)) for row in groups[task]]
        for key,value in measurements(sample).items():
            draws[key].append(value)
    point=measurements(rows)
    return {key:{'value':point[key],'ci95':interval(draws[key])} for key in METRICS}

def paired_change(base,trained,seed=20261005,repeats=2000):
    groups=[]
    for rows in (base,trained):
        group=defaultdict(list)
        for row in rows: group[row['task']].append(row)
        groups.append(group)
    if set(groups[0])!=set(groups[1]):
        raise ValueError('Paired comparison needs identical tasks')
    tasks=sorted(groups[0]); rng=random.Random(seed)
    values=[]
    for _ in range(repeats):
        chosen=rng.choices(tasks,k=len(tasks))
        rates=[measurements([r for task in chosen for r in g[task]])['correctness'] for g in groups]
        values.append(rates[1]-rates[0])
    return {'value':measurements(trained)['correctness']-measurements(base)['correctness'],
            'ci95':interval(values)}

def format_metric(m):
    if m['value'] is None: return '—'
    ci=m['ci95']
    return f"{m['value']:.4f}"+(f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ' [CI undefined]')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,help='K1–K4: label to complete held-out scores JSONL paths')
    p.add_argument('--out',type=Path,default=ROOT/'REPORT.md',help='K1–K4: generated Markdown destination')
    a=p.parse_args()
    lines=['# Kernel engineering pilot report','',render().strip(),'']
    records=sorted((ROOT/'runs/gates').glob('c4-*.json'))
    if records:
        lines+=['## Base-policy TRAIN signal history','',
                '| Attempt | Model | Task | Correct / n | pass@1 | pass@8 | P_signal |',
                '|---|---|---|---:|---:|---:|---:|']
        throughputs=[]
        for attempt,path in enumerate(records,1):
            record=json.loads(path.read_text()); count=0
            for check in record['checks']:
                for model,summary in check.get('summaries',{}).items():
                    for task,values in summary['tasks'].items():
                        count+=values['n']
                        lines.append(f"| {attempt} | {model} | {task} | {values['correct']} / {values['n']} | "
                            f"{values['pass_at_1']:.4f} | {values['pass_at_8']:.4f} | {values['p_signal']:.4f} |")
            if count:
                start=datetime.strptime(Path(record['artifacts']).name,'%Y%m%dT%H%M%S.%fZ')
                end=datetime.strptime(record['time_utc'],'%Y%m%dT%H%M%S.%fZ')
                rate=3600*count/(end-start).total_seconds()
                throughputs.append(f"Attempt {attempt}: {rate:.1f} rollouts/hour end to end, including model loading/generation and serial evaluation.")
        lines+=['',*throughputs,'',
                'These are TRAIN selection measurements. Attempt 2 changes the prompt and task subset; it is not a paired model-improvement experiment.','']
    if not a.manifest:
        lines+=['No completed held-out pilot comparison is available. K1, K2 and K4 are underpowered.',
                'C3 establishes coverage of the declared cheat controls; it does not establish robustness to arbitrary exploits.',
                'C4 TRAIN signal is a prerequisite measurement, not evidence of RL improvement.']
    else:
        manifest=json.loads(a.manifest.read_text()); loaded={}; analysis={}
        from env.tasks import split_ids
        _,expected_tasks=split_ids()
        for label,path in manifest.items():
            rows=[json.loads(x) for x in Path(path).read_text().splitlines()]
            if set(r['task'] for r in rows)!=expected_tasks or not (Path(path).parent/'complete.json').is_file():
                raise ValueError('Only complete evaluations of the fixed held-out grid may enter this table')
            if any(r.get('evaluation_trials',0)<4 for r in rows):
                raise ValueError('Report requires four fresh checks per held-out program')
            loaded[label]=rows
        if not {'base','C','S'}<=loaded.keys():
            raise ValueError('Base, correctness arm and speed arm are required')
        lines+=['Each cell is an estimate with a 95% percentile bootstrap interval (2,000 task-cluster resamples).',
                'One generated program per task is a noisy pilot estimate. Fresh input checks are not counted as independent programs.',
                'Speed is the geometric mean conditional on correctness; changing the successful task set can change it.',
                'Empirical bootstrap intervals at all-pass/all-fail boundaries can be degenerate.','',
                '| Policy | Level | Tasks | Correctness (fast0) | fast1 | Correct-only speedup | Detector fraction |',
                '|---|---|---:|---|---|---|---|']
        for label,rows in loaded.items():
            analysis[label]={'taxonomy':taxonomy(rows),'levels':{}}
            for level in ('level1','level2'):
                part=[r for r in rows if r['task'].startswith(level+'/')]
                stats=bootstrap(part); analysis[label]['levels'][level]=stats
                lines.append(f"| {label} | {level} | {len({r['task'] for r in part})} | "+
                             ' | '.join(format_metric(stats[k]) for k in METRICS)+' |')
        lines+=['','| Paired correctness change vs base | Level | Estimate [95% CI] |','|---|---|---|']
        for label in ('C','S'):
            analysis[label]['paired_correctness_change']={}
            for level in ('level1','level2'):
                change=paired_change(*[[r for r in loaded[x] if r['task'].startswith(level+'/')]
                                      for x in ('base',label)])
                analysis[label]['paired_correctness_change'][level]=change
                lines.append(f'| {label} | {level} | {format_metric(change)} |')
        write_json(a.out.with_suffix('.json'),analysis)
        lines+=['','These short pilots estimate effects; detector flags require source/runtime review before claims of exploitation.',
                'K1/K4 evidence is given by paired correctness intervals above. K2 requires reading speed and correctness together;',
                'conditional speed alone is insufficient to claim an improved general kernel engineer.']
    a.out.write_text('\n'.join(lines)+'\n')
    print(a.out)

if __name__ == "__main__":
    main()
