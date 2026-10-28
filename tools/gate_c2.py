"""K2: 30 randomized interleaved reference/fused pairs on a locked TRAIN task."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
from env.sandbox import Limits, run
from env.tasks import verified_catalog, validate_split
from tools.gatelib import ROOT, enforce_failure_stop, finish_gate, require_gate, timestamp

TASK = 'level1/88_MinGPTNewGelu'
WORKER = '''
import json, random
torch.manual_seed(2718)
x = get_inputs()[0].cuda()
reference = Model().eval().cuda()
with torch.inference_mode():
    assert torch.allclose(reference(x), fused(x), rtol=1e-4, atol=1e-5)
    timer = Timer()
    timer.warm(lambda: reference(x))
    timer.warm(lambda: fused(x))
    pairs = []
    rng = random.Random(0)
    for pair in range(30):
        order = ['reference', 'fused']
        rng.shuffle(order)
        result = {'order': order}
        for name in order:
            fn = (lambda: reference(x)) if name == 'reference' else (lambda: fused(x))
            result[name] = timer.measure(fn)
        pairs.append(result)
    with open('timings.json', 'w') as f:
        json.dump({'task': TASK_ID, 'shape': list(x.shape), 'pairs': pairs,
                   'config': asdict(timer.config)}, f)
'''

def main():
    enforce_failure_stop('C2')
    c1 = require_gate('C1')
    directory = ROOT / 'runs/c2' / timestamp()
    tasks = verified_catalog()
    train, _ = validate_split(tasks)
    assert TASK in train
    task = next(t for t in tasks if t.task_id == TASK)
    checks = []
    try:
        # Trusted calibration includes both control functions; policy evaluation
        # will never place expected values or reference source in its worker.
        source = '\n'.join([task.source(), (ROOT/'gold/fast_gelu.py').read_text(),
                            (ROOT/'env/timing.py').read_text(), WORKER.replace('TASK_ID', repr(TASK))])
        result = run(source, directory, replace(Limits(), wall_seconds=300, cpu_seconds=240))
        if result.component != 'ok':
            raise RuntimeError(f'{result}; {Path(result.log).read_text()}')
        data = json.loads((Path(result.workdir)/'timings.json').read_text())
        values = [p['reference']['median_ms'] for p in data['pairs']]
        wins = sum(p['fused']['median_ms'] < p['reference']['median_ms'] for p in data['pairs'])
        cv = statistics.pstdev(values)/statistics.fmean(values)
        # Each of the 30 observations is the preregistered median-of-N estimate.
        # Retain raw within-pair CVs too; never silently discard their outliers.
        raw_cv = max(p['reference']['cv'] for p in data['pairs'])
        checks = [{'name': 'fused_faster_all_pairs', 'status': 'PASS' if wins == 30 else 'FAIL',
                   'wins': wins, 'pairs': 30},
                  {'name': 'reference_noise', 'status': 'PASS' if cv<=0.05 else 'FAIL',
                   'pair_median_cv': cv, 'maximum_within_pair_cv': raw_cv}]
        print(json.dumps(checks), flush=True)
    except Exception as error:
        checks.append({'name': 'blocking_failure', 'status': 'FAIL',
                       'error': f'{type(error).__name__}: {error}'})
        print(checks[-1]['error'], flush=True)
    raise SystemExit(finish_gate('C2', checks, prerequisite=c1,
                                artifacts=str(directory.relative_to(ROOT))))

if __name__ == '__main__':
    main()
