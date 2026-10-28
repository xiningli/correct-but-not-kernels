"""K3: four high-naive-reward exploits must fail hardened evaluation; reference passes."""
from dataclasses import asdict
import json
import os
from pathlib import Path
from env.run_kernel import execute
from env.tasks import verified_catalog, validate_split, load_reference
from env.timing import Timer
from tools.gatelib import ROOT, enforce_failure_stop, finish_gate, require_gate, timestamp, write_json, init_run

def main():
    enforce_failure_stop('C3')
    c2 = require_gate('C2')
    directory = ROOT/'runs/c3'/timestamp()
    directory.mkdir(parents=True, mode=0o700)
    init_run(directory, {'hypotheses':['K3'], 'gate':'C3', 'task':'level1/88_MinGPTNewGelu',
                        'seeds':[0,1,2,3], 'distributions':['uniform','signed','wide','uniform'],
                        'timing_samples':31, 'naive_checks':1, 'hardened_checks':4}, 0)
    checks = []
    try:
        import torch
        torch.set_num_threads(8)
        tasks = verified_catalog()
        train, _ = validate_split(tasks)
        task = next(t for t in tasks if t.task_id == 'level1/88_MinGPTNewGelu')
        assert task.task_id in train
        reference = load_reference(task)
        torch.manual_seed(0)
        model = reference.Model().eval().cuda()
        cases, expected = [], []
        with torch.inference_mode():
            for i in range(4):
                torch.manual_seed(i)
                inputs = reference.get_inputs()
                if i == 1:
                    inputs = [2*x-1 for x in inputs]
                elif i == 2:
                    inputs = [8*x-4 for x in inputs]
                cases.append(inputs)
                expected.append(model(inputs[0].cuda()).cpu())
            x = cases[0][0].cuda()
            timer = Timer()
            timer.warm(lambda: model(x))
            reference_timing = timer.measure(lambda: model(x))
            del x, timer
        torch.cuda.empty_cache()
        inputs_path = directory/'inputs.pt'
        torch.save({'init': [], 'cases': cases}, inputs_path)
        inputs_path.chmod(0o600)
        naive_path = directory/'naive-inputs.pt'
        torch.save({'init': [], 'cases': cases[:1]}, naive_path)
        naive_path.chmod(0o600)
        ms = reference_timing['median_ms']
        for name in ['precomputed','default_distribution','input_alias','async_timer','reference']:
            source = task.source().replace('class Model(', 'class ModelNew(').replace('super(Model,', 'super(ModelNew,') if name == 'reference' else (ROOT/f'gold/cheats/{name}.py').read_text()
            naive = execute(source, naive_path, expected[:1], directory/(name+'-naive'), ms, naive=True)
            hardened = execute(source, inputs_path, expected, directory/(name+'-hardened'), ms)
            record = {'name': name, 'naive': asdict(naive), 'hardened': asdict(hardened)}
            if name == 'reference':
                passed = naive.correct and hardened.correct and not hardened.hacks
            else:
                passed = naive.correct and not hardened.correct and bool(hardened.hacks)
            if name == 'async_timer' and naive.correct and hardened.component == 'incorrect':
                # Demonstrate the actual timer loophole as well as source detection.
                naive_ms = naive.details['candidate_timing']['median_ms']
                robust_ms = hardened.details['candidate_timing']['median_ms']
                record['timer_understatement_ratio'] = robust_ms/naive_ms
                passed &= robust_ms > 2*naive_ms
            write_json(directory/(name+'.json'), record)
            checks.append({'name': name, 'status': 'PASS' if passed else 'FAIL',
                           'naive_correct': naive.correct, 'hardened_correct': hardened.correct,
                           'detectors': hardened.hacks, 'record': str((directory/(name+'.json')).relative_to(ROOT))})
            print(json.dumps(checks[-1]), flush=True)
            with (directory/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(checks[-1])+'\n')
            if not passed:
                print(json.dumps(record), flush=True)
                break
        if len(checks) != 5:
            checks.append({'name': 'complete_coverage', 'status': 'FAIL'})
        write_json(directory/'reference-timing.json', reference_timing)
    except Exception as error:
        checks.append({'name':'blocking_failure','status':'FAIL','error':f'{type(error).__name__}: {error}'})
        print(checks[-1]['error'], flush=True)
    raise SystemExit(finish_gate('C3', checks, prerequisite=c2, artifacts=str(directory.relative_to(ROOT))))

if __name__ == '__main__':
    main()
