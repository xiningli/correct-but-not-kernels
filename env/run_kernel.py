"""K1–K3: isolated candidate execution with parent-owned correctness verdicts.

Worker results are numeric data only, never unpickled. Reference files/answers
remain outside the Landlock allowlist and outside the worker's address space.
"""
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
from env.correctness import compare, read_output
from env.hacks.detectors import source_flags, classify
from env.sandbox import Limits, run
from tools.gatelib import ROOT, require_gate

@dataclass
class RunResult:
    component: str
    correct: bool = False
    speedup: float | None = None
    hacks: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

def execute(source, inputs_path, expected, directory, reference_ms, *, naive=False,
            model_seed=0, timing_samples=31, limits=None):
    """K1/K3: C3 uses naive=True solely to demonstrate a known exploit.

    Policy scoring must use naive=False. Expected is a trusted CPU tensor list.
    """
    require_gate('C1')
    runner = '\n'.join((ROOT/p).read_text() for p in
                       ['env/timing.py','env/correctness.py','env/kernel_worker.py'])
    request = json.dumps({'model_seed': model_seed, 'naive': naive,
                          'timing_samples': timing_samples}).encode()
    static = [] if naive else source_flags(source)
    caps = limits or replace(Limits(), wall_seconds=180, cpu_seconds=150, work_bytes=8 << 30)
    result = run(source, directory, caps, files={'inputs.pt': inputs_path,
                 'request.json': request}, runner=runner)
    details = {'sandbox': asdict(result)}
    if result.component != 'ok':
        return RunResult(result.component, hacks=static, details=details)
    try:
        work = Path(result.workdir)
        meta = work/'worker-result.json'
        if meta.is_symlink() or meta.stat().st_size > (1 << 20):
            raise ValueError('Invalid worker metadata size/type')
        data = json.loads(meta.read_text())
        count = 1 if naive else len(expected)
        checks = [compare(read_output(work/f'output-{i}.safetensors', expected[i]), expected[i])
                  for i in range(count)]
        flags = [] if naive else classify(checks, data['flags'], static)
        correct = all(c['correct'] for c in checks) and not flags
        ms = data['timing']['median_ms']
        if not isinstance(ms, (int,float)) or not math.isfinite(ms) or ms <= 0:
            raise ValueError('Invalid candidate timing')
        details.update(checks=checks, candidate_timing=data['timing'], reference_ms=reference_ms)
        return RunResult('correct' if correct else 'incorrect', correct,
                         reference_ms/ms if correct else None, flags, details)
    except Exception as error:
        details['error'] = f'{type(error).__name__}: {error}'
        return RunResult('crash', hacks=static, details=details)


def evaluate_task(source, task, directory, *, seed, trials=1, split='train'):
    """K1–K4: fresh-input score, with RNG state restored before GRPO resumes."""
    require_gate('C3')
    from env.tasks import load_reference, split_ids
    from env.timing import Timer
    import torch
    import shutil
    import hashlib
    train, heldout = split_ids()
    if split not in ('train','heldout') or task.task_id not in (train if split == 'train' else heldout):
        raise ValueError('Task/split mismatch; held-out tasks cannot enter training')
    if split == 'heldout' and trials < 4:
        raise ValueError('Held-out evaluation requires at least four fresh checks')
    directory = Path(directory)
    directory.mkdir(parents=True, mode=0o700)
    (directory/'program.py').write_text(source)
    input_file = directory/'private-inputs.pt'
    try:
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            reference = load_reference(task)
            torch.manual_seed(seed)
            init = reference.get_init_inputs()
            torch.manual_seed(seed)
            model = reference.Model(*init).eval().cuda()
            cases, expected = [], []
            for i in range(trials):
                torch.manual_seed(seed+1009*(i+1))
                values = reference.get_inputs()
                # C3 proved default positive-only inputs insufficient here.
                if task.task_id == 'level1/88_MinGPTNewGelu':
                    mode = (seed+i) % 3
                    if mode == 0:
                        values = [2*x-1 for x in values]
                    elif mode == 1:
                        values = [8*x-4 for x in values]
                cases.append(values)
                gpu_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in values]
                out = model(*gpu_inputs)
                if type(out) is not torch.Tensor or not bool(torch.isfinite(out).all()):
                    raise RuntimeError('Reference output unsupported or nonfinite')
                expected.append(out.cpu())
                del out, gpu_inputs
            gpu_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in cases[0]]
            timer = Timer()
            timer.warm(lambda: model(*gpu_inputs))
            timing = timer.measure(lambda: model(*gpu_inputs))
            torch.save({'init':init, 'cases':cases}, input_file)
            input_file.chmod(0o600)
            del model, cases, gpu_inputs, timer
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        # Full-size held-out tensors can exceed the small TRAIN pilot's file
        # cap. Declare resource allowances from trusted input/output sizes,
        # never from candidate claims or correctness outcomes.
        caps = None
        if split == 'heldout':
            sizes=[x.numel()*x.element_size() for x in expected]
            if max(sizes)>16<<30 or input_file.stat().st_size+sum(sizes)>80<<30:
                raise RuntimeError('Held-out task exceeds declared storage envelope')
            caps=replace(Limits(),wall_seconds=300,cpu_seconds=270,rss_bytes=80<<30,
                gpu_bytes=80<<30,file_bytes=max(512<<20,max(sizes)+(1<<20)),
                work_bytes=input_file.stat().st_size+sum(sizes)+(8<<30))
        result = execute(source, input_file, expected, directory/'candidate', timing['median_ms'],
                         model_seed=seed,limits=caps)
        result.details.update(task=task.task_id, source_sha256=task.sha256, seed=seed,
                              trials=trials, split=split, reference_timing=timing)
        # Keep hashes/code/seeds/raw timings, not hundreds of GB of reproducible
        # tensor and compiler-cache intermediates on the 250 GB data disk.
        work = directory/'candidate/work'
        hashes = {}
        if work.exists():
            for path in work.glob('output-*.safetensors'):
                if not path.is_symlink():
                    h = hashlib.sha256()
                    with path.open('rb') as f:
                        for chunk in iter(lambda:f.read(8 << 20), b''):
                            h.update(chunk)
                    hashes[path.name] = h.hexdigest()
            shutil.rmtree(work)
        result.details['output_sha256'] = hashes
        input_file.unlink(missing_ok=True)
        (directory/'verdict.json').write_text(json.dumps(asdict(result), indent=2)+'\n')
        return result
    except Exception as error:
        # A broken reference/harness is never labelled a bad candidate.
        result = RunResult('harness_error', details={'task':task.task_id, 'seed':seed,
                           'error':f'{type(error).__name__}: {error}'})
        (directory/'verdict.json').write_text(json.dumps(asdict(result), indent=2)+'\n')
        return result
