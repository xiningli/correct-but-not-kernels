"""Current-boot GPU controls for same-program decomposition and fixed hosts."""
import argparse
import json
from pathlib import Path

from research.reliability_v2 import decomposition
from research.reliability_v2.controls import REFERENCE, TRITON
from research.reliability_v2.fixed_interface import assemble, control_kernel, extract_code
from research.reliability_v2.new_tasks import catalog, make_inputs, trusted_reference
from research.reliability_v2.strict_controls import CUDA
from research.reliability_v2.strict_evaluator import evaluate
from tools.gatelib import ROOT
from tools.replication import sha256


def source_hashes():
    paths = [*(ROOT / 'research/reliability_v2').glob('*.py'), ROOT / 'experiments/cuda-capability-v4.yaml']
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def boot_identity():
    import torch
    return {'pid1_start_ticks': Path('/proc/1/stat').read_text().split()[21],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda}


def main():
    import torch
    import types
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    initial_hashes = source_hashes()
    class Reference(torch.nn.Module):
        def forward(self, x):
            return x + 1
    reference = types.SimpleNamespace(Model=Reference, get_inputs=lambda: [torch.randn(257)], get_init_inputs=lambda: [])
    checks = []
    cases = [
        ('torch_numerical_only', REFERENCE, 'contract_violation', True, False),
        ('triton_actual', TRITON, 'correct', True, True),
        ('cuda_actual', CUDA, 'correct', True, True),
        ('wrong_math', TRITON.replace('value + 1.', 'value + 2.'), 'incorrect', False, True),
        ('dummy_plus_torch', TRITON.replace('return y', 'return x + 1'), 'contract_violation', True, False),
        ('initialization_crash', REFERENCE.replace('class ModelNew(nn.Module):', 'class ModelNew(nn.Module):\n    def __init__(self):\n        raise ValueError("controlled initialization failure")'), 'crash', None, None),
        ('forward_crash', REFERENCE.replace('return x + 1', 'raise ValueError("controlled forward failure")'), 'crash', None, None),
        ('weight_initialization_separate', TRITON.replace('class ModelNew(nn.Module):', 'class ModelNew(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.weight = nn.Parameter(torch.randn(1))'), 'correct', True, True),
    ]
    def record(name, ok, result):
        checks.append({'name': name, 'passed': ok, 'result': result})
        (a.out / 'checks.json').write_text(json.dumps(checks, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'name': name, 'passed': ok, 'component': result['component'], 'error': result.get('error')}), flush=True)
        if not ok:
            raise RuntimeError('Control failed; no policy scoring permitted')
    for name, source, component, numerical, compliant in cases:
        r = decomposition.evaluate(source, reference, a.out / name, seed=2026101502)
        record(name, r['component'] == component and r['numerically_correct'] is numerical and r['contract_compliant'] is compliant, r)
    for task in [t for t in catalog() if t['split'] == 'development']:
        cases = [make_inputs(task, seed=2026101502 + c, case=c) for c in range(4)]
        expected = [trusted_reference(task)(*x) for x in cases]
        source = assemble(extract_code('```python\n' + control_kernel(task) + '```'), task, 'fixed_host')
        r = evaluate(source, cases, expected, a.out / ('fixed-' + task['id'].split('/')[-1]), seed=2026101502)
        record(task['id'], r['correct'] is True, r)
    if source_hashes() != initial_hashes:
        raise ValueError('Code changed during controls')
    (a.out / 'receipt.json').write_text(json.dumps({'status': 'PASS', 'controls': len(checks),
        'source_sha256': initial_hashes, 'boot': boot_identity()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
