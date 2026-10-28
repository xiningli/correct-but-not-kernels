"""Immutable v6 runtime and preparation identities; previous studies stay sealed."""
import json
from pathlib import Path
import yaml
from research.diagnostics_v1.common import ROOT, digest, object_hash, save, utc, source_seal as old_seal

PLAN = ROOT / 'experiments/cuda-diagnostics-v6.yaml'
WEIGHTS = ROOT / 'experiments/gemma31b-weights-v6.json'
BASE = ROOT / 'runs/diagnostics-v6'


def source_seal():
    return {**old_seal(), **{str(p.relative_to(ROOT)): digest(p) for p in
            [PLAN, WEIGHTS, *sorted((ROOT / 'research/diagnostics_v2').glob('*.py'))]}}


def load_prepared(path):
    path = Path(path)
    receipt = json.loads((path / 'receipt.json').read_text())
    if receipt['source_sha256'] != source_seal():
        raise ValueError('v6 prepared source changed')
    for name, sha in receipt['files'].items():
        if digest(path / name) != sha:
            raise ValueError('v6 prepared file changed: ' + name)
    from research.diagnostics_v1.prepare import load_prepared as old_prepared
    plan = yaml.safe_load((path / 'config.yaml').read_text())
    old = ROOT / plan['controls']['prepared_v5']
    _, tasks, _ = old_prepared(old)
    if digest(old / 'prepared.json') != receipt['v5_prepared_sha256']:
        raise ValueError('Initial diagnostic preparation changed')
    if tasks != json.loads((path / 'tasks.json').read_text()):
        raise ValueError('Public tasks changed')
    return plan, tasks, json.loads((path / 'models.json').read_text())


def verify_model(record, *, full=False):
    directory = ROOT / 'models' / record['model']
    for name, spec in record['files'].items():
        p = directory / name
        if p.stat().st_size != spec['bytes'] or (full or not name.endswith('.safetensors')) and digest(p) != spec['sha256']:
            raise ValueError('Model identity changed: ' + record['model'] + '/' + name)
    return directory
