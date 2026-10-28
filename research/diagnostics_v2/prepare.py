"""CPU preparation after pinned downloads; no CUDA or model loading."""
import argparse
import json
from pathlib import Path
import yaml
from tools.replication import verify_frozen
from tools.weights import inspect_weights
from research.diagnostics_v1.prepare import load_prepared as old_prepared
from research.diagnostics_v2.common import BASE, PLAN, ROOT, WEIGHTS, digest, save, source_seal, utc


def prepare(out):
    verify_frozen(ROOT / 'runs/replication-v1.1')
    plan = yaml.safe_load(PLAN.read_text())
    if plan['training_enabled'] or plan['formal_test_enabled']:
        raise ValueError('Development only')
    previous = ROOT / plan['controls']['prepared_v5']
    _, tasks, _ = old_prepared(previous)
    downloaded = json.loads((BASE / 'preparation/download-receipt.json').read_text())
    if downloaded['manifest_sha256'] != digest(WEIGHTS):
        raise ValueError('31B weight receipt is not the current pinned snapshot')
    large = json.loads(WEIGHTS.read_text())
    models = {}
    for model in plan['models']:
        d = ROOT / 'models' / model
        structure = inspect_weights(d)
        files = [*sorted(d.glob('*.safetensors')), *[d / n for n in
            ['config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
             'chat_template.jinja', 'generation_config.json'] if (d / n).exists()]]
        record = {'model': model, 'structure': structure, 'files': {}}
        for p in files:
            print(json.dumps({'hashing_model': model, 'file': p.name}), flush=True)
            record['files'][p.name] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
        if model == 'gemma-4-31B-it':
            for item in large['files']:
                p = d / item['path']
                measured = record['files'].get(p.name) or {'bytes': p.stat().st_size, 'sha256': digest(p)}
                if measured != {'bytes': item['bytes'], 'sha256': item['sha256']}:
                    raise ValueError('31B changed after download')
                record['files'][p.name] = measured
        models[model] = record
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    (out / 'config.yaml').write_bytes(PLAN.read_bytes())
    save(out / 'tasks.json', tasks); save(out / 'models.json', models)
    receipt = {'status': 'CPU_prepared_GPU_validation_pending', 'utc': utc(),
        'source_sha256': source_seal(), 'v5_prepared_sha256': digest(previous / 'prepared.json'),
        'download_receipt_sha256': digest(BASE / 'preparation/download-receipt.json'),
        'files': {n: digest(out / n) for n in ['config.yaml', 'tasks.json', 'models.json']}}
    save(out / 'receipt.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().out)))


if __name__ == '__main__':
    main()
