"""Prepare only the ungenerated 31B arm from one exact stopped-run archive."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile

import yaml
from research.diagnostics_v2 import common as original

ROOT = original.ROOT
BASE = ROOT / 'runs/diagnostics-v6-resume31b'
PROTOCOL = ROOT / 'experiments/cuda-diagnostics-v6-resume31b.yaml'


def source_seal():
    files = [PROTOCOL, *sorted((ROOT / 'research/diagnostics_resume_v6').glob('*.py'))]
    return {**original.source_seal(), **{str(p.relative_to(ROOT)): original.digest(p) for p in files}}


def resumed_plan(parent, protocol):
    if parent != yaml.safe_load(original.PLAN.read_text()):
        raise ValueError('Parent protocol changed')
    plan = copy.deepcopy(parent)
    plan['id'] = protocol['id']
    plan['models'] = [protocol['resume_model']]
    plan['inference']['gpu_memory_utilization'] = protocol['gpu_memory_utilization']
    for key in ['episodes', 'logical_attempt_cap', 'unique_generation_cap']:
        plan['development'][key] //= len(parent['models'])
    return plan


def parent_files(archive):
    protocol = yaml.safe_load(PROTOCOL.read_text())
    archive = Path(archive)
    if (archive.stat().st_size != protocol['parent_archive_bytes']
            or original.digest(archive) != protocol['parent_archive_sha256']):
        raise ValueError('Parent archive identity mismatch')
    with tarfile.open(archive) as tar:
        # Read metadata as bytes; never extract or import archived candidate programs.
        files = {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}
    controller = protocol['parent_controller'] + '/'
    seal = json.loads(files[controller + 'source-seal.json'])
    if seal != original.source_seal():
        raise ValueError('Original v6 runtime changed')
    for name, sha in seal.items():
        if hashlib.sha256(files[name]).hexdigest() != sha:
            raise ValueError('Archived source identity mismatch: ' + name)
    prep = protocol['parent_prepared'] + '/'
    receipt = json.loads(files[prep + 'receipt.json'])
    for name, sha in receipt['files'].items():
        if hashlib.sha256(files[prep + name]).hexdigest() != sha:
            raise ValueError('Parent prepared file changed: ' + name)
    episodes = [json.loads(line) for line in files[controller + 'repairs/episodes.jsonl'].splitlines()]
    nodes = [json.loads(line) for line in files[controller + 'repairs/nodes.jsonl'].splitlines()]
    if (len(episodes) != 12 or len(nodes) != 37
            or any(row['model'] != 'gemma-4-12B-it' for row in episodes + nodes)
            or any('/gemma-4-31B-it/' in name and name.endswith('.jsonl') for name in files)):
        raise ValueError('Cannot resume an arm that already generated candidates')
    return protocol, {name: files[prep + name] for name in ['config.yaml', 'receipt.json', 'tasks.json', 'models.json']}


def prepare(archive, out):
    protocol, files = parent_files(archive)
    plan = resumed_plan(yaml.safe_load(files['config.yaml']), protocol)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'parent-evidence.tgz').write_bytes(Path(archive).read_bytes())
    for name in ['tasks.json', 'models.json']:
        (out / name).write_bytes(files[name])
    (out / 'config.yaml').write_text(yaml.safe_dump(plan, sort_keys=False))
    original.save(out / 'continuation.json', {
        'parent_archive_sha256': protocol['parent_archive_sha256'],
        'parent_receipt_sha256': hashlib.sha256(files['receipt.json']).hexdigest(),
        'source_sha256': source_seal(), 'model': protocol['resume_model'],
        'completed_12B_preserved': True, 'parent_31B_candidates': 0,
        'GPU_validation': 'pending', 'utc': original.utc()})
    receipt = json.loads(files['receipt.json'])
    receipt.update(status='continuation_CPU_prepared_GPU_validation_pending', utc=original.utc(),
        files={name: original.digest(out / name) for name in
               ['config.yaml', 'tasks.json', 'models.json', 'continuation.json', 'parent-evidence.tgz']})
    original.save(out / 'receipt.json', receipt)
    return receipt


def load_prepared(path):
    path = Path(path)
    result = original.load_prepared(path)
    protocol, files = parent_files(path / 'parent-evidence.tgz')
    continuation = json.loads((path / 'continuation.json').read_text())
    if continuation['source_sha256'] != source_seal():
        raise ValueError('Continuation source changed')
    if result[0] != resumed_plan(yaml.safe_load(files['config.yaml']), protocol):
        raise ValueError('Unexpected continuation protocol changes')
    for name in ['tasks.json', 'models.json']:
        if (path / name).read_bytes() != files[name]:
            raise ValueError('Continuation changed parent ' + name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare(args.parent_archive, args.out)
    print(json.dumps({'status': receipt['status'], 'prepared': str(args.out)}))


if __name__ == '__main__':
    main()
