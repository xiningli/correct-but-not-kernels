"""D1/D2: CPU preparation with identities; no model generation or GPU execution."""
import argparse
import json
from pathlib import Path

import yaml

from research.diagnostics_v1.common import PLAN, ROOT, digest, object_hash, save, source_seal, utc
from research.diagnostics_v1.tasks import CASES, fixed_control, public_task
from tools.replication import verify_frozen


def load_prepared(directory):
    directory = Path(directory)
    receipt = json.loads((directory / 'prepared.json').read_text())
    if receipt['source_sha256'] != source_seal():
        raise ValueError('Prepared source seal changed')
    for name, wanted in receipt['files'].items():
        if digest(directory / name) != wanted:
            raise ValueError('Prepared artifact changed: ' + name)
    plan = yaml.safe_load((directory / 'config.yaml').read_text())
    tasks = json.loads((directory / 'public-tasks.json').read_text())
    if tasks != {k: public_task(k) for k in CASES}:
        raise ValueError('Public source/specification drift')
    return plan, tasks, json.loads((directory / 'policies.json').read_text())


def prepare(directory, policy_path):
    verify_frozen(ROOT / 'runs/replication-v1.1')
    plan = yaml.safe_load(PLAN.read_text())
    if plan['formal_test_enabled'] or plan['training_enabled']:
        raise ValueError('Development preparation only')
    policies = {x['label']: x for x in json.loads(Path(policy_path).read_text())}
    selected = {label: policies[label] for label in plan['development']['policies']}
    for label, policy in selected.items():
        if policy['model'] != 'gemma-4-12B-it' or (label == 'base') != (policy['adapter'] is None):
            raise ValueError('Unexpected development policy')
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    tasks = {k: public_task(k) for k in CASES}
    save(out / 'public-tasks.json', tasks)
    save(out / 'policies.json', selected)
    # Private provenance is kept outside sampler inputs. This is not a test freeze.
    save(out / 'upstream-provenance.json', {k: fixed_control(k)[1] for k in CASES})
    (out / 'config.yaml').write_bytes(PLAN.read_bytes())
    receipt = {'status': 'CPU_prepared_not_GPU_validated', 'utc': utc(),
               'formal_test_enabled': False, 'independent_defects': len(tasks),
               'expected_attempts': len(tasks) * len(selected) * len(plan['development']['conditions'])
                                    * plan['development']['replicates'],
               'source_sha256': source_seal(), 'frozen_study_sha256': digest(ROOT / 'runs/replication-v1.1/frozen.json'),
               'files': {n: digest(out / n) for n in ['public-tasks.json', 'policies.json',
                         'upstream-provenance.json', 'config.yaml']}}
    if receipt['expected_attempts'] != plan['development']['expected_attempts']:
        raise ValueError('Attempt denominator changed')
    save(out / 'prepared.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--policies', type=Path, required=True)
    a = p.parse_args()
    receipt = prepare(a.out, a.policies)
    print(json.dumps({k: receipt[k] for k in ['status', 'independent_defects', 'expected_attempts']}))


if __name__ == '__main__':
    main()
