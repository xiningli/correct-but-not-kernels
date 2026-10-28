"""K1–K4: immutable, relocatable phase-2 protocol and generated job manifests."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import yaml
from env.tasks import split_ids, verified_catalog
from tools.gatelib import ROOT, init_run, require_gate, write_json

DEFAULT_PLAN = ROOT/'experiments/replication-v1.1.yaml'

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def stable_seed(namespace, task, sample=0):
    value = f'{namespace}:{task}:{sample}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], 'big') % (2**31-1)

def candidate_seeds(training_seed,count):
    """K1/K4: adjacent training seeds must not share 319/320 candidate draws."""
    return [stable_seed(training_seed,'sft-candidate',i) for i in range(count)]

def local_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Manifest paths must be repository-relative')
    return ROOT/path

def load_plan(path=DEFAULT_PLAN):
    plan = yaml.safe_load(Path(path).read_text())
    train, _ = split_ids()
    if not set(plan['train_tasks']) <= train or not plan['train_tasks']:
        raise ValueError('Training tasks must be in the unchanged TRAIN split')
    if plan['arms'] != ['C', 'S', 'SFT']:
        raise ValueError('The approved study requires C, S and SFT')
    for key, length in [('train_seeds', 3), ('evaluation_seeds', 8)]:
        values = plan[key]
        if len(values) != length or len(set(values)) != length or any(type(x) is not int for x in values):
            raise ValueError(f'{key} must contain {length} distinct integer seeds')
    if plan['fresh_checks'] < 4 or plan['max_steps'] != 40:
        raise ValueError('Protocol drift in fresh checks or optimizer budget')
    if plan['evaluation_cadence']!=10:
        raise ValueError('The preregistered intermediate evaluation cadence is ten steps')
    if set(plan['train_seeds']) & set(plan['evaluation_seeds']):
        raise ValueError('Training and evaluation seed namespaces must differ')
    if plan['sft']['candidates_per_seed'] != plan['max_steps']*8:
        raise ValueError('SFT generation budget must match 40 GRPO groups of 8')
    if plan['sft']['selection'] != 'highest_train_validation_correctness_then_lowest_learning_rate':
        raise ValueError('Undeclared SFT selection rule')
    for value in [plan['pilot'], *[x['path'] for x in plan['pilot_adapters'].values()]]:
        local_path(value)
    return plan

def source_hashes():
    files = [p for folder in ['env','train','eval','tools','tasks']
             for p in (ROOT/folder).rglob('*.py')]
    files += [ROOT/'env/heldout.json', ROOT/'env/kernelbench.lock.json', ROOT/'tasks/prompt.md']
    files += list((ROOT/'scripts').glob('*.sh'))+[ROOT/'pyproject.toml']
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}

def verify_frozen(directory):
    directory = Path(directory)
    frozen = json.loads((directory/'frozen.json').read_text())
    if sha256(directory/'plan.yaml') != frozen['plan_sha256']:
        raise ValueError('Frozen protocol changed')
    if source_hashes() != frozen['source_sha256']:
        raise ValueError('Executable source changed after freezing; create a new protocol version')
    for name, digest in frozen['generated_sha256'].items():
        if sha256(directory/name) != digest:
            raise ValueError(f'Frozen job changed: {name}')
    return load_plan(directory/'plan.yaml'), frozen

def require_replication_gate(name, directory):
    plan, frozen = verify_frozen(directory)
    record = json.loads((ROOT/require_gate(name)).read_text())
    if record.get('protocol_sha256') != sha256(Path(directory)/'frozen.json'):
        raise RuntimeError(f'{name} belongs to a different frozen study')
    return plan, frozen

def evaluation_request(plan, pilot_config, *, adapter=None, seeds=None, split='heldout', tasks=None):
    _, heldout = split_ids()
    seeds = plan['evaluation_seeds'] if seeds is None else seeds
    req = copy.deepcopy(pilot_config['evaluation'])
    req.update(model=plan['model'], adapter=adapter, tasks=sorted(heldout) if tasks is None else tasks,
        samples_per_task=len(seeds), sample_seeds=seeds, seed=seeds[0],
        input_seed=plan['input_seed'], fresh_checks=plan['fresh_checks'], split=split,
        generation_timeout_seconds=plan['generation_timeout_seconds'], record_token_ids=True,
        split_sha256=sha256(ROOT/'env/heldout.json'))
    if split=='train':req['input_seed']=stable_seed(plan['input_seed'],'train-inputs',seeds[0])
    if adapter:
        req['adapter_sha256']=sha256(local_path(adapter)/'adapter_model.safetensors')
        req['adapter_config_sha256']=sha256(local_path(adapter)/'adapter_config.json')
    return req

def freeze(plan_path, directory):
    directory=Path(directory).resolve()
    plan = load_plan(plan_path)
    require_gate('C6'); verified_catalog()
    if directory.exists():
        raise FileExistsError('Never overwrite a frozen experiment')
    pilot_config = yaml.safe_load((local_path(plan['pilot'])/'config.yaml').read_text())
    if any(cfg['tasks']!=plan['train_tasks'] or cfg['model']!=plan['model']
           for cfg in pilot_config['training'].values()):
        raise ValueError('Replication must retain the pilot training task and model')
    init_run(directory, plan, plan['analysis']['seed'])
    (directory/'plan.yaml').write_bytes(Path(plan_path).read_bytes())
    requests = directory/'requests'; requests.mkdir()
    configs = directory/'configs'; configs.mkdir()
    for label in ['base','C','S']:
        adapter = None if label == 'base' else plan['pilot_adapters'][label]['path']
        write_json(requests/f'reeval-{label}.json', evaluation_request(plan,pilot_config,adapter=adapter))
    sentinel = evaluation_request(plan,pilot_config,seeds=[plan['evaluation_seeds'][0]])
    sentinel['reference_replay'] = True
    write_json(requests/'reference-replay.json',sentinel)
    jobs=[]
    for seed in plan['train_seeds']:
        # Rotate order, retaining every arm/seed, to reduce order confounding.
        order = plan['arms'][len(jobs)//3 % 3:] + plan['arms'][:len(jobs)//3 % 3]
        for arm in order:
            cfg=copy.deepcopy(pilot_config['training']['C' if arm=='SFT' else arm])
            cfg.update(stage='R3',seed=seed,method='sft' if arm=='SFT' else 'grpo',
                run_id=f'{plan["id"]}/training/{arm}-{seed}',
                protocol=str(directory.relative_to(ROOT)))
            cfg['sft']=copy.deepcopy(plan['sft'])
            cfg['sft']['learning_rate']=None  # TRAIN-only tuning must resolve this.
            (configs/f'{arm}-{seed}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
            jobs.append({'arm':arm,'seed':seed,'config':f'configs/{arm}-{seed}.yaml'})
    write_json(directory/'jobs.json',jobs)
    generated={str(p.relative_to(directory)):sha256(p) for folder in [requests,configs]
               for p in sorted(folder.iterdir())}
    generated['jobs.json']=sha256(directory/'jobs.json')
    write_json(directory/'frozen.json',{'plan_sha256':sha256(directory/'plan.yaml'),
        'source_sha256':source_hashes(),'generated_sha256':generated,
        'pilot_config_sha256':sha256(local_path(plan['pilot'])/'config.yaml'),
        'interpretation':'New generation/training seeds on previously observed tasks; not a new test set.'})
    print(directory)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','verify'])
    p.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    p.add_argument('--out',type=Path,default=ROOT/'runs/replication-v1.1')
    a=p.parse_args()
    if a.action=='freeze':freeze(a.plan,a.out)
    else:verify_frozen(a.out);print('Frozen source and configuration verified')

if __name__=='__main__':main()
