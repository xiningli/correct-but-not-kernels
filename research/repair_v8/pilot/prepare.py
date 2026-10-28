"""Portable three-task package and CPU preflight. No model/GPU execution."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil

from research.repair_v8.prepare import sha
from .artifacts import write_json
from research.repair_v8.qualify import copy_snapshot, tree_manifest, verify_files
from .data import TASKS, SEEDS, make_case, public_task
from .editor import build_scope, validate, function_text, function_nodes, apply_response, digest
from .protocol import messages, canonical, object_hash

PROJECT = Path(__file__).resolve().parents[3]


def tokenizer(directory):
    from transformers import PreTrainedTokenizerFast
    cfg = json.loads((directory / 'tokenizer_config.json').read_text())
    tok = PreTrainedTokenizerFast(tokenizer_file=str(directory / 'tokenizer.json'),
                                 **{k: cfg[k] for k in ['bos_token', 'eos_token', 'pad_token', 'unk_token']})
    tok.chat_template = (directory / 'chat_template.jinja').read_text()
    return tok


def code_files():
    paths = list((PROJECT / 'research/repair_v8/pilot').glob('*.py'))
    paths += [PROJECT / p for p in [
        'research/repair_v8/__init__.py', 'research/repair_v8/prepare.py', 'research/repair_v8/qualify.py',
        'research/repair_v8/cases.py', 'research/repair_v8/supervise.py',
        'env/__init__.py', 'env/hacks/__init__.py', 'env/hacks/detectors.py', 'research/reliability_v2/strict_contract.py',
        'experiments/cuda-repair-v8-pilot.json', 'experiments/gemma31b-weights-v6.json',
        'experiments/repair-v8-qualification.lock', 'scripts/repair_v8_pilot_autodl.sh']]
    paths += sorted((PROJECT / 'tests/hostile').glob('*.py'))
    paths += [PROJECT / 'tests/test_repair_v8_pilot.py', PROJECT / 'tests/test_repair_v8_qualification.py',
              PROJECT / 'tests/test_repair_v8_supervise.py']
    return sorted(paths)


def read_context(root, task):
    if task == 'liger_attnres':
        path = 'src/liger_kernel/transformers/functional.py'
        return {path + ':liger_attn_res': function_text((root / path).read_text(), 'liger_attn_res')}
    if task == 'liger_groupnorm':
        path = 'src/liger_kernel/transformers/group_norm.py'
        return {path: (root / path).read_text()}
    return {}


def load(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    verify_files(PROJECT, manifest['code'])
    for key, info in manifest['tasks'].items():
        verify_files(root / 'snapshots' / key / 'before', info['files'])
        for rev in ['before', 'fixed']:
            path = root / 'modules' / key / (rev + '.py')
            if sha(path) != info[rev + '_module_sha256']:
                raise ValueError('Prepared module changed')
    for row in manifest['assets']:
        if sha(root / row['path']) != row['sha256']:
            raise ValueError('Prepared asset changed: ' + row['path'])
    return manifest, json.loads((root / 'config.json').read_text())


def prepare(qualification, root, tokenizer_path):
    config = json.loads((PROJECT / 'experiments/cuda-repair-v8-pilot.json').read_text())
    root.mkdir(parents=True, exist_ok=False)
    token_dir = root / 'tokenizer'; token_dir.mkdir()
    weight_manifest = json.loads((PROJECT / 'experiments/gemma31b-weights-v6.json').read_text())
    for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']:
        expected = next(r for r in weight_manifest['files'] if r['path'] == name)
        if sha(tokenizer_path / name) != expected['sha256']:
            raise ValueError('Tokenizer identity mismatch')
        shutil.copy2(tokenizer_path / name, token_dir / name)
    tok = tokenizer(token_dir)
    tasks, checks, asset_paths = {}, [], [root / 'config.json']
    write_json(root / 'config.json', config)
    from .evaluate import source_flags
    for key in config['tasks']:
        row = TASKS[key]
        receipt = json.loads((qualification / 'source-receipts' / (row['slug'] + '-before.json')).read_text())
        source_root = qualification / 'sources' / row['slug'] / 'before'
        verify_files(source_root, receipt['files'])
        snapshot = root / 'snapshots' / key / 'before'
        copy_snapshot(source_root, snapshot, receipt['files'])
        original = (snapshot / row['path']).read_text()
        fixed_source = qualification / 'sources' / row['slug'] / 'fixed'
        fixed_receipt = json.loads((qualification / 'source-receipts' / (row['slug'] + '-fixed.json')).read_text())
        verify_files(fixed_source, fixed_receipt['files'])
        fixed = (fixed_source / row['path']).read_text()
        scope = build_scope(original, row['protected']); validate(scope, fixed)
        if source_flags(original, original) or source_flags(fixed, original):
            raise ValueError('Trusted implementation violates new execution contract: ' + key)
        original_nodes, fixed_nodes = function_nodes(ast.parse(original)), function_nodes(ast.parse(fixed))
        operations = [{'op': 'replace_function', 'name': name, 'source': function_text(fixed, name)}
                      for name in original_nodes if ast.dump(original_nodes[name]) != ast.dump(fixed_nodes[name])]
        full = canonical({'files': [{'path': row['path'], 'expected_sha256': digest(original), 'source': fixed}]})
        edits = canonical({'files': [{'path': row['path'], 'expected_sha256': digest(original), 'operations': operations}]})
        produced = apply_response({row['path']: original}, {row['path']: scope}, edits, 'atomic_edits')[row['path']]
        if ast.dump(ast.parse(produced)) != ast.dump(ast.parse(fixed)):
            raise ValueError('Official fix cannot be expressed by this atomic surface: ' + key)
        apply_response({row['path']: original}, {row['path']: scope}, full, 'full_module')
        for rev, code in [('before', original), ('fixed', fixed)]:
            path = root / 'modules' / key / (rev + '.py'); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(code)
        public = public_task(key); public['readonly_context'] = read_context(snapshot, key)
        path = root / 'public' / (key + '.json'); write_json(path, public); asset_paths.append(path)
        prompt_checks = {}
        for interface in ['full_module', 'atomic_edits']:
            prompt = messages(public, {row['path']: original}, interface,
                              {'component': 'incorrect', 'tests_observed': 2, 'tests_passed': 1})
            tokens = tok.apply_chat_template(prompt, add_generation_prompt=True, enable_thinking=False)
            if len(tokens) > config['max_prompt_tokens']:
                raise ValueError('Prompt exceeds common cap')
            prompt_checks[interface] = {'tokens': len(tokens), 'token_ids_sha256': object_hash(tokens)}
        original_wire = canonical({'files': [{'path': row['path'], 'expected_sha256': digest(original), 'source': original}]})
        original_tokens = len(tok.encode(original_wire, add_special_tokens=False))
        if original_tokens > config['max_original_response_tokens'] or len(tok.encode(full, add_special_tokens=False)) > config['sampling']['max_tokens']:
            raise ValueError('Full-module response cap would unfairly exclude a valid fix')
        oracle_cases = 0
        for panel in ['visible', 'hidden']:
            for seed in SEEDS[panel]:
                for spec in row[panel]:
                    make_case(spec, seed); oracle_cases += 1
        tasks[key] = {'repo': row['repo'], 'before_commit': receipt['commit'], 'fixed_commit': fixed_receipt['commit'],
                      'files': tree_manifest(snapshot), 'before_module_sha256': digest(original), 'fixed_module_sha256': digest(fixed)}
        checks.append({'task': key, 'oracle_cases': oracle_cases, 'official_fix_interface_parity': True,
                       'original_full_response_tokens': original_tokens, 'prompts': prompt_checks,
                       'formal_task': False})
    assets = [*asset_paths, *token_dir.iterdir()]
    manifest = {'schema': 'v8-mini-preparation-v1', 'scope': config['scope'], 'tasks': tasks,
                'code': [{'path': str(p.relative_to(PROJECT)), 'sha256': sha(p)} for p in code_files()],
                'assets': [{'path': str(p.relative_to(root)), 'sha256': sha(p)} for p in sorted(assets)],
                'model_manifest_sha256': sha(PROJECT / 'experiments/gemma31b-weights-v6.json')}
    write_json(root / 'manifest.json', manifest)
    report = {'status': 'CPU_preflight_pass_GPU_gates_pending', 'checks': checks,
              'logical_episodes': len(TASKS) * len(config['arms']) * config['replicas'],
              'max_logical_attempts': len(TASKS) * len(config['arms']) * config['replicas'] * config['attempts'],
              'model_runs': 0, 'GPU_runs': 0, 'formal_tasks': 0, 'training_runs': 0}
    write_json(root / 'cpu-preflight.json', report)
    load(root)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.qualification.resolve(), a.root.resolve(), a.tokenizer.resolve()), indent=2))


if __name__ == '__main__':
    main()
