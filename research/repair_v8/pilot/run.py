"""Paired round scheduler, durable attempts, then frozen-patch hidden judging."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

from research.repair_v8.prepare import sha
from .artifacts import write_json
from research.repair_v8.supervise import stop_tree
from .data import TASKS
from .editor import EditError, apply_response, build_scope
from .evaluate import evaluate, upstream_run
from .isolation import memory_reader
from .prepare import load, tokenizer, PROJECT
from .protocol import messages, object_hash, seed_for, feedback_with_token_cap, error_signature
from .sample import rows


def sealed_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Resume identity changed: ' + str(path))
    else:
        write_json(path, value)


def cached_evaluate(prepared, task, source, panel, path, config, runtime, *, sanitizer=False):
    identity = {'task': task, 'source_sha256': object_hash(source), 'panel': panel,
                'preparation_sha256': sha(prepared / 'manifest.json'), 'sanitizer': sanitizer}
    bind = path.with_suffix('.binding.json')
    sealed_json(bind, identity)
    if (path / 'result.json').exists():
        return json.loads((path / 'result.json').read_text())
    if path.exists():
        # Interrupted execution retries the identical saved patch/cases only.
        # Preserve partial logs and explicitly count this as infrastructure work.
        index = len(list(path.parent.glob(path.name + '.interrupted-*')))
        path.rename(path.with_name(path.name + f'.interrupted-{index}'))
    return evaluate(prepared, task, source, panel, path, config, runtime=runtime, sanitizer=sanitizer)


def generate(prepared, entries, directory, inference_python, model, config):
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / 'request.json'; output = directory / 'generations.jsonl'
    sealed_json(request, {'preparation_sha256': sha(prepared / 'manifest.json'), 'entries': entries})
    if not entries:
        return {}
    saved = rows(output)
    expected = {x['id']: x for x in entries}
    if set(saved) - set(expected):
        raise ValueError('Generation file contains unknown requests')
    if set(saved) != set(expected):
        memory = memory_reader(); before = memory()
        env = os.environ.copy()
        env.update(PYTHONPATH=str(PROJECT), PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   VLLM_USE_FLASHINFER_SAMPLER='0', OMP_NUM_THREADS='2',
                   PATH=str(inference_python.parent) + ':/usr/local/cuda/bin:/usr/bin:/bin')
        index = len(list(directory.glob('generation-*.log')))
        start = time.monotonic()
        with (directory / f'generation-{index}.log').open('x') as stream:
            child = subprocess.Popen([str(inference_python), '-u', '-m', 'research.repair_v8.pilot.sample',
                '--prepared', str(prepared), '--request', str(request), '--out', str(output), '--model', str(model)],
                cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=config['generation_round_seconds'])
            except BaseException:
                stop_tree(child)
                raise
            finally:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        write_json(directory / f'cost-{index}.json', {'seconds': time.monotonic() - start, 'returncode': code,
                   'request_sha256': sha(request), 'new_physical_generations': len(rows(output)) - len(saved)})
        if code:
            raise RuntimeError('Generation failed; saved completions retained: ' + str(directory))
        deadline = time.monotonic() + 30
        while memory() > before + (512 << 20):
            if time.monotonic() >= deadline:
                raise RuntimeError('Inference GPU allocations remain; refuse evaluation overlap')
            time.sleep(.5)
    saved = rows(output)
    if set(saved) != set(expected):
        raise ValueError('Incomplete generation batch')
    for key, record in saved.items():
        for field, value in expected[key].items():
            if field != 'messages' and record[field] != value:
                raise ValueError('Saved generation identity drift')
    return saved


def execute(prepared, controls_path, out, inference_python, model):
    manifest, config = load(prepared)
    controls = json.loads(controls_path.read_text())
    if controls['status'] != 'PASS' or controls['preparation_sha256'] != sha(prepared / 'manifest.json'):
        raise ValueError('Need passing controls from this exact preparation')
    out.mkdir(parents=True, exist_ok=True)
    sealed_json(out / 'bindings.json', {'prepared_sha256': sha(prepared / 'manifest.json'),
                'controls_sha256': sha(controls_path), 'model_path': str(model), 'inference_python': str(inference_python)})
    tok = tokenizer(prepared / 'tokenizer')
    public = {t: json.loads((prepared / 'public' / (t + '.json')).read_text()) for t in config['tasks']}
    for task in public:
        public[task]['allowed_host_ops'] = controls['tasks'][task].get('allowed_ops', [])
        public[task]['allowed_memcheck_host_ops'] = controls['tasks'][task].get('allowed_memcheck_ops', [])
    originals = {t: {TASKS[t]['path']: (prepared / 'modules' / t / 'before.py').read_text()} for t in config['tasks']}
    scopes = {t: {p: build_scope(s, TASKS[t]['protected']) for p, s in files.items()} for t, files in originals.items()}
    states = {}
    for task in config['tasks']:
        for replica in range(config['replicas']):
            for arm in config['arms']:
                key = f'{task}-r{replica}-{arm}'
                states[key] = {'task': task, 'replica': replica, 'arm': arm, 'source': originals[task],
                               'feedback': controls['tasks'][task]['before_visible'], 'done': False, 'attempts': []}
    for step in range(config['attempts']):
        load(prepared)
        batch, associations, overflows = {}, {}, {}
        active = [key for key, state in states.items() if not state['done']]
        if not active:
            break
        for key in active:
            state = states[key]; arm = config['arms'][state['arm']]; task = state['task']
            feedback = feedback_with_token_cap(state['feedback'], step > 0 and arm['diagnostics'], tok, config['max_detail_tokens'])
            prompt = messages(public[task], state['source'], arm['interface'], feedback)
            ids = tok.apply_chat_template(prompt, add_generation_prompt=True, enable_thinking=False)
            physical = f'{task}-r{state["replica"]}-{arm["interface"] if step == 0 else state["arm"]}-s{step}'
            associations[key] = physical
            if len(ids) > config['max_prompt_tokens']:
                overflows[key] = True
                continue
            entry = {'id': physical, 'task': task, 'step': step, 'interface': arm['interface'], 'replica': state['replica'],
                     'messages': prompt, 'prompt_sha256': object_hash(prompt), 'prompt_token_ids_sha256': object_hash(ids),
                     'sampling_seed': seed_for(config['seed_namespace'], task, state['replica'], step)}
            if physical in batch and batch[physical] != entry:
                raise ValueError('Paired first prompts are not identical')
            batch[physical] = entry
        order = sorted(batch)
        random.Random(seed_for(config['seed_namespace'], 'order', 0, step)).shuffle(order)
        generated = generate(prepared, [batch[k] for k in order], out / f'round-{step}', inference_python, model, config)
        eval_order = list(active)
        random.Random(seed_for(config['seed_namespace'], 'evaluation-order', 0, step)).shuffle(eval_order)
        for key in eval_order:
            state = states[key]; task = state['task']; physical = associations[key]
            attempt_path = out / 'attempts' / key / f'{step}.json'
            binding = {'source_before_sha256': object_hash(state['source']), 'physical_generation': physical}
            if attempt_path.exists():
                attempt = json.loads(attempt_path.read_text())
                if attempt['binding'] != binding:
                    raise ValueError('Attempt resume state mismatch')
            else:
                attempt = {'binding': binding, 'applied': False, 'visible_pass': False,
                           'source': state['source'], 'step': step, 'physical_generation': physical,
                           'prompt_tokens': 0, 'output_tokens': 0, 'generation_seconds': 0.}
                try:
                    if key in overflows:
                        raise EditError('context_limit')
                    generation = generated[physical]
                    for metric in ['prompt_tokens', 'output_tokens', 'generation_seconds']:
                        attempt[metric] = generation[metric]
                    if generation['finish_reason'] != 'stop':
                        raise EditError('generation_did_not_stop')
                    candidate = apply_response(state['source'], scopes[task], generation['text'], config['arms'][state['arm']]['interface'])
                    attempt.update(applied=True, source=candidate)
                    result = cached_evaluate(prepared, task, candidate, 'visible', out / 'visible' / physical,
                                             config, controls['tasks'][task])
                    attempt['visible_pass'] = result['passed']
                    attempt['feedback'] = result
                except EditError as e:
                    attempt['feedback'] = {'component': 'format', 'editor_error': str(e), 'tests_observed': 0, 'tests_passed': 0, 'passed': False, 'cases': []}
                write_json(attempt_path, attempt)
            state['source'] = attempt['source']; state['feedback'] = attempt['feedback']
            state['done'] = attempt['visible_pass']; state['attempts'].append(attempt)
            write_json(out / 'progress.json', {'phase': 'repair', 'step': step, 'last_episode': key,
                        'visible_pass_episodes': sum(s['done'] for s in states.values())})
    # Freeze EVERY selected patch before inspecting ANY hidden candidate score.
    selected = {key: {'task': state['task'], 'arm': state['arm'], 'replica': state['replica'],
                     'source': state['source'], 'source_sha256': object_hash(state['source']), 'visible_pass': state['done'],
                     'attempts': len(state['attempts']), 'physical_generations': [a['physical_generation'] for a in state['attempts']],
                     'logical_prompt_tokens': sum(a['prompt_tokens'] for a in state['attempts']),
                     'logical_output_tokens': sum(a['output_tokens'] for a in state['attempts']),
                     'logical_generation_seconds': sum(a['generation_seconds'] for a in state['attempts']),
                     'logical_visible_seconds': sum(a['feedback'].get('seconds', 0.) for a in state['attempts']),
                     'failure_signatures': [error_signature(a['feedback']) for a in state['attempts']]} for key, state in states.items()}
    sealed_json(out / 'selected-patches.json', selected)
    selected_hash = sha(out / 'selected-patches.json')
    for key in sorted(selected):
        state = selected[key]; task = state['task']
        path = out / 'judging' / key
        if (path / 'final.json').exists():
            if json.loads((path / 'final.json').read_text())['selected_manifest_sha256'] != selected_hash:
                raise ValueError('Judging selection changed')
            continue
        hidden = cached_evaluate(prepared, task, state['source'], 'hidden', path / 'hidden', config, controls['tasks'][task])
        # Run regression panels even when visible tests fail: this supports the
        # diagnostic distinction between no repair and newly broken behavior.
        suite_path = path / 'upstream'
        if (suite_path / 'upstream-result.json').exists():
            suite = json.loads((suite_path / 'upstream-result.json').read_text())
        else:
            if suite_path.exists():
                suite_path.rename(suite_path.with_name('upstream.interrupted-' + str(time.time_ns())))
            if (path / 'hidden/checkout').exists():
                suite = upstream_run(task, path / 'hidden/checkout', suite_path, config['upstream_seconds'],
                                     controls['tasks'][task]['upstream_passed_nodeids'])
            else:
                suite = {'passed': False, 'not_run': 'static_contract_rejection'}
        # Sanitizer is required to award success, but cannot rescue a failed patch.
        memory = None
        if state['visible_pass'] and hidden['passed'] and suite['passed']:
            memory = cached_evaluate(prepared, task, state['source'], 'hidden', path / 'memcheck', config, controls['tasks'][task], sanitizer=True)
        passed_before = set(controls['tasks'][task]['before_hidden_pass_ids'])
        regressed = [x['spec']['id'] + '/' + str(x['seed']) for x in hidden['cases']
                     if x['spec']['id'] + '/' + str(x['seed']) in passed_before and not x['passed']]
        semantic = [x['spec']['id'] + '/' + str(x['seed']) for x in hidden['cases']
                    if x['spec']['id'] + '/' + str(x['seed']) in passed_before
                    and x.get('worker', {}).get('status') == 'complete'
                    and (any(not c['passed'] for c in x['checks']) or
                         set(x['flags']) & {'guard_storage_modified', 'outplace_input_modified', 'inplace_alias_contract'})]
        final = {'episode': key, 'task': task, 'arm': state['arm'], 'replica': state['replica'],
                 'selected_manifest_sha256': selected_hash, 'visible_pass': state['visible_pass'],
                 'hidden_pass': hidden['passed'], 'upstream_pass': suite['passed'], 'memory_pass': bool(memory and memory['passed']),
                 'success': bool(state['visible_pass'] and hidden['passed'] and suite['passed'] and memory and memory['passed']),
                 'before_passing_cases_now_failed': regressed,
                 'observed_semantic_regressions': semantic,
                 'limitation': 'Observed case failures include non-executable patches; not all are semantic regressions.'}
        write_json(path / 'final.json', final)
        write_json(out / 'progress.json', {'phase': 'hidden_judging', 'last_episode': key})
    from .report import report
    report(out, config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'controls', 'out', 'inference-python', 'model']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); execute(a.prepared.resolve(), a.controls.resolve(), a.out.resolve(), a.inference_python.absolute(), a.model.resolve())


if __name__ == '__main__':
    main()
