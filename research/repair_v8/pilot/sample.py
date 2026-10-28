"""One generation round in the existing inference venv; exit before evaluation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

from research.repair_v8.prepare import sha
from .artifacts import write_json
from .prepare import load, PROJECT
from .protocol import object_hash


def rows(path):
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['id'] in result:
            raise ValueError('Duplicate generation identity')
        result[row['id']] = row
    return result


def verify_model(model):
    manifest = json.loads((PROJECT / 'experiments/gemma31b-weights-v6.json').read_text())
    for row in manifest['files']:
        p = model / row['path']
        if not p.is_file() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Wrong/incomplete pinned model: ' + row['path'])
    return manifest['hf_revision']


def sample(prepared, request_path, out, model):
    _, config = load(prepared)
    request = json.loads(request_path.read_text())
    if request['preparation_sha256'] != sha(prepared / 'manifest.json'):
        raise ValueError('Request is not bound to this preparation')
    versions = {p: importlib.metadata.version(p) for p in config['inference_packages']}
    if any(versions[p].split('+')[0] != version for p, version in config['inference_packages'].items()):
        raise ValueError('Existing inference venv differs from pinned versions: ' + str(versions))
    entries = request['entries']
    if len({r['id'] for r in entries}) != len(entries):
        raise ValueError('Duplicate generation request')
    for row in entries:
        if row['prompt_sha256'] != object_hash(row['messages']):
            raise ValueError('Prompt checksum mismatch')
    saved = rows(out)
    if set(saved) - {r['id'] for r in entries}:
        raise ValueError('Unknown cached generation')
    for entry in entries:
        if entry['id'] in saved and saved[entry['id']]['prompt_sha256'] != entry['prompt_sha256']:
            raise ValueError('Resume would change a saved prompt')
    pending = [r for r in entries if r['id'] not in saved]
    if not pending:
        return
    revision = verify_model(model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False, local_files_only=True)
    for entry in pending:
        tokens = tok.apply_chat_template(entry['messages'], add_generation_prompt=True, enable_thinking=False)
        if len(tokens) > config['max_prompt_tokens'] or object_hash(tokens) != entry['prompt_token_ids_sha256']:
            raise ValueError('Inference and preparation tokenizer disagree or prompt exceeds cap')
    from vllm import LLM, SamplingParams
    start = time.monotonic()
    llm = LLM(model=str(model), dtype=config['dtype'], **config['inference'], trust_remote_code=False,
              seed=20261019, limit_mm_per_prompt={'image': 0, 'audio': 0, 'video': 0})
    setup = time.monotonic() - start
    # A run may resume a partially completed physical round. Never regenerate
    # existing completions; retain the original setup receipt and add a segment.
    index = len(list(out.parent.glob(out.stem + '.engine-*.json')))
    write_json(out.with_name(out.stem + f'.engine-{index}.json'), {
        'versions': versions, 'model_revision': revision, 'engine_setup_seconds': setup,
        'request_sha256': sha(request_path), 'pending_ids': [r['id'] for r in pending], 'pid': os.getpid()})
    with out.open('a') as stream:
        for entry in pending:
            started = time.monotonic()
            params = SamplingParams(n=1, seed=entry['sampling_seed'], **config['sampling'])
            response, = llm.chat([entry['messages']], params, chat_template_kwargs={'enable_thinking': False}, use_tqdm=False)
            candidate, = response.outputs
            record = {k: v for k, v in entry.items() if k != 'messages'}
            record.update(text=candidate.text, token_ids=list(candidate.token_ids),
                          prompt_tokens=len(response.prompt_token_ids), output_tokens=len(candidate.token_ids),
                          finish_reason=candidate.finish_reason, generation_seconds=time.monotonic() - started)
            if object_hash(list(response.prompt_token_ids)) != entry['prompt_token_ids_sha256']:
                raise ValueError('vLLM changed prompt tokenization')
            stream.write(json.dumps(record, ensure_ascii=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
            print(json.dumps({'generated': entry['id'], 'output_tokens': record['output_tokens']}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ['prepared', 'request', 'out', 'model']:
        p.add_argument('--' + flag, type=Path, required=True)
    a = p.parse_args(); sample(a.prepared.resolve(), a.request, a.out, a.model.resolve())


if __name__ == '__main__':
    main()
