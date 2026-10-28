"""K1/K4: text-only vLLM backend for hashed DEVELOPMENT prompts.

Derived from tools/sample_policy.py, without importing or executing candidates.
Only separate development tasks are permitted by this initial implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from research.reliability_v2.new_tasks import catalog
from research.reliability_v2.prepare_transfer import messages
from research.reliability_v2.repair_protocol import sampling_seed
from tools.gatelib import ROOT
from tools.replication import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    request = json.loads(a.request.read_text())
    if request.get('scope') != 'development_only':
        raise ValueError('This backend cannot sample the unsealed test candidates')
    if shutil.which('ninja') is None or shutil.which('nvcc') is None:
        raise RuntimeError('Launch with scripts/gpu_env.sh so the pinned compiler tools are on PATH')
    dev = {t['id']: t for t in catalog() if t['split'] == 'development'}
    entries = request['entries']
    if not entries or any(e['task'] not in dev for e in entries):
        raise ValueError('A non-development task reached the development sampler')
    for e in entries:
        prompt = messages(dev[e['task']], e['condition'])
        if (e['messages'] != prompt or e['prompt_sha256'] != hashlib.sha256(
                json.dumps(prompt, sort_keys=True).encode()).hexdigest()):
            raise ValueError('Prompt drift')
    policy = request['policy']
    adapter = policy['adapter']
    if adapter:
        path = ROOT / adapter
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Adapter escapes project')
        for name, field in [('adapter_model.safetensors', 'adapter_sha256'), ('adapter_config.json', 'adapter_config_sha256')]:
            if sha256(path / name) != policy[field]:
                raise ValueError('Adapter drift: ' + name)
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(ROOT / 'models' / policy['model']), dtype='bfloat16',
              max_model_len=8192, seed=request['namespace'], gpu_memory_utilization=0.8,
              max_num_seqs=4, trust_remote_code=False, enable_lora=bool(adapter), max_lora_rank=32,
              limit_mm_per_prompt={'image': 0, 'audio': 0, 'video': 0})
    lora = None
    if adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest(policy['label'], 1, str((ROOT / adapter).resolve()))
    with a.out.open('x') as stream:
        for start in range(0, len(entries), 4):
            batch = entries[start:start + 4]
            seeds = [sampling_seed(request['namespace'], e['task'] + '/' + e['condition'], 0, 0) for e in batch]
            params = [SamplingParams(n=1, seed=s, **request['sampling']) for s in seeds]
            outputs = llm.chat([e['messages'] for e in batch], params, lora_request=lora,
                               chat_template_kwargs={'enable_thinking': False}, use_tqdm=False)
            for entry, output, seed in zip(batch, outputs, seeds, strict=True):
                item, = output.outputs
                row = {'task': entry['task'], 'condition': entry['condition'], 'policy': policy['label'],
                       'prompt_sha256': entry['prompt_sha256'], 'sampling_seed': seed,
                       'text': item.text, 'new_tokens': len(item.token_ids),
                       'prompt_tokens': len(output.prompt_token_ids), 'finish_reason': item.finish_reason}
                stream.write(json.dumps(row) + '\n'); stream.flush()
            print(json.dumps({'generated': min(start + 4, len(entries)), 'total': len(entries)}), flush=True)


if __name__ == '__main__':
    main()
