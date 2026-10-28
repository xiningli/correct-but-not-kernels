"""D1/D2: batched development repair generation; candidate code never executes."""
import argparse
import json
from pathlib import Path
import shutil

from research.diagnostics_v1.common import ROOT, digest, object_hash
from research.diagnostics_v1.prepare import load_prepared
from research.diagnostics_v1.protocol import entries, validate_entries


def check_context(tokenizer, prompts, max_output, max_context):
    lengths = [len(tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False)) for m in prompts]
    if any(n + max_output > max_context for n in lengths):
        raise ValueError(f'Context overflow before generation: prompt={max(lengths)}, output={max_output}, context={max_context}')
    return lengths


def check_source_budget(tokenizer, tasks, max_output):
    lengths = {k: len(tokenizer.encode('```python\n' + t['source'] + '\n```',
                                      add_special_tokens=False)) for k, t in tasks.items()}
    if any(n + 1 > max_output for n in lengths.values()):
        raise ValueError('Original program exceeds output cap; do not confound repair with forced compression')
    return lengths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    request = json.loads(a.request.read_text())
    if request['scope'] != 'two_development_defects_only':
        raise ValueError('This sampler cannot generate on formal tests')
    plan, tasks, policies = load_prepared(request['prepared'])
    diagnostic_path = Path(request['visible_diagnostics'])
    if digest(diagnostic_path) != request['visible_diagnostics_sha256']:
        raise ValueError('Visible diagnostic changed')
    diagnostics = json.loads(diagnostic_path.read_text())
    expected = entries(tasks, diagnostics, plan)
    validate_entries(request['entries'], expected)
    policy = policies[request['policy']]
    if any(shutil.which(x) is None for x in ['ninja', 'nvcc']):
        raise RuntimeError('Use scripts/gpu_env.sh with the pinned compiler PATH')
    adapter = policy['adapter']
    if adapter:
        path = (ROOT / adapter).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Adapter path escapes project')
        for name, field in [('adapter_model.safetensors', 'adapter_sha256'),
                            ('adapter_config.json', 'adapter_config_sha256')]:
            if digest(path / name) != policy[field]:
                raise ValueError('Adapter hash mismatch')
    from transformers import AutoTokenizer
    model = ROOT / 'models' / policy['model']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    check_source_budget(tokenizer, tasks, plan['sampling']['max_tokens'])
    lengths = check_context(tokenizer, [e['messages'] for e in expected],
                            plan['sampling']['max_tokens'], plan['context']['max_model_len'])
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(model), dtype='bfloat16', max_model_len=plan['context']['max_model_len'],
              seed=plan['seeds']['generation'], gpu_memory_utilization=0.8, max_num_seqs=4,
              trust_remote_code=False, enable_lora=bool(adapter), max_lora_rank=32,
              limit_mm_per_prompt={'image': 0, 'audio': 0, 'video': 0})
    lora = None
    if adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest(policy['label'], 1, str((ROOT / adapter).resolve()))
    with a.out.open('x') as stream:
        for start in range(0, len(expected), 4):
            batch = expected[start:start + 4]
            params = [SamplingParams(n=1, seed=e['sampling_seed'], **plan['sampling']) for e in batch]
            outputs = llm.chat([e['messages'] for e in batch], params, lora_request=lora,
                              chat_template_kwargs={'enable_thinking': False}, use_tqdm=False)
            for entry, output in zip(batch, outputs, strict=True):
                candidate, = output.outputs
                if len(output.prompt_token_ids) + plan['sampling']['max_tokens'] > plan['context']['max_model_len']:
                    raise RuntimeError('Backend prompt accounting differs from preflight')
                row = {k: v for k, v in entry.items() if k != 'messages'}
                row.update(policy=policy['label'], text=candidate.text, token_ids=list(candidate.token_ids),
                           new_tokens=len(candidate.token_ids), prompt_tokens=len(output.prompt_token_ids),
                           finish_reason=candidate.finish_reason)
                stream.write(json.dumps(row) + '\n'); stream.flush()
            print(json.dumps({'generated': min(start + 4, len(expected)), 'total': len(expected)}), flush=True)


if __name__ == '__main__':
    main()
