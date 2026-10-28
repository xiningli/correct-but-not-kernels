"""Generate one round, then exit so candidate execution has an exclusive GPU."""
import argparse
import json
from pathlib import Path
from research.diagnostics_v2.common import ROOT, load_prepared, verify_model, object_hash
from research.diagnostics_v1.sample import check_context, check_source_budget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); r = json.loads(a.request.read_text())
    plan, tasks, models = load_prepared(r['prepared'])
    if r['scope'] != 'two_development_defects_only' or r['model'] not in models:
        raise ValueError('Invalid generation scope')
    entries = r['entries']
    if not entries or len({x['id'] for x in entries}) != len(entries):
        raise ValueError('Invalid request grid')
    for x in entries:
        if (x['task'] not in tasks or x['round'] not in range(plan['development']['rounds'])
                or x['prompt_sha256'] != object_hash(x['messages'])):
            raise ValueError('Prompt identity mismatch')
    model = verify_model(models[r['model']])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    check_source_budget(tokenizer, tasks, plan['sampling']['max_tokens'])
    check_context(tokenizer, [x['messages'] for x in entries], plan['sampling']['max_tokens'],
                  plan['inference']['max_model_len'])
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(model), dtype=plan['model_dtype'],
        max_model_len=plan['inference']['max_model_len'],
        gpu_memory_utilization=plan['inference']['gpu_memory_utilization'],
        max_num_seqs=plan['inference']['max_num_seqs'], enforce_eager=plan['inference']['enforce_eager'],
        seed=plan['seeds']['generation'], trust_remote_code=False,
        limit_mm_per_prompt={'image': 0, 'audio': 0, 'video': 0})
    with a.out.open('x') as stream:
        for start in range(0, len(entries), 2):
            batch = entries[start:start + 2]
            params = [SamplingParams(n=1, seed=x['sampling_seed'], **plan['sampling']) for x in batch]
            outputs = llm.chat([x['messages'] for x in batch], params,
                              chat_template_kwargs={'enable_thinking': plan['thinking']}, use_tqdm=False)
            for entry, output in zip(batch, outputs, strict=True):
                candidate, = output.outputs
                if len(output.prompt_token_ids) + plan['sampling']['max_tokens'] > plan['inference']['max_model_len']:
                    raise ValueError('Backend context exceeds preflight cap')
                row = {k: v for k, v in entry.items() if k != 'messages'}
                row.update(text=candidate.text, token_ids=list(candidate.token_ids),
                    prompt_tokens=len(output.prompt_token_ids), new_tokens=len(candidate.token_ids),
                    finish_reason=candidate.finish_reason)
                stream.write(json.dumps(row) + '\n'); stream.flush()
            print(json.dumps({'generated': min(start + 2, len(entries)), 'total': len(entries)}), flush=True)


if __name__ == '__main__':
    main()
