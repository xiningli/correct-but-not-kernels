"""K1–K4: single-turn vLLM sampling from a local Gemma snapshot/LoRA checkpoint.

Sampling/text-only setup adapted from glomergence gate_g4 at
e602f63207fe4e67eeda4dff7b53d49e92cc318b and this project's C0.
This process generates text only; policy programs are never executed here.
"""
import argparse
import json
from pathlib import Path
from env.tasks import verified_catalog
from tasks.prompts import messages
from tools.gatelib import ROOT

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True, help='K1–K4: frozen sampling request')
    p.add_argument('--out', type=Path, required=True, help='K1–K4: append-only generations')
    a = p.parse_args()
    cfg = json.loads(a.request.read_text())
    from vllm import LLM, SamplingParams
    adapter = cfg.get('adapter')
    llm = LLM(model=str(ROOT/'models'/cfg['model']), dtype='bfloat16',
              max_model_len=cfg['max_model_len'], seed=cfg['seed'],
              gpu_memory_utilization=cfg['eval_gpu_memory_utilization'],
              max_num_seqs=cfg['max_num_seqs'], trust_remote_code=False,
              enable_lora=bool(adapter), max_lora_rank=32,
              limit_mm_per_prompt={'image':0,'audio':0,'video':0})
    lora = None
    if adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest('checkpoint', 1, str(Path(adapter).resolve()))
    tasks = {t.task_id:t for t in verified_catalog()}
    if 'sample_seeds' in cfg:
        from tools.replication import stable_seed
        seeds=cfg['sample_seeds']
        if len(seeds)!=cfg['samples_per_task'] or len(set(seeds))!=len(seeds):
            raise ValueError('Independent samples require distinct declared seeds')
        # One request per task/sample: reproducible regardless of batch/order.
        entries=[(task,i,stable_seed(seed,task)) for task in cfg['tasks'] for i,seed in enumerate(seeds)]
        with a.out.open('x') as f:
            for start in range(0,len(entries),cfg['max_num_seqs']):
                batch_entries=entries[start:start+cfg['max_num_seqs']]
                params=[SamplingParams(n=1,seed=seed,**cfg['sampling']) for _,_,seed in batch_entries]
                outputs=llm.chat([messages(tasks[t]) for t,_,_ in batch_entries],params,
                    lora_request=lora,chat_template_kwargs={'enable_thinking':False},use_tqdm=False)
                for (task,index,seed),output in zip(batch_entries,outputs,strict=True):
                    item,=output.outputs
                    row={'task':task,'sample':index,'text':item.text,'tokens':len(item.token_ids),
                        'finish_reason':item.finish_reason,'model':cfg['model'],'sampling_seed':seed,
                        'source_sha256':tasks[task].sha256}
                    if cfg.get('record_token_ids'):row['token_ids']=list(item.token_ids)
                    f.write(json.dumps(row)+'\n');f.flush()
                print(f'generated {start+len(batch_entries)}/{len(entries)}',flush=True)
        return
    batch=max(1,cfg['max_num_seqs']//cfg['samples_per_task'])
    with a.out.open('x') as f:
        for start in range(0,len(cfg['tasks']),batch):
            task_ids=cfg['tasks'][start:start+batch]
            params=[SamplingParams(n=cfg['samples_per_task'],seed=cfg['seed']+start+j,
                                   **cfg['sampling']) for j in range(len(task_ids))]
            outputs=llm.chat([messages(tasks[t]) for t in task_ids],params,lora_request=lora,
                            chat_template_kwargs={'enable_thinking':False},use_tqdm=False)
            for j,(task_id,output) in enumerate(zip(task_ids,outputs,strict=True)):
                for i,item in enumerate(output.outputs):
                    row={'task':task_id,'sample':i,'text':item.text,'tokens':len(item.token_ids),
                         'finish_reason':item.finish_reason,'model':cfg['model'],
                         'sampling_seed':cfg['seed']+start+j,'source_sha256':tasks[task_id].sha256}
                    f.write(json.dumps(row)+'\n');f.flush()
                print(json.dumps({'task':task_id,'samples':len(output.outputs)}),flush=True)

if __name__ == '__main__':
    main()
