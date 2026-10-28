"""K1/K3: read-only teacher-forced KL replay; generated programs are text, never executed."""
import argparse
import json
from pathlib import Path
from env.tasks import verified_catalog
from tasks.prompts import messages, prompt_token_ids
from tools.gatelib import ROOT, write_json
from tools.replication import load_plan, local_path

def logps(model,ids,prompt_length):
    import torch
    n=ids.shape[1]-prompt_length
    with torch.inference_mode():
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=n+1)
        logits=output.logits[:,-n-1:-1,:].float()
        targets=ids[:,prompt_length:]
        return (logits.gather(-1,targets.unsqueeze(-1)).squeeze(-1)-logits.logsumexp(-1)).cpu()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan=load_plan(a.plan)
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from train.sft import load_text_model
    if not torch.cuda.is_available():raise RuntimeError('KL probe requires GPU')
    model_path=str(ROOT/'models'/plan['model'])
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=load_text_model(model_path).cuda().eval()
    tasks={t.task_id:t for t in verified_catalog()}
    rows=[json.loads(x) for x in (local_path(plan['pilot'])/'training-C/rollouts.jsonl').read_text().splitlines()]
    replay=[r for r in rows if r['step']==34]
    if len(replay)!=8 or any(r['task'] not in plan['train_tasks'] for r in replay):
        raise ValueError('Expected the eight TRAIN completions immediately preceding optimizer step 35')
    examples=[];base=[]
    for r in replay:
        prompt=prompt_token_ids(tokenizer,messages(tasks[r['task']]))
        completion=tokenizer.encode(r['text'],add_special_tokens=False)+[tokenizer.convert_tokens_to_ids('<turn|>')]
        ids=torch.tensor([prompt+completion],device='cuda')
        base.append(logps(model,ids,len(prompt)));examples.append((ids,len(prompt)))
    model=PeftModel.from_pretrained(model,str(local_path(plan['pilot_adapters']['C']['path'])),
        local_files_only=True,is_trainable=False).eval()
    results=[];drift=0.
    for i,((ids,length),reference) in enumerate(zip(examples,base,strict=True)):
        with model.disable_adapter():disabled=logps(model,ids,length)
        drift=max(drift,float((reference-disabled).abs().max()))
        policy=logps(model,ids,length)
        ratio=reference-policy
        kl=ratio.exp()-ratio-1
        results.append({'sample':i,'tokens':int(kl.numel()),'kl_mean':float(kl.mean()),
            'kl_max':float(kl.max()),'max_log_ratio':float(ratio.max()),
            'finite':bool(torch.isfinite(kl).all()),'max_kl_token_position':int(kl.argmax())})
    passed=drift<=1e-4 and all(r['finite'] for r in results)
    write_json(a.out,{'status':'PASS' if passed else 'FAIL','disabled_adapter_base_max_abs_logp_difference':drift,
        'samples':results,'scope':'Post-update step-35 adapter, retokenized TRAIN completions; diagnostic replay, '
        'not exact pre-update token replay and not a root-cause proof. Original token log probabilities were not saved.'})
    if not passed:raise RuntimeError('Reference invariance or finite-KL audit failed')

if __name__=='__main__':main()
