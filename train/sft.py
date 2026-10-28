"""K1/K4: completion-only LoRA SFT on verified TRAIN programs, with resumable state."""
import json
from pathlib import Path
from types import SimpleNamespace
from env.tasks import verified_catalog
from tasks.prompts import messages, prompt_token_ids
from tools.gatelib import ROOT, init_run, write_json
from tools.replication import local_path, sha256
from train.sft_data import select_successes

def encode_example(tokenizer, prompt, completion_ids, max_length):
    prompt_ids=prompt_token_ids(tokenizer,prompt)
    completion=list(completion_ids)
    if not completion:raise ValueError('Empty SFT completion')
    if completion[-1]!=tokenizer.eos_token_id:completion.append(tokenizer.eos_token_id)
    ids=prompt_ids+completion
    if len(ids)>max_length:raise ValueError('SFT sequence exceeds declared context; no silent truncation')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),
            'labels':[-100]*len(prompt_ids)+completion}

def collate_examples(examples,pad_id):
    import torch
    length=max(len(x['input_ids']) for x in examples)
    return {key:torch.tensor([x[key]+[fill]*(length-len(x[key])) for x in examples],dtype=torch.long)
            for key,fill in [('input_ids',pad_id),('attention_mask',0),('labels',-100)]}

def load_text_model(path):
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    if cfg.model_type not in ('gemma4','gemma4_unified'):
        raise ValueError('Only the declared Gemma models are permitted')
    return AutoModelForImageTextToText.from_pretrained(path,dtype='bfloat16',
        attn_implementation='sdpa',local_files_only=True,trust_remote_code=False)

def run(cfg,resume=False,max_steps=None):
    from datasets import Dataset
    from peft import LoraConfig,get_peft_model
    from transformers import AutoTokenizer,Trainer,TrainingArguments,set_seed
    from train.launch import callbacks,lora_hash,resume_receipt
    import torch
    if not torch.cuda.is_available():raise RuntimeError('SFT requires the rented GPU; no CPU fallback')
    directory=ROOT/'runs'/cfg['run_id']
    data=local_path(cfg['sft']['data'])
    if sha256(data)!=cfg['sft']['data_sha256']:raise ValueError('SFT data drift')
    rows=[json.loads(x) for x in data.read_text().splitlines()]
    if len(select_successes(rows,cfg['tasks']))!=len(rows):raise ValueError('Unverified row in SFT dataset')
    if resume:
        import yaml
        if yaml.safe_load((directory/'config.yaml').read_text())!=cfg:raise ValueError('SFT resume config drift')
        if (directory/'kill.json').exists():raise RuntimeError('SFT stopped by declared criterion')
    else:
        if directory.exists():raise FileExistsError('Use explicit resume')
        init_run(directory,cfg,cfg['seed'])
    checkpoints=sorted((directory/'checkpoints').glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1]))
    if resume and not checkpoints:raise ValueError('SFT optimizer checkpoint missing')
    set_seed(cfg['seed'])
    model_path=str(ROOT/'models'/cfg['model'])
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,padding_side='right')
    tokenizer.eos_token='<turn|>'
    approved=json.loads((Path(model_path)/'generation_config.json').read_text())['eos_token_id']
    if tokenizer.eos_token_id not in approved:raise ValueError('Unexpected Gemma EOS')
    tasks={t.task_id:t for t in verified_catalog()}
    examples=[encode_example(tokenizer,messages(tasks[r['task']]),r['token_ids'],cfg['sft']['max_length']) for r in rows]
    model=load_text_model(model_path)
    model.config.use_cache=False
    model.config.get_text_config().use_cache=False
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    model=get_peft_model(model,LoraConfig(r=cfg['lora']['rank'],lora_alpha=cfg['lora']['alpha'],
        lora_dropout=0.,target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)',
        task_type='CAUSAL_LM'))
    model.enable_input_require_grads()
    monitor=SimpleNamespace(latest={})
    class CountingTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            if model.training:
                self.output_tokens=getattr(self,'output_tokens',0)+int((inputs['labels'][:,1:]!=-100).sum().item())
                monitor.latest={'sft/output_tokens_this_process':self.output_tokens}
            return super().compute_loss(model,inputs,return_outputs,num_items_in_batch)
    args=TrainingArguments(output_dir=str(directory/'checkpoints'),seed=cfg['seed'],
        max_steps=max_steps or cfg['grpo']['max_steps'],learning_rate=cfg['sft']['learning_rate'],
        lr_scheduler_type='constant',per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg['sft']['batch_size'],max_grad_norm=cfg['sft']['max_grad_norm'],
        bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        logging_steps=1,save_steps=1,save_total_limit=2,report_to=[],remove_unused_columns=False)
    trainer=CountingTrainer(model=model,args=args,train_dataset=Dataset.from_list(examples),
        data_collator=lambda batch:collate_examples(batch,tokenizer.pad_token_id),processing_class=tokenizer,
        callbacks=callbacks(cfg,directory,monitor))
    if resume:
        from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
        set_peft_model_state_dict(trainer.model,load_peft_weights(str(checkpoints[-1])),adapter_name='default')
        resume_receipt(directory,checkpoints[-1],lora_hash(trainer.model))
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if resume else None)
    final=directory/'checkpoints'/f'policy-step-{trainer.state.global_step}'
    trainer.save_model(str(final))
    write_json(directory/f'completion-step-{trainer.state.global_step}.json',
        {'step':trainer.state.global_step,'adapter':str(final),'lora_hash':lora_hash(trainer.model)})
