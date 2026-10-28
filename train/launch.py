"""K1–K4: LoRA GRPO, colocated sleeping vLLM, gated CUDA rewards and kill criteria.

Adapter structure reused from glomergence/train/launch.py at
e602f63207fe4e67eeda4dff7b53d49e92cc318b. CUDA evaluation is serialized and is
allowed only after vLLM has slept. Checkpoints include optimizer/RNG state.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import yaml
from env.reward import score_completion, score_result
from env.tasks import verified_catalog, validate_split
from tasks.prompts import messages
from tools.gatelib import ROOT, init_run, require_gate, write_json
from train.kill import KillController

def lora_hash(model):
    digest=hashlib.sha256()
    for name,param in sorted(model.named_parameters()):
        if 'lora_' in name:
            digest.update(name.encode())
            digest.update(param.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()

def is_text_parameter(name):
    """K1–K4: frozen unused multimodal aliases must not enter text-only vLLM."""
    if name.startswith(('model.language_model.','lm_head.')):
        return True
    if name.startswith(('model.embed_vision.','model.embed_audio.','model.vision_embedder.',
                        'model.vision_tower.','model.audio_tower.')):
        return False
    raise ValueError(f'Unexpected parameter outside the declared Gemma text/MM namespaces: {name}')

def install_text_weight_sync(trainer,directory):
    """K1/K2: keep TRL's PEFT merge/unmerge, only stream the trained text path."""
    if any(p.requires_grad and '.language_model.' not in name for name,p in trainer.model.named_parameters()):
        raise RuntimeError('Text-only synchronization cannot omit trainable nontext parameters')
    backend=trainer.vllm_generation
    original=backend._iter_named_params
    sync_count=0
    def text_parameters():
        nonlocal sync_count
        sync_count+=1;loaded=[];skipped=[]
        for name,param in original():
            if is_text_parameter(name):
                loaded.append(name)
                yield name,param
            else:
                skipped.append(name)
        if not loaded:raise RuntimeError('No language-model weights synchronized')
        # Resume uses a fresh process, so retain each session's sync sequence.
        path=directory/'weight-sync'/f'{time.time_ns()}-{sync_count}.json'
        write_json(path,{'loaded':loaded,'skipped_frozen_multimodal':skipped})
    backend._iter_named_params=text_parameters

class KernelReward:
    """K1–K3: explicit inference sleep + memory reclamation before each reward."""
    __name__='kernel_reward'
    def __init__(self, cfg, directory):
        self.cfg,self.directory=cfg,directory
        self.tasks={t.task_id:t for t in verified_catalog()}
        self.calls=0
        self.trainer=None
        self.latest={}

    def __call__(self, completions, task_id, completion_ids=None, trainer_state=None, log_metric=None, **kwargs):
        import torch
        import pynvml
        backend=self.trainer.vllm_generation
        if not backend.enable_sleep_mode or not backend._llm_weights_sleeping:
            raise RuntimeError('vLLM must be asleep before CUDA rewards')
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        pynvml.nvmlInit(); gpu=pynvml.nvmlDeviceGetHandleByIndex(0)
        start_mem=pynvml.nvmlDeviceGetMemoryInfo(gpu).used
        step=int(trainer_state.global_step)
        self.calls+=1
        texts=[x if isinstance(x,str) else ''.join(m.get('content','') for m in x) for x in completions]
        rows=[]; t0=time.monotonic()
        for i,(text,task) in enumerate(zip(texts,task_id)):
            seed=self.cfg['seed']+100000*step+100*self.calls+i
            path=self.directory/'rollouts'/f'{step:05d}-{self.calls:04d}-{i:02d}'
            result=score_completion(text,self.tasks[task],path,seed=seed)
            value=score_result(result,self.cfg['reward_arm'])
            row={'step':step,'call':self.calls,'index':i,'task':task,'seed':seed,
                 'text':text,'reward':value,'result':asdict(result)}
            with (self.directory/'rollouts.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            rows.append(row)
            print(f'[reward] step={step} sample={i} result={result.component} reward={value}',flush=True)
            if value is None:
                raise RuntimeError(f'Reward infrastructure failed: {result.details}')
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        end_mem=pynvml.nvmlDeviceGetMemoryInfo(gpu).used
        lengths=[len(x) for x in completion_ids] if completion_ids is not None else [len(t) for t in texts]
        self.latest={'env/valid':statistics.fmean(r['result']['correct'] for r in rows),
            'env/harness_errors':sum(r['result']['component']=='harness_error' for r in rows),
            'env/sandbox_violations':sum(r['result']['component']=='sandbox_blocked' for r in rows),
            'env/detector_fraction':statistics.fmean(bool(r['result']['hacks']) for r in rows),
            'env/median_length':statistics.median(lengths),
            'env/truncation_fraction':statistics.fmean(n>=self.cfg['grpo']['max_completion_length'] for n in lengths),
            'env/eval_seconds':time.monotonic()-t0,
            'env/gpu_before_bytes':start_mem,'env/gpu_after_bytes':end_mem,
            'env/gpu_delta_bytes':end_mem-start_mem}
        # A leaked worker context is infrastructure failure, not a model reward.
        if end_mem-start_mem > 64 << 20:
            self.latest['env/harness_errors']+=1
        if log_metric:
            for k,v in self.latest.items():
                log_metric(k,v)
        return [r['reward'] for r in rows]

def callbacks(cfg,directory,reward):
    from transformers import TrainerCallback
    control=KillController(cfg['kill'])
    previous=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    for row in previous:
        control.observe(row,0)
    started=time.time()
    prior_seconds=max((r.get('elapsed_seconds',0) for r in previous),default=0)
    class Guard(TrainerCallback):
        def on_log(self,args,state,control_state,logs=None,**kwargs):
            record={**(logs or {}),'step':state.global_step,
                    'elapsed_seconds':prior_seconds+time.time()-started}
            if 'loss' in record or 'reward' in record:
                record.update(reward.latest)
                reason=control.observe(record,record['elapsed_seconds'])
                if reason:
                    control_state.should_training_stop=True
                    control_state.should_save=True
                    if not (directory/'kill.json').exists():
                        write_json(directory/'kill.json',{'reason':reason,'step':state.global_step,
                                   'metrics':serializable_metrics(record)})
            with (directory/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(serializable_metrics(record),allow_nan=False)+'\n')
            return control_state
    return [Guard()]

def serializable_metrics(record):
    """K1–K3: retain a nonfinite failure instead of losing the stop record."""
    return {k:str(v) if isinstance(v,float) and not math.isfinite(v) else v
            for k,v in record.items()}

def resume_receipt(directory, checkpoint, digest):
    """K1–K4: keep original reload evidence and allow a crashed resume to retry."""
    record={'checkpoint':str(checkpoint),'lora_hash':digest}
    path=directory/f'resume-{checkpoint.name}.json'
    if path.exists():
        if json.loads(path.read_text())!=record:raise ValueError('Resume adapter differs from prior reload')
        write_json(directory/'resume-attempts'/f'{time.time_ns()}.json',record)
    else:write_json(path,record)

def build_trainer(cfg,directory,max_steps):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    import trl.generation.vllm_generation as generation
    # Pinned TRL doesn't expose multimodal profiling limits. Constrain its LLM
    # constructor locally, as validated in C0/C4; never patch installed files.
    original=generation.LLM
    class TextOnlyLLM(original):
        def __init__(self,*args,**kwargs):
            kwargs['limit_mm_per_prompt']={'image':0,'audio':0,'video':0}
            super().__init__(*args,**kwargs)
    generation.LLM=TextOnlyLLM
    tasks={t.task_id:t for t in verified_catalog()}
    training,_=validate_split(list(tasks.values()))
    if not set(cfg['tasks'])<=training:
        raise ValueError('Held-out leakage in training config')
    ds=Dataset.from_list([{'prompt':messages(tasks[t]),'task_id':t}
                         for _ in range(cfg['grpo']['max_steps']+8) for t in cfg['tasks']])
    g=cfg['grpo']; v=cfg['vllm']; lo=cfg['lora']
    args=GRPOConfig(output_dir=str(directory/'checkpoints'),seed=cfg['seed'],max_steps=max_steps,
        learning_rate=g['learning_rate'],lr_scheduler_type='constant',
        per_device_train_batch_size=g['micro_batch'],
        gradient_accumulation_steps=g['group_size']//g['micro_batch'],
        num_generations=g['group_size'],max_completion_length=g['max_completion_length'],
        temperature=1.0,top_p=.95,top_k=64,chat_template_kwargs={'enable_thinking':False},
        beta=g['beta'],loss_type='dapo',max_grad_norm=1.0,
        use_vllm=True,vllm_mode='colocate',vllm_gpu_memory_utilization=v['gpu_memory_utilization'],
        vllm_max_model_length=v['max_model_len'],vllm_enable_sleep_mode=True,
        vllm_importance_sampling_mode=g['importance_sampling_mode'],
        vllm_importance_sampling_clip_max=g['importance_sampling_clip_max'],
        model_init_kwargs={'dtype':'bfloat16','attn_implementation':'sdpa'},
        gradient_checkpointing=True,bf16=True,logging_steps=1,save_steps=g['save_steps'],
        save_total_limit=cfg.get('checkpoint_keep'),report_to=[],log_completions=False)
    peft=LoraConfig(r=lo['rank'],lora_alpha=lo['alpha'],lora_dropout=0.0,
        target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)',
        task_type='CAUSAL_LM')
    model=str(ROOT/'models'/cfg['model'])
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,padding_side='left')
    turn_id=tokenizer.convert_tokens_to_ids('<turn|>')
    stop_ids=json.loads((Path(model)/'generation_config.json').read_text())['eos_token_id']
    if turn_id not in stop_ids:raise RuntimeError('Gemma turn end must be an approved model stop token')
    tokenizer.eos_token='<turn|>'
    reward=KernelReward(cfg,directory)
    trainer=GRPOTrainer(model=model,reward_funcs=[reward],args=args,train_dataset=ds,
        processing_class=tokenizer,peft_config=peft,callbacks=callbacks(cfg,directory,reward))
    reward.trainer=trainer
    install_text_weight_sync(trainer,directory)
    return trainer

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path,help='K1–K4: frozen GRPO run config')
    p.add_argument('--resume',action='store_true',help='K1–K4: restore adapter + optimizer + RNG')
    p.add_argument('--max-steps',type=int,help='K1: C5 checkpoint/resume boundary')
    a=p.parse_args(); cfg=yaml.safe_load(a.config.read_text())
    require_gate('C4')
    if cfg['stage']=='C6': require_gate('C5')
    if cfg['stage'] in ('R1','R3'):
        from tools.replication import require_replication_gate,local_path
        require_replication_gate('R0' if cfg['stage']=='R1' else 'R2',local_path(cfg['protocol']))
    if not cfg.get('enabled'): raise RuntimeError('Training config is disabled')
    KillController(cfg['kill'])
    if cfg.get('method')=='sft':
        from train.sft import run
        run(cfg,a.resume,a.max_steps)
        return
    directory=ROOT/'runs'/cfg['run_id']
    if a.resume:
        if yaml.safe_load((directory/'config.yaml').read_text()) != cfg:
            raise RuntimeError('Resume config differs from original')
        if (directory/'kill.json').exists(): raise RuntimeError('Run was stopped by a declared criterion')
    else:
        if directory.exists(): raise RuntimeError('Run exists; use explicit resume')
        init_run(directory,cfg,cfg['seed'])
    checkpoints=sorted((directory/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    trainer=build_trainer(cfg,directory,a.max_steps or cfg['grpo']['max_steps'])
    if a.resume:
        if not checkpoints: raise RuntimeError('No resumable checkpoint')
        # Load the adapter explicitly into the fresh PEFT trainer before vLLM
        # first sync. Keep its KL reference as the disabled-adapter base model.
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        set_peft_model_state_dict(trainer.model,load_peft_weights(str(checkpoints[-1])),adapter_name='default')
        resume_receipt(directory,checkpoints[-1],lora_hash(trainer.model))
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if a.resume else None)
    final=directory/'checkpoints'/f'policy-step-{trainer.state.global_step}'
    trainer.save_model(str(final))
    write_json(directory/f'completion-step-{trainer.state.global_step}.json',
               {'step':trainer.state.global_step,'adapter':str(final),'lora_hash':lora_hash(trainer.model)})
    print(f'[launch] saved step {trainer.state.global_step} to {final}',flush=True)

if __name__ == '__main__':
    main()
