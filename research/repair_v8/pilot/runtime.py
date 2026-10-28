"""CPU startup checks for the two existing environments and pinned model."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from .artifacts import write_json
from .prepare import load, PROJECT


def check(prepared, inference_python, model, out):
    _, config = load(prepared)
    actual = {p: importlib.metadata.version(p) for p in ['torch', 'triton', 'transformers', 'safetensors']}
    expected = {'torch': '2.8.0', 'triton': '3.4.0', 'transformers': '4.57.1', 'safetensors': '0.8.0'}
    if sys.version_info[:2] != (3, 12) or any(actual[p].split('+')[0] != v for p, v in expected.items()):
        raise RuntimeError('Wrong evaluation venv: ' + str(actual))
    import torch
    if torch.version.cuda != '12.8':
        raise RuntimeError('Wrong evaluation CUDA runtime')
    code = '''import importlib.metadata,json,sys
from pathlib import Path
from research.repair_v8.pilot.sample import verify_model
cfg=json.loads(Path(sys.argv[1]).read_text())
v={p:importlib.metadata.version(p) for p in cfg['inference_packages']}
assert all(v[p].split('+')[0]==version for p,version in cfg['inference_packages'].items()), v
revision=verify_model(Path(sys.argv[2]))
from transformers import AutoTokenizer
from research.repair_v8.pilot.prepare import tokenizer
from research.repair_v8.pilot.protocol import messages,object_hash
prepared=Path(sys.argv[3]); a=tokenizer(prepared/'tokenizer'); b=AutoTokenizer.from_pretrained(sys.argv[2],local_files_only=True,trust_remote_code=False)
for task in cfg['tasks']:
 public=json.loads((prepared/'public'/(task+'.json')).read_text())
 source={public['path']:(prepared/'modules'/task/'before.py').read_text()}
 for interface in ['full_module','atomic_edits']:
  m=messages(public,source,interface,{'component':'incorrect','tests_observed':2,'tests_passed':1})
  x=a.apply_chat_template(m,add_generation_prompt=True,enable_thinking=False)
  y=b.apply_chat_template(m,add_generation_prompt=True,enable_thinking=False)
  assert x==y, 'Tokenizer mismatch'
print(json.dumps({'versions':v,'model_revision':revision,'tokenizer_parity':True}))
'''
    env = os.environ.copy(); env.update(PYTHONPATH=str(PROJECT), CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    completed = subprocess.run([str(inference_python), '-c', code, str(prepared / 'config.json'), str(model), str(prepared)],
                               env=env, capture_output=True, text=True, timeout=1200)
    result = {'evaluation_packages': actual, 'inference_check_returncode': completed.returncode,
              'inference_stdout': completed.stdout[-12000:], 'inference_stderr': completed.stderr[-12000:],
              'GPU_runs': 0, 'status': 'pass' if completed.returncode == 0 else 'blocked'}
    write_json(out, result)
    if completed.returncode:
        raise RuntimeError('Inference preflight failed: ' + str(out))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'inference-python', 'model', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(check(a.prepared.resolve(), a.inference_python.absolute(), a.model.resolve(), a.out)))


if __name__ == '__main__':
    main()
