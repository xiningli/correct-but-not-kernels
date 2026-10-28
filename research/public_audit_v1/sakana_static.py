"""Static + logged-profile audit of SakanaAI/AI-CUDA-Engineer-Archive (CUDA C++ kernels).

No dataset code is compiled or executed: kernels are classified from source text and
from the archive's own Torch_Profile (parsed with ast.literal_eval, top-10 events only).
  python -m research.public_audit_v1.sakana_static --data <dir with level_*.parquet> --out <json>
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

LB = r'(?:__launch_bounds__\s*\([^)]*\)\s*)?'
# `__launch_bounds__` may sit before or after `void`.
GLOBAL = re.compile(r'__global__\s+' + LB + r'void\s+' + LB + r'(\w+)\s*[(<]')
# Host-side library compute that does the task's heavy lifting (not allocation/views).
HOST_COMPUTE = re.compile(
    r'\b(?:at|torch)::(?:nn::functional::)?(conv\w*|matmul|mm|addmm|bmm|baddbmm|linear|einsum|'
    r'batch_norm|group_norm|layer_norm|instance_norm|max_pool\w*|avg_pool\w*|adaptive_\w+|'
    r'softmax|log_softmax|relu|gelu|sigmoid|tanh|lstm|gru|embedding|scaled_dot_product_attention|'
    r'sum|mean|max|min|cumsum|cumprod|argmax|argmin|norm|dropout|hardtanh|clamp|mul|add|sub|div)\s*\(')
HEAVY = re.compile(r'conv|matmul|mm$|addmm|bmm|baddbmm|linear|einsum|lstm|gru|scaled_dot')
# Profiler events showing library compute: ATen compute ops (CPU-side records) or vendor GPU kernels.
ATEN_OP = re.compile(r'^aten::(?:_?convolution|conv\w*|cudnn_\w+|mm|addmm|bmm|baddbmm|matmul|linear|'
                     r'native_batch_norm|_?batch_norm\w*|native_group_norm|group_norm|native_layer_norm|layer_norm|'
                     r'max_pool\w*|avg_pool\w*|_softmax|softmax|lstm|gru|_scaled_dot_product\w*)$')
VENDOR_KERNEL = re.compile(r'cudnn|cublas|cutlass|xmma|sgemm|splitKreduce')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launched_static(code, name):
    # `name<<<`, `name<T><<<`, or a launch through cudaLaunchKernel/cooperative APIs naming it.
    return bool(re.search(rf'\b{re.escape(name)}\s*(?:<[^<>;]*>)?\s*<<<', code)
                or re.search(rf'cudaLaunch\w*\s*\(\s*\(?\s*(?:void\s*\*\s*\)?\s*)?&?\s*{re.escape(name)}\b', code))


def classify(code):
    names = sorted(set(GLOBAL.findall(code)))
    if not names:
        return 'no_declared_kernel', names
    if any(launched_static(code, n) for n in names):
        return 'launch_site', names
    # Any other mention after the definition (function pointers, templates) is unresolved, not decorative.
    if all(len(re.findall(rf'\b{re.escape(n)}\b', code)) == 1 for n in names):
        return 'definition_only', names
    return 'unresolved', names


def profile(text, names):
    if text is None:
        return None
    events = ast.literal_eval(text)
    keys = list(events)
    return {'events': len(keys),
            'declared_seen': any(k.split('(')[0].split('<')[0].replace('void ', '').strip() in names for k in keys),
            'vendor_seen': any(ATEN_OP.match(k) or (VENDOR_KERNEL.search(k) and not any(n in k for n in names))
                               for k in keys)}


def host_gemm_or_conv(code):
    return any(HEAVY.match(op) for op in set(HOST_COMPUTE.findall(code)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = {'dataset': 'SakanaAI/AI-CUDA-Engineer-Archive', 'license': 'CC-BY-4.0',
           'source': 'https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive', 'levels': {}}
    for level in [1, 2, 3]:
        path = args.data / f'level_{level}.parquet'
        df = pd.read_parquet(path)
        c = Counter()
        for _, r in df[df['Correct'] == True].iterrows():  # noqa: E712 (numpy bool column)
            cls, names = classify(r['CUDA_Code'])
            heavy = host_gemm_or_conv(r['CUDA_Code'])
            c[f'class/{cls}'] += 1
            c[f'class/{cls}/host_gemm_or_conv={heavy}'] += 1
            fast = (r['CUDA_Speedup_Native'] or 0) > 1
            c[f'faster_than_native/{cls}/host_gemm_or_conv={heavy}'] += fast
            c['faster_than_native'] += fast
            p = profile(r['Torch_Profile'], names)
            if p is None:
                c['profile/missing'] += 1
                continue
            c[f'profile/{cls}/declared_seen={p["declared_seen"]}'] += 1
            c[f'profile/{cls}/vendor_seen={p["vendor_seen"]}'] += 1
            c[f'profile/{cls}/truncated={p["events"] >= 10}'] += 1
            # Validation: static host GEMM/conv call vs library compute in the archive's own profile.
            c[f'profile/agreement/static={heavy}/profile={p["vendor_seen"]}'] += 1
        out['levels'][level] = {'sha256': sha(path), 'rows': len(df),
                                'correct': int((df['Correct'] == True).sum()),  # noqa: E712
                                'tasks_with_correct': int(df[df['Correct'] == True]['Task_ID'].nunique()),  # noqa: E712
                                'counts': dict(sorted(c.items()))}
    out['scope'] = ('static source classes plus the archive\'s own top-10 Torch_Profile events; '
                    'absence from a truncated profile is not evidence of non-launch')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
