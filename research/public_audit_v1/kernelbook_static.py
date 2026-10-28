"""Static decorative-kernel audit of GPUMODE/KernelBook (TorchInductor-generated Triton).

Reads the parquet without executing any dataset code (ast parsing only).
  python -m research.public_audit_v1.kernelbook_static --data <dataset_permissive.parquet> --out <json>
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from research.paper_v1.training_shortcut_audit import static_class
from research.public_audit_v1.sakana_static import HEAVY

# Inductor routes GEMM/conv and other library calls through extern_kernels or aten ops.
EXTERN = re.compile(r'\bextern_kernels\.(\w+)|\btorch\.ops\.aten\.(\w+)')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(row):
    # The translated module is `<entry_point>New`; alias it so the reachability walk starts there.
    source = f"{row['triton_code']}\nModelNew = {row['entry_point']}New\n"
    cls = static_class(source)
    extern = sorted({a or b for a, b in EXTERN.findall(row['triton_code'])})
    return cls, extern


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.data)
    classes, extern_any, extern_ops, extern_heavy = Counter(), Counter(), Counter(), Counter()
    by_synthetic = {True: Counter(), False: Counter()}
    for _, row in df.iterrows():
        cls, extern = classify(row)
        classes[cls] += 1
        by_synthetic[bool(row['synthetic'])][cls] += 1
        if cls == 'reachable':
            extern_any[bool(extern)] += 1
            extern_ops.update(extern)
            extern_heavy[any(HEAVY.match(op) for op in extern)] += 1
    out = {'dataset': 'GPUMODE/KernelBook dataset_permissive.parquet', 'license': 'June 9 Researcher Reciprocity License',
           'source': 'https://huggingface.co/datasets/GPUMODE/KernelBook', 'sha256': sha(args.data), 'rows': len(df),
           'static_class': dict(classes),
           'static_class_by_synthetic': {str(k): dict(v) for k, v in by_synthetic.items()},
           'reachable_with_extern_or_aten_call': {str(k): v for k, v in extern_any.items()},
           'reachable_with_extern_gemm_or_conv': {str(k): v for k, v in extern_heavy.items()},
           'extern_or_aten_ops_in_reachable': dict(extern_ops.most_common(25)),
           'scope': 'static source classes only; no dataset code executed'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
