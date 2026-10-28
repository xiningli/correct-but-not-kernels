"""K1–K4/C0: ModelScope downloads must pass tensor/metadata verification.
No hf-mirror fallback. Explicit node action; never called during local preparation.
"""
import argparse
import os
from pathlib import Path
from tools.gatelib import ROOT, timestamp, write_json
from tools.weights import MODEL_IDS, inspect_weights, load_weights

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", type=Path, default=ROOT / "models", help="K1–K4: explicit weight destination")
    p.add_argument("--structure-only", action="store_true",
                   help="K1–K4: no-GPU prefetch; validate files/index without importing torch or loading models")
    a = p.parse_args()
    # The legacy modelscope entry point imports torch even for file downloads.
    # K1–K4/C0: use its hub-only API to fit the no-GPU container's memory quota.
    from modelscope_hub import HubApi
    api = HubApi()
    for model_id in MODEL_IDS:
        dest = a.models_dir.resolve() / model_id.rsplit("/", 1)[1]
        api.download_repo(model_id, "model", local_dir=str(dest), max_workers=2)
        verification = inspect_weights(dest) if a.structure_only else load_weights(dest)
        write_json(ROOT / "runs/downloads" / f"{dest.name}-{timestamp()}.json",
                   {"model_id": model_id, "source": "modelscope.cn", "verification": verification,
                    "range_workers": os.environ.get("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "SDK default")})
        print(f"Verified {model_id}: {dest}", flush=True)

if __name__ == "__main__":
    main()
