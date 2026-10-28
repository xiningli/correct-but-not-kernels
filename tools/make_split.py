"""K1–K4: freeze task identities before results exist; never silently overwrite a split."""
import hashlib
import json
import subprocess

from env.tasks import DEFAULT_BENCH, ROOT, catalog


def main():
    paths = [ROOT / "env/kernelbench.lock.json", ROOT / "env/heldout.json"]
    if any(p.exists() for p in paths):
        raise SystemExit("Split/lock already exists. Preserve it; regeneration requires an explicit design decision.")
    tasks = catalog()
    revision = subprocess.check_output(["git", "-C", str(DEFAULT_BENCH), "rev-parse", "HEAD"], text=True).strip()
    # K1–K4: fixed seed, 20% L1 holdout; all L2 held out. No model outcomes are inspected.
    seed = "cuda-gemma4-k1-k4-v1"
    l1 = sorted((t.task_id for t in tasks if t.level == 1), key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    n = max(1, len(l1) // 5)
    lock = {"url": "https://github.com/ScalingIntelligence/KernelBench.git", "revision": revision,
            "task_sha256": {t.task_id: t.sha256 for t in tasks}}
    split = {"hypotheses": ["K1", "K2", "K3", "K4"], "kernelbench_revision": revision,
             "method": "sha256(seed:task_id), first floor(N/5) Level-1 tasks held out", "seed": seed,
             "train_level1": sorted(l1[n:]), "heldout_level1": sorted(l1[:n]),
             "heldout_level2": sorted(t.task_id for t in tasks if t.level == 2)}
    for path, data in zip(paths, (lock, split)):
        with path.open("x") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    print("Created lock and immutable split. Commit before any training.")


if __name__ == "__main__":
    main()
