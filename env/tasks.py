"""K1–K4: content-addressed KernelBench task catalog; imports execute no task code."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "third_party/KernelBench"
LOCK = ROOT / "env/kernelbench.lock.json"


@dataclass(frozen=True)
class Task:
    task_id: str
    level: int
    path: Path
    sha256: str

    def source(self):
        src = self.path.read_bytes()
        if hashlib.sha256(src).hexdigest() != self.sha256:
            raise ValueError(f"Task changed after catalog: {self.task_id}")
        return src.decode("utf-8")


def catalog(root: Path = DEFAULT_BENCH):
    tasks = []
    for level in (1, 2):
        paths = sorted((root / "KernelBench" / f"level{level}").glob("[0-9]*.py"))
        if not paths:
            raise ValueError(f"No Level {level} tasks under {root}")
        for path in paths:
            tasks.append(Task(f"level{level}/{path.stem}", level, path,
                              hashlib.sha256(path.read_bytes()).hexdigest()))
    return tasks


def verified_catalog(root: Path = DEFAULT_BENCH):
    lock = json.loads(LOCK.read_text())
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if sha != lock["revision"]:
        raise ValueError(f"KernelBench revision drift: {sha} != {lock['revision']}")
    tasks = catalog(root)
    if {t.task_id: t.sha256 for t in tasks} != lock["task_sha256"]:
        raise ValueError("KernelBench task content differs from committed lock")
    return tasks


def split_ids(path: Path = ROOT / "env/heldout.json"):
    split = json.loads(path.read_text())
    train, heldout = set(split["train_level1"]), set(split["heldout_level1"] + split["heldout_level2"])
    if not train or not heldout or train & heldout:
        raise ValueError("Empty or overlapping train/held-out split")
    return train, heldout


def validate_split(tasks, path: Path = ROOT / "env/heldout.json"):
    split = json.loads(path.read_text())
    train, heldout = split_ids(path)
    all_ids = [*split["train_level1"], *split["heldout_level1"], *split["heldout_level2"]]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != {t.task_id for t in tasks}:
        raise ValueError("Split must cover each locked task exactly once")
    if any(not x.startswith("level1/") for x in train | set(split["heldout_level1"])):
        raise ValueError("Training and Level-1 holdout must contain only Level 1")
    if any(not x.startswith("level2/") for x in split["heldout_level2"]):
        raise ValueError("Level-2 holdout contains a different level")
    if split["kernelbench_revision"] != json.loads(LOCK.read_text())["revision"]:
        raise ValueError("Split/KernelBench revision mismatch")
    return train, heldout


def load_reference(task: Task):
    """Only official, locked reference modules. NEVER use for policy/cheat programs."""
    allowed = {t.task_id: t for t in verified_catalog(task.path.parents[2])}
    if task != allowed.get(task.task_id):
        raise ValueError("Only pinned upstream references can be loaded")
    source = task.source()
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("kernelbench_reference", loader=None))
    exec(compile(source, str(task.path), "exec"), module.__dict__)
    for name in ("Model", "get_inputs", "get_init_inputs"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Reference missing {name}: {task.task_id}")
    return module
