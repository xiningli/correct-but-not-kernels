"""K1–K4/C0: reject empty, truncated or incomplete safetensors snapshots before loading."""
from __future__ import annotations

import json
import struct
from pathlib import Path

MODEL_IDS = ("google/gemma-4-12B-it", "google/gemma-4-E4B-it")


def inspect_weights(directory: Path):
    """Structural validation only; C0 additionally loads tensors, config and tokenizer."""
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"Weight directory does not exist: {directory}")
    if not (directory / "config.json").is_file():
        raise ValueError(f"Missing config.json in {directory}; configuration.json is not a substitute")
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") not in {"gemma4", "gemma4_unified"}:
        raise ValueError(f"Expected Gemma 4 config, got {config.get('model_type')!r}")
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"Cannot find any model weights: {directory}/*.safetensors")
    tensors = {}
    sizes = {}
    for path in files:
        size = path.stat().st_size
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"Truncated safetensors header: {path.name}")
            header_bytes = struct.unpack("<Q", raw)[0]
            if not 2 <= header_bytes <= min(size - 8, 100_000_000):
                raise ValueError(f"Invalid safetensors header size: {path.name}")
            header = json.loads(f.read(header_bytes))
        ranges = []
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            begin, end = spec["data_offsets"]
            if not (0 <= begin <= end <= size - 8 - header_bytes):
                raise ValueError(f"Truncated tensor payload: {path.name}:{name}")
            if name in tensors:
                raise ValueError(f"Duplicate tensor: {name}")
            tensors[name] = path.name
            ranges.append((begin, end))
        if not ranges:
            raise ValueError(f"No tensors in {path.name}")
        offset = 0
        for begin, end in sorted(ranges):
            if begin != offset:
                raise ValueError(f"Tensor payload gaps or overlaps: {path.name}")
            offset = end
        if offset != size - 8 - header_bytes:
            raise ValueError(f"Unexpected trailing bytes: {path.name}")
        sizes[path.name] = size
    index_path = directory / "model.safetensors.index.json"
    if len(files) > 1 and not index_path.is_file():
        raise ValueError("Sharded weights require model.safetensors.index.json")
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
        if mapping != tensors:
            missing = sorted(set(mapping.values()) - set(sizes))
            raise ValueError(f"Index does not match tensors on disk; missing shards: {missing}")
    return {"directory": str(directory), "model_type": config["model_type"],
            "architectures": config.get("architectures"), "files": sizes,
            "tensor_count": len(tensors), "total_bytes": sum(sizes.values()),
            "validation": "structure_only_not_a_model_load"}


def load_weights(directory: Path):
    """Load every tensor via safetensors and offline model metadata; no GPU allocation."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    record = inspect_weights(directory)
    loaded = 0
    for name in record["files"]:
        with safe_open(directory / name, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Touch mapped data, not just the header. Full primary-model load is vLLM's check.
                if tensor.numel():
                    flat = tensor.reshape(-1)
                    flat[0].item()
                    flat[-1].item()
                loaded += 1
                del tensor
    cfg = AutoConfig.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    if not tok.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True, enable_thinking=False):
        raise ValueError("Empty chat template output")
    return {**record, "validation": "all_tensors_loaded_cpu_and_offline_metadata",
            "loaded_tensors": loaded, "config_class": type(cfg).__name__, "tokenizer_class": type(tok).__name__}
