"""K1–K4: trusted C0 probes in disposable processes; never execute policy programs.
Sampling adapted from glomergence/tools/gate_g4.py at e602f63207fe4e67eeda4dff7b53d49e92cc318b.
C0 generates a fixed text response, not kernel code.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from tools.gatelib import write_json

SMOKE_TASKS = ("level1/1_Square_matrix_multiplication_", "level1/2_Standard_matrix_multiplication_")
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

def reference_smoke():
    import torch
    from env.tasks import load_reference, validate_split, verified_catalog
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    tasks = verified_catalog()
    train, _ = validate_split(tasks)
    by_id = {t.task_id: t for t in tasks}
    torch.manual_seed(0)
    results = []
    for task_id in SMOKE_TASKS:
        if task_id not in train:
            raise RuntimeError("Smoke task must belong to the committed training split")
        mod = load_reference(by_id[task_id])
        model = mod.Model(*mod.get_init_inputs()).eval().cuda()
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in mod.get_inputs()]
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output = model(*inputs)
        torch.cuda.synchronize()
        if not isinstance(output, torch.Tensor) or not bool(torch.isfinite(output).all()):
            raise RuntimeError(f"Non-tensor or non-finite reference output: {task_id}")
        results.append({"task_id": task_id, "source_sha256": by_id[task_id].sha256,
                        "output_shape": list(output.shape), "dtype": str(output.dtype),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "finite": True})
        del output, inputs, model, mod
        torch.cuda.empty_cache()
    return {"scope": "trusted_reference_smoke_not_performance_or_correctness_evaluation",
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "results": results}

def generation(model: Path):
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(model), dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.85, seed=0, trust_remote_code=False,
              limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})
    outputs = llm.chat([[{"role": "user", "content": "Reply with exactly the word READY."}]],
                       SamplingParams(max_tokens=64, seed=0, **SAMPLING), use_tqdm=False,
                       chat_template_kwargs={"enable_thinking": False})
    text = outputs[0].outputs[0].text
    final = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.S)
    if "<|channel>thought" in final or not re.search(r"\bREADY\b", final):
        raise RuntimeError(f"vLLM did not answer the smoke prompt: {text!r}")
    return {"model": str(model), "text": text, "sampling": SAMPLING,
            "finish_reason": outputs[0].outputs[0].finish_reason,
            "tokens": len(outputs[0].outputs[0].token_ids)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["reference", "weights", "generate"], help="K1–K4: C0 probe")
    p.add_argument("--model", type=Path, help="K1–K4: local snapshot, no implicit download")
    p.add_argument("--out", type=Path, required=True, help="K1–K4: immutable probe result")
    a = p.parse_args()
    try:
        if a.mode == "reference":
            result = reference_smoke()
        elif a.mode == "weights":
            from tools.weights import load_weights
            result = load_weights(a.model)
        else:
            result = generation(a.model)
    except Exception as e:
        write_json(a.out, {"status": "FAIL", "error": f"{type(e).__name__}: {e}"})
        raise
    write_json(a.out, {"status": "PASS", **result})
    print(json.dumps({"status": "PASS", "mode": a.mode, "result": str(a.out)}))

if __name__ == "__main__":
    main()
