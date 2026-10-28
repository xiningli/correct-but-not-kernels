#!/usr/bin/env bash
# Separate v8 pilot deployment. Never install into the old inference venv.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
phase="${1:-help}"
prepared="${PILOT_PREPARED:-$project_dir/prepared}"
pilot_run="${PILOT_RUN:-$project_dir/runs/pilot}"
old_project="/root/autodl-tmp/cuda-gemma4-llm"
inference_python="${PILOT_INFERENCE_PYTHON:-$old_project/.venv/bin/python}"
model="${PILOT_MODEL:-$old_project/models/gemma-4-31B-it}"
old_eval="/root/autodl-tmp/repair-v8-batch1-20261019T011025Z/runs/repair-v8-qualification-20261018/venv/bin/python"
evaluation_python="${PILOT_EVAL_PYTHON:-$old_eval}"
if [[ ! -x "$evaluation_python" && -x "$project_dir/.eval-venv/bin/python" ]]; then
  evaluation_python="$project_dir/.eval-venv/bin/python"
fi
export PYTHONPATH="$project_dir" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PATH="/usr/local/cuda/bin:/root/miniconda3/bin:$old_project/.cache/bin:$PATH"
export VLLM_CACHE_ROOT="$project_dir/.cache/vllm" TRITON_CACHE_DIR="$project_dir/.cache/triton"
export CUDA_CACHE_PATH="$project_dir/.cache/cuda" TMPDIR="$project_dir/.cache/tmp"
export TORCH_EXTENSIONS_DIR="$project_dir/.cache/torch_extensions" FLASHINFER_WORKSPACE_BASE="$old_project"
export HF_HOME="$old_project/.cache/huggingface" TORCH_HOME="$project_dir/.cache/torch"
mkdir -p "$TMPDIR" "$pilot_run"
case "$phase" in
  prepare)
    if [[ ! -x "$evaluation_python" ]]; then
      evaluation_python="$project_dir/.eval-venv/bin/python"
      bootstrap="${PILOT_BOOTSTRAP_PYTHON:-python3.12}"
      "$bootstrap" -m venv "$project_dir/.eval-venv"
      "$evaluation_python" -m pip install --require-hashes -r experiments/repair-v8-qualification.lock
      echo "Evaluation environment created at $evaluation_python; set PILOT_EVAL_PYTHON to this path for later invocations."
    fi
    CUDA_VISIBLE_DEVICES='' "$evaluation_python" -m research.repair_v8.pilot.runtime \
      --prepared "$prepared" --inference-python "$inference_python" --model "$model" --out "$pilot_run/runtime-preflight.json"
    ;;
  run)
    command -v compute-sanitizer >/dev/null
    export CUDA_VISIBLE_DEVICES=0
    exec "$evaluation_python" -u -m research.repair_v8.pilot.supervise \
      --prepared "$prepared" --out "$pilot_run" --inference-python "$inference_python" --model "$model" --power-off
    ;;
  *)
    echo 'Usage: bash scripts/repair_v8_pilot_autodl.sh prepare|run'
    echo 'prepare: CPU environment/model/tokenizer verification. run: guarded GPU controls, four-arm development pilot, report, backup acknowledgement and AutoDL shutdown.'
    ;;
esac
