#!/usr/bin/env bash
# Development-task qualification only. Does not train or evaluate Gemma.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
phase="${1:-help}"
run_root="${2:-$project_dir/runs/repair-v8-qualification-20261018}"
export PYTHONPATH="$project_dir"
export PYTHONNOUSERSITE=1
qualification_python="$run_root/venv/bin/python"
case "$phase" in
  prepare)
    bootstrap_python="${QUALIFICATION_PYTHON:-python3.12}"
    "$bootstrap_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
    if [[ ! -x "$qualification_python" ]]; then
      if command -v uv >/dev/null 2>&1; then
        uv venv --python "$bootstrap_python" "$run_root/venv"
      else
        "$bootstrap_python" -m venv "$run_root/venv"
      fi
    fi
    if command -v uv >/dev/null 2>&1; then
      uv pip sync --python "$qualification_python" --require-hashes experiments/repair-v8-qualification.lock
    else
      "$qualification_python" -m pip install --require-hashes -r experiments/repair-v8-qualification.lock
    fi
    CUDA_VISIBLE_DEVICES='' "$qualification_python" -m research.repair_v8.qualify plan --root "$run_root"
    CUDA_VISIBLE_DEVICES='' "$qualification_python" -m research.repair_v8.qualify cpu-check --root "$run_root"
    CUDA_VISIBLE_DEVICES='' "$qualification_python" scripts/repair_v8_cpu_preflight.py --root "$run_root"
    ;;
  check)
    CUDA_VISIBLE_DEVICES='' "$qualification_python" -m research.repair_v8.qualify cpu-check --root "$run_root"
    CUDA_VISIBLE_DEVICES='' "$qualification_python" scripts/repair_v8_cpu_preflight.py --root "$run_root"
    ;;
  run)
    # Full qualification requires this tool. Fail before occupying the GPU if absent.
    command -v compute-sanitizer >/dev/null 2>&1 || {
      echo 'compute-sanitizer is required; install the CUDA development tools before running.' >&2
      exit 3
    }
    exec "$qualification_python" -u -m research.repair_v8.qualify gpu-run --root "$run_root"
    ;;
  *)
    echo 'Usage: bash scripts/repair_v8_autodl.sh prepare|check|run [absolute run directory]'
    echo 'prepare installs a separate locked Python 3.12 environment and runs CPU checks.'
    echo 'run executes trusted before/fixed GPU controls only; no Gemma or training.'
    ;;
esac
