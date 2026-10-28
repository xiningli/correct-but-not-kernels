#!/usr/bin/env bash
# K1–K4/C0: explicit CPU preparation or GPU C0; never starts C1–C6 or training.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
stage=${1:-status}
case "$stage" in prepare|prefetch|status|c0) ;; *) echo 'Usage: bash scripts/run_autodl.sh prepare|prefetch|status|c0' >&2; exit 2 ;; esac
mkdir -p runs/setup
exec 9>runs/setup/autodl.lock
flock -n 9 || { echo 'Another AutoDL setup/C0 command is running.' >&2; exit 1; }
run_stamp=$(date -u +%Y%m%dT%H%M%S)-$$
log_path="runs/setup/$stage-$run_stamp.log"
# K1–K4/C0: keep the file log writable if an SSH viewer disconnects.
exec > >(tee -a --output-error=warn-nopipe "$log_path") 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "Stopped (exit $rc). Preserve log: $log_path"; fi' EXIT
echo "Stage $stage | UTC $run_stamp | log $log_path"
if [ "$stage" = prepare ]; then
  bash scripts/setup_node.sh
  .venv/bin/python -m tools.preflight
  echo 'CPU preparation complete. After GPU boot and disk expansion: bash scripts/run_autodl.sh c0'
  exit 0
fi
if [ -x .venv/bin/python ]; then
  node_python=$PWD/.venv/bin/python
elif [ -x /root/miniconda3/bin/python ]; then
  node_python=/root/miniconda3/bin/python
else
  node_python=$(command -v python3)
fi
if [ "$stage" = status ]; then
  scripts/gpu_env.sh "$node_python" -m tools.inventory
  exit 0
fi
if [ "$stage" = prefetch ]; then
  scripts/gpu_env.sh "$node_python" -m tools.check_node --download-only
  # K1–K4: keep simultaneous wheel extraction within the no-GPU memory quota.
  UV_CONCURRENT_DOWNLOADS=1 bash scripts/setup_node.sh --prefetch
  .venv/bin/python -m pytest -q
  scripts/gpu_env.sh .venv/bin/python -m tools.download_models --models-dir "$PWD/models" --structure-only
  echo 'Downloads and structural validation complete. No model was loaded; run c0 after GPU boot.'
  exit 0
fi
# Refuse before large installs/downloads; readiness records are not formal gate attempts.
scripts/gpu_env.sh "$node_python" -m tools.check_node
"$node_python" -c 'from tools.gatelib import enforce_failure_stop; enforce_failure_stop("C0")'
bash scripts/setup_node.sh --gpu
.venv/bin/python -m pytest -q
scripts/gpu_env.sh .venv/bin/python -m tools.download_models --models-dir "$PWD/models"
scripts/gpu_env.sh .venv/bin/python -m tools.gate_c0 --models-dir "$PWD/models"
echo 'C0 PASS. Commit the emitted evidence record and source before C1. Stop: no further gate or training was started.'
