#!/usr/bin/env bash
# K1–K4. Adapted from glomergence/scripts/gpu_env.sh (e602f63207fe4e67eeda4dff7b53d49e92cc318b).
# K1/K2: KERNEL_CUDA_HOME can name a matching CUDA 13 toolkit; exclude Conda compilers.
set -euo pipefail
source "$(dirname "$0")/autodl_env.sh"
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | awk 'tolower($0) !~ /conda/' | paste -sd:)
export PATH
unset CC CXX CUDA_HOME CUDA_PATH CONDA_PREFIX NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
if [ -n "${KERNEL_CUDA_HOME:-}" ]; then
  test -x "$KERNEL_CUDA_HOME/bin/nvcc" || { echo "Missing nvcc in KERNEL_CUDA_HOME" >&2; exit 1; }
  export CUDA_HOME="$KERNEL_CUDA_HOME" PATH="$KERNEL_CUDA_HOME/bin:$PATH"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec "$@"
