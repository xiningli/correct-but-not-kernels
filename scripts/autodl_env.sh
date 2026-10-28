#!/usr/bin/env bash
# K1–K4/C0: source from setup/runner; all large caches and managed Python live on the data disk.
# Every exported option below serves reproducible C0 stack setup; caller overrides are preserved.
KERNEL_PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export UV_CACHE_DIR=${UV_CACHE_DIR:-$KERNEL_PROJECT_ROOT/.cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$KERNEL_PROJECT_ROOT/.cache/python}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$KERNEL_PROJECT_ROOT/.cache/pip}
export HF_HOME=${HF_HOME:-$KERNEL_PROJECT_ROOT/.cache/huggingface}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-$KERNEL_PROJECT_ROOT/.cache/modelscope}
# K1–K4/C0: split the large single-file snapshots into bounded streaming ranges.
export MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=${MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS:-16}
export TORCH_HOME=${TORCH_HOME:-$KERNEL_PROJECT_ROOT/.cache/torch}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$KERNEL_PROJECT_ROOT/.cache/torch_extensions}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$KERNEL_PROJECT_ROOT/.cache/vllm}
# FlashInfer appends .cache/flashinfer to this base directory.
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$KERNEL_PROJECT_ROOT}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$KERNEL_PROJECT_ROOT/.cache/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$KERNEL_PROJECT_ROOT/.cache/cuda}
export PATH="$KERNEL_PROJECT_ROOT/.cache/bin:$KERNEL_PROJECT_ROOT/.venv/bin:$HOME/.local/bin:$PATH"
export UV_CONCURRENT_BUILDS=${UV_CONCURRENT_BUILDS:-1}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
export UV_CONCURRENT_INSTALLS=${UV_CONCURRENT_INSTALLS:-1}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-3}
export PYTHONUNBUFFERED=1
if [ -f /etc/network_turbo ] || [ -d /root/autodl-tmp ]; then
  export UV_PYTHON_INSTALL_MIRROR=${UV_PYTHON_INSTALL_MIRROR:-https://registry.npmmirror.com/-/binary/python-build-standalone}
  export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
fi
