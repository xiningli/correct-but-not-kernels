#!/usr/bin/env bash
# K1–K4. Adapted from glomergence/scripts/setup_node.sh (e602f63207fe4e67eeda4dff7b53d49e92cc318b).
# Default: dev tools. --prefetch: install pinned GPU wheels without loading CUDA.
# --gpu additionally requires a visible CUDA-13-capable driver.
# No provisioning, weight downloads, training or shell-profile edits.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/autodl_env.sh
extras=dev
check_driver=0
case "${1:-}" in
  "") ;;
  --gpu) extras=dev,gpu; check_driver=1 ;;
  --prefetch) extras=dev,gpu ;;
  *) echo 'Usage: bash scripts/setup_node.sh [--gpu|--prefetch]' >&2; exit 2 ;;
esac
if [ "$check_driver" = 1 ]; then
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\..*/\1/p' | head -1)
  if [ -z "$cuda" ] || [ "$cuda" -lt 13 ]; then
    echo 'Need a driver supporting CUDA >=13.0; choose an appropriate AutoDL host.' >&2; exit 1
  fi
fi
if ! command -v uv >/dev/null; then
  # Bootstrap with image Python even when SSH PATH omits Conda. Do not modify Conda.
  if [ -x /root/miniconda3/bin/python ]; then
    /root/miniconda3/bin/python -m pip install --no-deps --target .cache/uv-bootstrap \
      --index-url "${UV_DEFAULT_INDEX:-https://pypi.org/simple}" 'uv==0.11.6'
    mkdir -p .cache/bin
    uv_binary=$(find "$PWD/.cache/uv-bootstrap" -type f -name uv -perm /111 -print -quit)
    test -n "$uv_binary" || { echo 'uv wheel installed without an executable' >&2; exit 1; }
    ln -s "$uv_binary" .cache/bin/uv
  else
    uv_installer=$(mktemp)
    curl -fLsS --connect-timeout 15 --max-time 120 https://astral.sh/uv/install.sh -o "$uv_installer"
    UV_UNMANAGED_INSTALL="$KERNEL_PROJECT_ROOT/.cache/bin" sh "$uv_installer"
    rm -f "$uv_installer"
  fi
fi
if [ ! -x .venv/bin/python ]; then
  uv venv --managed-python -p 3.12 .venv
fi
.venv/bin/python -c 'import pathlib,sys,sysconfig; assert sys.version_info[:2] == (3, 12), "Use uv-managed Python 3.12"; assert (pathlib.Path(sysconfig.get_path("include"))/"Python.h").is_file(), "Missing Python.h needed for Triton JIT"'
asset_sources=()
if [ "$extras" = dev,gpu ]; then
  # K1–K4/C0: permit an integrity-checked offline wheel when GitHub is unreachable.
  cubin_wheel=.cache/wheels/flashinfer_cubin-0.6.18.post1-py3-none-any.whl
  if [ -f "$cubin_wheel" ]; then
    printf '%s  %s\n' bbacb5b8bbf429e43bf2740bb45551d981f41d0462842e36ee6fb3e3452763ab "$cubin_wheel" | sha256sum -c -
    asset_sources=(--find-links "$PWD/.cache/wheels")
  else
    asset_sources=(--find-links https://flashinfer.ai/whl/flashinfer-cubin/)
  fi
fi
uv pip install -p .venv/bin/python "${asset_sources[@]}" -e ".[${extras}]"
.venv/bin/python -m tools.fetch_kernelbench
scripts/gpu_env.sh .venv/bin/python -m tools.inventory
echo 'Setup complete. This is not C0 PASS. Follow README.md for weight verification and C0.'
