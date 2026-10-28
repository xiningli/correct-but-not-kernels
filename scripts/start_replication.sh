#!/usr/bin/env bash
# K1–K4: explicit GPU-boot entry point; no work is started by merely logging in.
set -euo pipefail
cd "$(dirname "$0")/.."
test "$(id -u)" = 0
test "$(dirname "$PWD")" = /root/autodl-tmp
exec scripts/gpu_env.sh .venv/bin/python -u -m tools.run_replication \
  --study "$PWD/runs/replication-v1.1" --stage "${1:-all}" --power-off
