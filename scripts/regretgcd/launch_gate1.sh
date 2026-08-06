#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/artifacts/logs/regretgcd_gate1_${STAMP}.log"
EXIT_FILE="$ROOT/artifacts/logs/regretgcd_gate1_${STAMP}.exitcode"
mkdir -p "$ROOT/artifacts/logs"

nohup setsid bash -c '
  set +e
  cd "$1"
  export PYTHONPATH="$PWD/src"
  export CUDA_VISIBLE_DEVICES=0
  "$PWD/.venv/bin/python" scripts/regretgcd/gate1_learned_router.py \
    --config configs/regretgcd/cub_seed0.yaml --resume
  status=$?
  if [ "$status" -eq 0 ]; then
    "$PWD/.venv/bin/python" scripts/regretgcd/compare_sota.py \
      --result artifacts/regretgcd/cub/seed_0/result.json \
      --registry configs/regretgcd/sota_registry.yaml \
      --output artifacts/regretgcd/cub/seed_0/comparison
    compare_status=$?
    if [ "$compare_status" -ne 0 ]; then status=$compare_status; fi
  fi
  printf "%s\n" "$status" > "$2"
  exit "$status"
' _ "$ROOT" "$EXIT_FILE" </dev/null >"$LOG" 2>&1 &

PID=$!
printf 'PID: %s\nLOG: %s\nEXIT: %s\n' "$PID" "${LOG#$ROOT/}" "${EXIT_FILE#$ROOT/}"
