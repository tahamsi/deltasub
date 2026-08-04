#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/ubuntu/research/deltasub
source .venv/bin/activate

export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0

ROOT="artifacts/deltasub_residual/cub/seed_0/anticollapse"
CONFIG="configs/deltasub_v2/cub_seed0_residual_anticollapse.yaml"

echo "===== PRACTICAL DELTASUB ANTI-COLLAPSE ====="

python -m \
  deltasub.experiment.deltasub_residual_training \
  --config "$CONFIG" \
  --resume

echo
echo "===== MATCHED SEED-0 GATE ====="

python -m \
  deltasub.experiment.deltasub_residual_compare \
  --candidate-result "$ROOT/result.json" \
  --output "$ROOT/comparison.json" \
  --bootstrap-draws 2000
