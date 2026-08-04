#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/ubuntu/research/deltasub
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0

python scripts/intervention_stability/cub_seed0_gate.py \
  --output artifacts/intervention_stability/cub/seed_0 \
  --batch-size 16 \
  --bootstrap-draws 2000 \
  --resume
