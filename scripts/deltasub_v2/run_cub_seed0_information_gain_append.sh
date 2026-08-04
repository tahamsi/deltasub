#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/ubuntu/research/deltasub
source .venv/bin/activate

export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0

python \
  scripts/deltasub_v2/information_gain_append_gate.py \
  --output \
  artifacts/deltasub_information_gain/cub/seed_0 \
  --epochs 30 \
  --candidate-chunk-size 32 \
  --bootstrap-draws 2000 \
  --resume
