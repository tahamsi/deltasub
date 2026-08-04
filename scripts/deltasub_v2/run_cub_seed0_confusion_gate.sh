#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/ubuntu/research/deltasub
source .venv/bin/activate

export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0

python \
  scripts/deltasub_v2/confusion_evidence_gate.py \
  --output \
  artifacts/deltasub_confusion/cub/seed_0 \
  --epochs 12 \
  --bootstrap-draws 2000 \
  --resume
