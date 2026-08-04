#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/ubuntu/research/deltasub
source .venv/bin/activate

export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0

python -m \
  deltasub.experiment.old_label_utility_router \
  --config \
  configs/deltasub_v2/cub_seed0_supervised_deltasub.yaml \
  --checkpoint \
  artifacts/deltasub_v2/supervised/cub/seed_0/deltasub/checkpoint_last.pt \
  --oracle-predictions \
  artifacts/deltasub_v2/supervised/cub/seed_0/oracle_routing/predictions.npz \
  --output \
  artifacts/deltasub_v2/supervised/cub/seed_0/old_label_router \
  --resume
