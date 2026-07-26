#!/usr/bin/env bash
set -euo pipefail

# Sequential single-A100 driver. It intentionally stops after the diagnostic unless
# DIAGNOSTIC_VERDICT.md is populated by completed real-data experiments.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p artifacts/logs artifacts/runs
exec > >(tee -a artifacts/logs/a100_diagnostic.stdout.log)
exec 2> >(tee -a artifacts/logs/a100_diagnostic.stderr.log >&2)

python -m unittest discover -s tests -p 'test_*.py' -v
python -m deltasub.cli doctor
python -m deltasub.cli doctor memory \
  --config configs/experiment/cub_deltasub.yaml \
  --hardware configs/hardware/a100_80gb.yaml

# Infrastructure smoke test on the server GPU. Synthetic metrics are tagged and excluded
# from paper aggregation.
python -m deltasub.cli smoke \
  --output artifacts/runs/a100_synthetic_smoke \
  --size 8 \
  --seed 0 \
  --device cuda:0 \
  --resume

cat <<'NOTICE'
Infrastructure validation completed.

Real CUB/SelEx training commands are deliberately not launched by this scaffold:
licensed datasets, exact upstream split files, DINOv2 checkpoints, and the verified
per-anchor SelEx integration must be supplied and pass their gates first. Do not treat
the synthetic run as a scientific result.
NOTICE
