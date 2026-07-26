#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"

python -m unittest discover -s tests -p 'test_*.py' -v
python -m deltasub.cli doctor
python -m deltasub.cli smoke \
  --output artifacts/runs/synthetic_smoke \
  --size 8 \
  --seed 0 \
  --device cpu \
  --resume
