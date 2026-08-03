#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/ubuntu/research/deltasub"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES=0

BASE_RESULT="artifacts/deltasub_v2/cub/seed_0/selex/result.json"
OURS_CONFIG="configs/deltasub_v2/cub_seed0_decoupled.yaml"
OURS_RESULT="artifacts/deltasub_v2/cub/seed_0/deltasub_v2_decoupled/result.json"
COMPARISON="artifacts/deltasub_v2/cub/seed_0/comparison_decoupled.json"

test -f "$BASE_RESULT"

python -m deltasub.experiment.deltasub_v2_training \
  --config "$OURS_CONFIG" \
  --resume

python - <<'PY'
import json
from pathlib import Path

baseline = json.loads(
    Path(
        "artifacts/deltasub_v2/cub/seed_0/"
        "selex/result.json"
    ).read_text(encoding="utf-8")
)

ours = json.loads(
    Path(
        "artifacts/deltasub_v2/cub/seed_0/"
        "deltasub_v2_decoupled/result.json"
    ).read_text(encoding="utf-8")
)

metrics = ("all", "old", "new", "hmean")

comparison = {
    "schema_version": "deltasub-v2.comparison.v1",
    "dataset": "cub",
    "seed": 0,
    "baseline": baseline,
    "deltasub_v2_decoupled": ours,
    "delta": {
        metric: (
            float(ours["metrics"][metric])
            - float(baseline["metrics"][metric])
        )
        for metric in metrics
    },
}

path = Path(
    "artifacts/deltasub_v2/cub/seed_0/"
    "comparison_decoupled.json"
)
path.write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("metric          selex      revised-deltasub      delta")
for metric in metrics:
    left = float(baseline["metrics"][metric])
    right = float(ours["metrics"][metric])

    print(
        f"{metric:<8}     {left:.6f}     "
        f"{right:.6f}     {right-left:+.6f}"
    )

print()
print("branches:")
print(
    json.dumps(
        ours["branch_metrics"],
        indent=2,
        sort_keys=True,
    )
)
PY
