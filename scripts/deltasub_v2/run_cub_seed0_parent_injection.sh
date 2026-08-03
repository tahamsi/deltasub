#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/ubuntu/research/deltasub
source .venv/bin/activate

export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0

ROOT="artifacts/deltasub_v2/cub/seed_0_parent_injection"

python -m deltasub.experiment.deltasub_v2_training \
  --config configs/deltasub_v2/cub_seed0_selex_fair.yaml \
  --resume

python -m deltasub.experiment.deltasub_v2_training \
  --config configs/deltasub_v2/cub_seed0_parent_injection.yaml \
  --resume

python - <<'PY'
import json
from pathlib import Path

root = Path(
    "artifacts/deltasub_v2/cub/"
    "seed_0_parent_injection"
)

baseline = json.loads(
    (root / "selex/result.json").read_text(
        encoding="utf-8"
    )
)
ours = json.loads(
    (root / "deltasub/result.json").read_text(
        encoding="utf-8"
    )
)

metrics = ("all", "old", "new", "hmean")

comparison = {
    "schema_version": "deltasub-v2.comparison.v1",
    "dataset": "cub",
    "seed": 0,
    "baseline": baseline["metrics"],
    "deltasub": ours["metrics"],
    "branches": ours["branch_metrics"],
    "diagnostics": ours["diagnostics"],
    "delta": {
        key: (
            float(ours["metrics"][key])
            - float(baseline["metrics"][key])
        )
        for key in metrics
    },
}

path = root / "comparison.json"
path.write_text(
    json.dumps(comparison, indent=2, sort_keys=True)
    + "\n",
    encoding="utf-8",
)

print("metric          selex      deltasub      delta")
for key in metrics:
    left = float(baseline["metrics"][key])
    right = float(ours["metrics"][key])

    print(
        f"{key:<8}     {left:.6f}     "
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
