#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/ubuntu/research/deltasub
source .venv/bin/activate

export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0

ROOT="artifacts/deltasub_v2/supervised/cub/seed_0"

echo "===== FULLY SUPERVISED SELEX ====="
python -m deltasub.experiment.deltasub_v2_training \
  --config \
  configs/deltasub_v2/cub_seed0_supervised_selex.yaml \
  --resume

echo
echo "===== FULLY SUPERVISED DELTASUB ====="
python -m deltasub.experiment.deltasub_v2_training \
  --config \
  configs/deltasub_v2/cub_seed0_supervised_deltasub.yaml \
  --resume

echo
echo "===== REPRESENTATION GATE ====="
python - <<'PY'
import json
from pathlib import Path

root = Path(
    "artifacts/deltasub_v2/supervised/"
    "cub/seed_0"
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

delta = {
    metric: (
        float(ours["metrics"][metric])
        - float(baseline["metrics"][metric])
    )
    for metric in metrics
}

passed = (
    delta["all"] > 0.0
    and delta["new"] > 0.0
    and delta["hmean"] > 0.0
)

comparison = {
    "schema_version": (
        "deltasub-v2.supervised-gate.v1"
    ),
    "dataset": "cub",
    "seed": 0,
    "training_protocol": "fully_supervised",
    "baseline": baseline,
    "deltasub": ours,
    "delta": delta,
    "representation_gate": (
        "passed" if passed else "failed"
    ),
    "gate_rule": {
        "all_delta_positive": True,
        "new_delta_positive": True,
        "hmean_delta_positive": True,
    },
}

path = root / "comparison.json"
path.write_text(
    json.dumps(
        comparison,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print(
    "metric          supervised-selex  "
    "supervised-deltasub      delta"
)

for metric in metrics:
    left = float(baseline["metrics"][metric])
    right = float(ours["metrics"][metric])

    print(
        f"{metric:<8}     "
        f"{left:.6f}           "
        f"{right:.6f}       "
        f"{right-left:+.6f}"
    )

print()
print(
    "representation gate: "
    + ("PASSED" if passed else "FAILED")
)
print(f"comparison: {path}")
PY
