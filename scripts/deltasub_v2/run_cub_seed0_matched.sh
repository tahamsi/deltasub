#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/ubuntu/research/deltasub"
cd "$ROOT"

source .venv/bin/activate
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES=0

BASE_CONFIG="configs/deltasub_v2/cub_seed0_selex.yaml"
OURS_CONFIG="configs/deltasub_v2/cub_seed0.yaml"

echo "===== MATCHED SELEX ====="
python -m deltasub.experiment.deltasub_v2_training \
  --config "$BASE_CONFIG" \
  --resume

echo
echo "===== DELTASUB V2 ====="
python -m deltasub.experiment.deltasub_v2_training \
  --config "$OURS_CONFIG" \
  --resume

echo
echo "===== MATCHED COMPARISON ====="
python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/deltasub_v2/cub/seed_0")

paths = {
    "selex": root / "selex" / "result.json",
    "deltasub_v2": root / "deltasub_v2" / "result.json",
}

results = {
    name: json.loads(path.read_text(encoding="utf-8"))
    for name, path in paths.items()
}

baseline = results["selex"]
ours = results["deltasub_v2"]

metrics = ("all", "old", "new", "hmean")

comparison = {
    "schema_version": "deltasub-v2.comparison.v1",
    "dataset": "cub",
    "seed": 0,
    "methods": {
        name: {
            "metrics": result["metrics"],
            "runtime_seconds": result["runtime_seconds"],
            "gpu_hours": result["gpu_hours"],
            "trainable_parameters": result[
                "trainable_parameters"
            ],
            "peak_cuda_memory_bytes": result[
                "peak_cuda_memory_bytes"
            ],
        }
        for name, result in results.items()
    },
    "deltasub_v2_minus_selex": {
        metric: (
            float(ours["metrics"][metric])
            - float(baseline["metrics"][metric])
        )
        for metric in metrics
    },
    "deltasub_branches": ours["branch_metrics"],
    "deltasub_diagnostics": ours["diagnostics"],
}

output = root / "comparison.json"
output.write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("metric          selex      deltasub-v2      delta")
for metric in metrics:
    left = float(baseline["metrics"][metric])
    right = float(ours["metrics"][metric])
    print(
        f"{metric:<8}     {left:.6f}     "
        f"{right:.6f}     {right-left:+.6f}"
    )

print()
print(f"comparison: {output}")
PY
