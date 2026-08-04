#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/ubuntu/research/deltasub
source .venv/bin/activate

export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0

ROOT="artifacts/deltasub_residual/cub/seed_0"
FULL_CONFIG="configs/deltasub_v2/cub_seed0_residual_full.yaml"
FULL_RESULT="$ROOT/full/result.json"
COMPARISON="$ROOT/full/comparison.json"

echo "===== PRACTICAL DELTASUB RESIDUAL ====="

python -m \
  deltasub.experiment.deltasub_residual_training \
  --config "$FULL_CONFIG" \
  --resume

echo
echo "===== MATCHED SEED-0 GATE ====="

python -m \
  deltasub.experiment.deltasub_residual_compare \
  --candidate-result "$FULL_RESULT" \
  --output "$COMPARISON" \
  --bootstrap-draws 1000

GATE="$(
python - <<'PY'
import json
from pathlib import Path

value = json.loads(
    Path(
        "artifacts/deltasub_residual/"
        "cub/seed_0/full/comparison.json"
    ).read_text(encoding="utf-8")
)

print(value["gate"])
PY
)"

if [ "$GATE" != "passed" ]; then
  echo
  echo "Full practical method failed the gate."
  echo "Ablation training was not launched."
  exit 0
fi

echo
echo "===== PREDEFINED ABLATIONS ====="

for NAME in \
  local_only \
  prediction_residual \
  orthogonal_residual
do
  python -m \
    deltasub.experiment.deltasub_residual_training \
    --config \
    "configs/deltasub_v2/cub_seed0_residual_${NAME}.yaml" \
    --resume
done

echo
echo "===== ABLATION SUMMARY ====="

python - <<'PY'
import json
from pathlib import Path

root = Path(
    "artifacts/deltasub_residual/"
    "cub/seed_0"
)

comparison = json.loads(
    (
        root
        / "full"
        / "comparison.json"
    ).read_text(encoding="utf-8")
)

baseline = comparison["baseline"]

order = (
    "local_only",
    "prediction_residual",
    "orthogonal_residual",
    "full",
)

summary = {
    "schema_version": (
        "deltasub-residual.ablation.v1"
    ),
    "baseline": baseline,
    "variants": {},
}

print(
    "method                    "
    "all       old       new      hmean"
)

print(
    f"{'matched_selex':<25}"
    f"{baseline['all']:>9.6f}"
    f"{baseline['old']:>10.6f}"
    f"{baseline['new']:>10.6f}"
    f"{baseline['hmean']:>11.6f}"
)

for name in order:
    result = json.loads(
        (
            root
            / name
            / "result.json"
        ).read_text(encoding="utf-8")
    )

    metrics = {
        key: float(
            result["metrics"][key]
        )
        for key in (
            "all",
            "old",
            "new",
            "hmean",
        )
    }

    delta = {
        key: (
            metrics[key]
            - baseline[key]
        )
        for key in metrics
    }

    summary["variants"][name] = {
        "metrics": metrics,
        "delta_vs_baseline": delta,
        "trainable_parameters": (
            result[
                "trainable_parameters"
            ]
        ),
        "runtime_seconds": (
            result["runtime_seconds"]
        ),
        "peak_cuda_memory_bytes": (
            result[
                "peak_cuda_memory_bytes"
            ]
        ),
    }

    print(
        f"{name:<25}"
        f"{metrics['all']:>9.6f}"
        f"{metrics['old']:>10.6f}"
        f"{metrics['new']:>10.6f}"
        f"{metrics['hmean']:>11.6f}"
    )

path = root / "ablation_summary.json"

path.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(f"summary: {path}")
PY
