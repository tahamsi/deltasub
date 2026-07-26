# DeltaSub

DeltaSub is a research repository for testing counterfactual value-of-subdivision
routing in fine-grained generalized category discovery. The repository currently
provides tested token geometry, Haar detail tokens, per-anchor contrastive reduction,
deterministic paired-gain machinery, two-stream candidate sampling, routing losses,
budget bucketing, manifests, result schemas, and a complete synthetic smoke pipeline.

It does **not** contain completed CUB, Aircraft, Cars, CIFAR-10, or ImageNet-100
experiments. Synthetic metrics are marked `synthetic_only: true` and are excluded from
paper tables. Several requested baselines remain unavailable or require clean-room
reimplementation; see `BASELINE_STATUS.md`.

## Local installation and validation

Use Python 3.10 or newer:

```bash
python -m pip install -e '.[dev]'
make smoke
```

The smoke command runs the unit and integration tests, checks the environment, executes
one tiny optimization step, verifies deterministic repeated base evaluation, collects
paired gains for a two-image subset, tests resume, and writes a complete synthetic run
under `artifacts/runs/synthetic_smoke`.

To rerun directly:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=src python -m deltasub.cli smoke \
  --output artifacts/runs/synthetic_smoke \
  --size 8 --seed 0 --device cpu --resume
```

## Running on one A100 80 GB

Clone or copy the repository onto the server, create an isolated environment, install a
CUDA-compatible PyTorch build, and install DeltaSub:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build appropriate for the server CUDA driver first.
python -m pip install -e '.[dev,data]'
nvidia-smi
bash scripts/run_a100_diagnostic.sh
```

The script uses only `CUDA_VISIBLE_DEVICES=0`, runs jobs sequentially, measures memory,
and performs a small CUDA smoke run. It does not invoke DDP or launch a real-data sweep.
The measured recommendations are written to `artifacts/hardware_profile.yaml`.

Before real research training, supply:

1. Legally obtained CUB, FGVC-Aircraft, and Stanford Cars data.
2. Exact SelEx/SSB class-split files pinned to the inspected upstream revision.
3. The official DINOv2 ViT-B/14 checkpoint and recorded SHA256.
4. A completed exact integration of the upstream SelEx objective whose per-anchor mean
   passes the scalar-equivalence test.

This repository intentionally refuses to present a synthetic implementation as a
reproduction of SelEx or any unavailable baseline.

## Result artifacts

Every synthetic smoke run writes the same required artifact envelope expected from a
real run:

```text
config.yaml
resolved_config.yaml
environment.json
git_commit.txt
dataset_manifest_checksum.txt
backbone_checkpoint_hash.txt
metrics.json
metrics.jsonl
selection_statistics.parquet
efficiency.json
compute.json
checkpoint_best.pt
checkpoint_last.pt
stdout.log
stderr.log
```

Inspect a run with:

```bash
python -m json.tool artifacts/runs/synthetic_smoke/metrics.json
python -m json.tool artifacts/runs/synthetic_smoke/compute.json
python - <<'PY'
import pandas as pd
print(pd.read_parquet("artifacts/runs/synthetic_smoke/selection_statistics.parquet"))
PY
```

Generate CSV, Markdown, and LaTeX tables from completed **non-synthetic** runs:

```bash
bash scripts/collect_results.sh
```

The command exits rather than aggregating synthetic metrics. Generated tables are placed
under `paper/generated_tables/`. Archive `artifacts/runs`, `artifacts/compute_report.csv`,
the generated table/figure directories, `DIAGNOSTIC_VERDICT.md`, and the Git commit SHA
together to preserve reproducibility.

## Scientific execution order

Follow `PLAN.md`: implement and pass M1–M4 gates, run CUB seed 0, warm up detail tokens,
collect a small deterministic M=4 gain cache, run the full-patch audit, and write the
decisive diagnostic. Core and full tiers must remain blocked unless the diagnostic is
positive. Original-protocol numbers must never be combined with common-protocol results.
