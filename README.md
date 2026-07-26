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
2. Exact SelEx/SSB class-split files from the pinned local upstream checkout. JSON,
   YAML, and trusted pinned pickle formats are supported; their SHA256 and provenance
   are recorded in `split_validation.json`.
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

## M1 data manifests

The M1 manifest schema is versioned and requires sample identity, dataset/path, original
class ID/name, known/novel and labelled/unlabelled assignments, train/test membership,
optional bounding boxes, source archive SHA256, and exact split provenance. JSONL records
are sorted canonically before serialization, so repeated preparation from the same data,
split file, and configuration produces the same bytes and checksum.

Preparation is local-only. CUB and Aircraft retain their explicit official download
command, Cars requires `--source manual`, and ImageNet-100 requires a licensed
ImageNet-1K root and is never downloaded. CIFAR-10 binary batches are losslessly
extracted to local PPM files. Every preparation requires archive checksum provenance and
an exact local split file; absent inputs fail with an actionable error.

Example command surface:

```bash
python -m deltasub.cli references inspect
python -m deltasub.cli data prepare cub --root DATA --split-file SPLIT.json --archive CUB.tgz
python -m deltasub.cli data prepare aircraft --root DATA --split-file SPLIT.pkl --archive AIRCRAFT.tar.gz
python -m deltasub.cli data prepare cars --root DATA --source manual --split-file SPLIT.pkl --archive CARS.tgz
python -m deltasub.cli data prepare cifar10 --root DATA --split-file SPLIT.json --archive CIFAR.tar.gz
python -m deltasub.cli data prepare imagenet100 --root OUTPUT --imagenet-root IMAGENET \
  --split-file SPLIT.json
python -m deltasub.cli data validate cub --root DATA
python -m deltasub.cli data validate-all --root DATASETS
```

For ImageNet-100, `OUTPUT/splits/imagenet100_wnids.txt` must additionally contain the
exact 100 unique WNIDs from the pinned split source. A local
`source_archive.sha256` may be used where the licensed source is already extracted.
No real dataset counts or checksums are claimed by this repository.

## CLI coverage

The currently exposed CLI surface is:

```text
doctor
doctor memory
smoke
data download {aircraft,cub}
data prepare {aircraft,cars,cifar10,cub,imagenet100}
data validate {aircraft,cars,cifar10,cub,imagenet100}
data validate-all
references inspect
paper build-all
```

Every command and nested `--help` path is covered by `tests/integration/test_cli.py`.
Download dispatch is tested without downloading multi-gigabyte archives; official URLs
and dataset checksums are exercised by the downloader when the user invokes it.

Stages 0–5 training, gain collection, router training, diagnosis, audits, GCD evaluation,
and efficiency evaluation are not exposed. Those commands belong to later milestones;
attempting to use them produces an argparse “invalid choice” error rather than silently
running a substitute implementation.

## Scientific execution order

Follow `PLAN.md`: implement and pass M1–M4 gates, run CUB seed 0, warm up detail tokens,
collect a small deterministic M=4 gain cache, run the full-patch audit, and write the
decisive diagnostic. Core and full tiers must remain blocked unless the diagnostic is
positive. Original-protocol numbers must never be combined with common-protocol results.
