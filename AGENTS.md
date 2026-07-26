# DeltaSub contributor instructions

## Scientific integrity

- Never fabricate metrics, timings, provenance, completion status, or diagnostic outcomes.
- Keep common-protocol results separate from original-protocol reference results.
- Never substitute vanilla ViT for an unavailable baseline. Report `N/A` with a reason.
- Call ports and reimplementations by their explicit names; do not imply they are official.
- Preserve upstream license notices and attribution. Do not copy code from a source whose
  license is absent or incompatible.
- Do not start the full tier unless `DIAGNOSTIC_VERDICT.md` records a positive verdict
  based on completed deterministic experiments and the explicit CLI confirmation is given.

## Execution constraints

- Target exactly one NVIDIA A100 80 GB at `cuda:0`; do not introduce DDP or multi-GPU
  assumptions.
- Run training, gain collection, audits, and evaluation sequentially and resumably.
- Long-running jobs must use atomic checkpoints and report elapsed/remaining time,
  GPU-hours, peak VRAM, throughput, and forward-pass counts.
- Counterfactual gain collection must use deterministic paired evaluations, unchanged
  augmented tensors and batch context, frozen assignments/prototypes, and candidate
  microbatches.
- Never automatically download ImageNet. Stanford Cars requires a documented manual
  source unless the user explicitly authorizes a Kaggle slug.

## Required gates

Before GPU training, all of the following must pass:

1. Per-anchor SelEx mean matches the original scalar reduction.
2. Repeated base loss and feature determinism checks pass.
3. Dataset split and manifest tests pass.
4. Child-token shape and Haar reconstruction tests pass.

Stop at a failed critical gate, retain logs, and update `PLAN.md` and
`BASELINE_STATUS.md`. The default paper tier is `diagnostic`.

## Engineering conventions

- Use a `src/` package layout and the single `python -m deltasub.cli` entry point.
- Configurations and result schemas are versioned; cached gains require exact compatibility.
- Use exploration records only for unbiased regression, calibration, and population
  estimates. Keep prioritized records explicitly labelled.
- Tests must have CPU smoke coverage; GPU tests must be marked and runnable on one GPU.
- Tables aggregate completed artifacts only and must never train or evaluate models.
- At each milestone, run applicable tests, save command output, update planning/status
  documents, and commit the coherent milestone.
