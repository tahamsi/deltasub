# DeltaSub implementation plan

Status date: 2026-07-30

DeltaSub tests whether counterfactual subdivision gain is a better routing target than
attention, deletion importance, or cheap detail scores for fine-grained generalized
category discovery. Work is deliberately staged: no full sweep can precede a decisive,
deterministic diagnostic.

## Current state

- [x] M0: inspect cited papers and repositories; record pinned upstream heads, licenses,
  protocols, incompatibilities, and unavailable code. Local provenance and documentation
  invariants are covered by `tests/unit/test_m0_provenance.py`.
- [x] M1: environment, CLI skeleton, deterministic dataset manifests, exact SSB/SelEx
  split validation, and foundational unit tests.
- [x] M2: repaired pinned-source DINOv2 loading, Stage-0 optimization, and executable
  exact-reference SelEx gate. CPU and pinned architecture validation are complete;
  marked CUDA coverage remains environment-skipped rather than fabricated.
- [x] M3: exact direct child projection, Haar detail representation, parent-aware
  positions, hard parent consistency, stable sequence assembly, and reconstruction tests.
- [x] M4: deterministic paired real-token counterfactual engine, immutable batch-context
  hashes, versioned incremental Parquet cache, validation, and interruption/resume.
- [x] M5: two-stream candidate sampler, stratified replay buffer, router loss/training,
  calibration and ranking metrics.
- [x] M6: deterministic fixed/threshold/dual budget control, score-ranked hard top-K,
  adaptive sequence assembly, exact-length/configured buckets, effective/padded
  accounting, strict checkpoint/resume, and a synthetic non-reportable fixture.
- [x] M7: deterministic clean-room SubViT diagnostic reference and comparison framework.
  Only synthetic non-reportable fixtures ran; a real verdict remains unavailable.
- [x] M8: common-protocol adapters and synthetic comparison fixtures for ViT, SubViT,
  DeltaSub, MSViT, and DART. All remain evidence-derived `fixture_only`.
- [ ] M9: strict seed-0 real-asset preflight and campaign/verdict layer implemented;
  production execution remains blocked on the missing M6 trainer and GCD-v2 evaluator.
  No diagnostic, core, or full experiment has run.
- [ ] M10: defensible lower-priority ports, secondary datasets, transfer and robustness.
- [ ] M11: artifact audit, generated tables/figures, compute report and reproduction report.

## M0 findings and decisions

1. SelEx and the original GCD repository are official MIT-licensed implementations.
   SelEx depends on batch contrastive/hierarchical objectives and `kmeans_pytorch`; M2
   must preserve its exact masks, weights, pseudo-label context, and reduction.
2. DINOv2 is Apache-2.0 and exposes ViT-B/14 (768-dimensional patch embeddings). Its
   standard code path is a suitable official backbone integration, but DeltaSub token
   injection will require a carefully tested adapter.
3. TransFG, CF-ViT, LF-ViT, MSViT, DART, SATA, SpiralFovea, ARTA, and SubViT use different
   tasks, backbones, resolutions, or token semantics. Common-protocol variants are ports
   or reimplementations, never official reproductions.
4. The public MSViT URL is a generalized batch-shaping utility repository, not a complete
   end-to-end MSViT model. The baseline must therefore be named
   `MSViT-GCD-Reimplementation`.
5. The cited SATA GitHub URL was unavailable on 2026-07-26. SATA remains `unavailable`
   until an official, licensed repository can be verified.
6. LF-ViT has public code and checkpoints but no root license. Inspection is permitted;
   copying or adapting its source is blocked pending license clarification.
7. SpiralFovea, ARTA, and SubViT are recent paper-only references for which no verified
   official repository was found during M0. Any implementation must be clean-room and
   explicitly labelled a port/reimplementation.
8. DART's root `LICENSE` is Apache-2.0, while its README says MIT. The root license is
   treated as authoritative and the discrepancy is retained as a provenance warning.

## Implemented validation scaffold

The repository now includes a dependency-light synthetic pipeline used only to test
geometry, loss decomposition, deterministic gain mechanics, sampling, routing, budgeting,
checkpoint/resume, artifact schemas, and CPU/CUDA execution. It is not a SelEx or DINOv2
reproduction and its metrics are excluded from paper aggregation.

## M1 completion

M1 provides a versioned JSONL manifest, canonical ordering and SHA256 serialization,
local parsers for CUB-200-2011, FGVC-Aircraft, Stanford Cars, CIFAR-10, and ImageNet-100,
explicit Cars/ImageNet legal guards, pinned split-file provenance validation, and CPU
fixture tests. No benchmark manifest has been generated: licensed datasets and the exact
pinned upstream split files are not present in this repository.

## M2 repair status

M2 verifies a local pinned DINOv2 checkout and source hashes, constructs official
ViT-B/14, and strictly loads the supplied complete state dictionary. Stage-0 performs
real two-view image optimization and resumable atomic checkpointing. The SelEx gate
executes the isolated MIT snapshot and is re-executed by validation. No benchmark ran.

## M3 completion

Every 224×224 input is partitioned row-major into 256 non-overlapping 14×14 parents,
then directly into TL, TR, BL, BR 7×7 children. Four quadrant projectors are initialized
from the loaded official DINOv2 convolution, and hard mean consistency is imposed before
three orthonormal horizontal, vertical, and diagonal Haar details are formed. Original
parents remain in the sequence; a supplied mask appends exactly three ordered details
per selected parent. Positions retain the parent-cell component plus a zero-initialized
learned mode embedding. CPU fixture and exact pinned official-architecture integration
tests passed; CUDA was unavailable and no benchmark or real checkpoint was run.

## M4 completion

M4 defines `gain(i,j) = per_anchor_loss_base(i) -
per_anchor_loss_counterfactual(i,j)`. The counterfactual retains every original prefix
and parent token and adds only parent `j`'s three M3 Haar details to both fixed views of
anchor `i`. Transformer inference is eval-only and unpadded per sample; the complete
fixed SelEx batch is then recomputed, so non-anchor spillover from the contrastive batch
is measured separately from the primary anchor label.

Batch contexts bind ordered IDs/views, materialized image hashes, augmentation metadata,
all label/pseudo-label/confusion tensors, model/projector/head states, configuration,
precision/device/mode, and source commit. Base reuse fails closed on any changed field.
The content-addressed cache uses explicit Parquet shards with atomic writes, checksummed
indices, deterministic ordering, streaming reads, idempotent resume, and recomputed
validation. The deterministic tiny fixture is synthetic and non-reportable.

## M5 completion

M5 adds a compact shared per-parent gain router that consumes only the 256 frozen
pre-transformer DINOv2 parent embeddings, their global mean, and normalized row/column
coordinates. It predicts all 256 M4 gains without selecting or inserting tokens.
Training joins features and labels by the complete immutable M4 record identity, splits
at stable sample identity, combines deterministic uniform-coverage and immutable
gain-informed streams, and uses bounded deterministic stratified-priority replay.

The objective is mean valid-anchor Huber regression plus mean deterministic within-anchor
logistic ranking, with an optional positive-gain BCE auxiliary. Invalid anchors remain
auditable but are excluded from optimization. Complete validation records determine best
checkpoints through a configured ranking/correlation metric. The synthetic fixture
exercises atomic last/best checkpoints, replay eviction and restore, exact resume, and
independent deterministic CPU reruns. It is explicitly non-reportable.

## M6 completion

M6 scores all 256 frozen pre-transformer parent embeddings, retains every parent, and
adds exactly three M3 Haar details for each selected parent. Selection orders parents by
descending router score with exact ties resolved by ascending parent index. Prefixes
remain first, followed by all row-major parents and score-ranked selected details in M3
horizontal/vertical/diagonal order.

Budget units are never conflated: `K` selected parents, `3K` added details, `256+3K`
spatial tokens, and `prefix+256+3K` total tokens. Fixed-K, threshold-with-bounds, and
dual-threshold modes are deterministic. With positive violation defined as realized
minus target usage, the projected update is
`lambda <- clamp(lambda + dual_lr * violation, lambda_min, lambda_max)` and uses only
completed training intervals.

Masked padded execution and deterministic length-bucketed execution restore original
sample order and report both effective and padded tokens. Hard top-K has no ordinary
gradient; router fine-tuning/surrogates are not part of M6. The fixture trains only a
small head, proves exact controller resume and frozen-state equality, and is synthetic,
diagnostic, and non-reportable. No real benchmark, pretrained checkpoint, or dataset ran.

## M7 diagnostic reference completion

M7 implements an isolated, source-attributed clean-room reference to the mechanism
described in the SubViT paper. It is neither the unavailable official implementation
nor an exact paper reproduction. ATS keeps all original parents and appends `f*f`
direct spatial children per selected parent. At `f=2`, this is four children, not
DeltaSub's three Haar details.

The framework extracts per-head CLS-to-parent attention from a configurable official
DINOv2 block, supports seeded head schedules, performs deterministic head-wise
feature-degradation selection, and distils the selected map into a separate
pre-transformer single-map router. The loss uses FP32 temperature-scaled map KL,
strict-pair logistic ranking, and a 256-entry top-K mask BCE. Ties select the lowest
parent/head index; empty strict-pair sets contribute finite zero.

Only synthetic fixtures exercised geometry, degradation, router optimization, exact
resume, comparisons, and artifact hashing. Teacher deletion forwards are training
diagnostics only. Router inference emits one map and needs one transformer pass. No
benchmark, paper table, or scientific verdict was produced. M8 subsequently added only
common-protocol adapter fixtures; M9 remains incomplete.

## M8 common baseline adapters

M8 defines a versioned strict input/output contract, evidence-derived availability,
common training artifacts, and comparisons by effective tokens, padded tokens,
approximate attention-token pairs, or optional synchronized latency. Token counts are
not FLOPs and equal selected-region K does not establish equal compute.

The exact labels are `ViT / DINOv2 + SelEx`, `DeltaSub`,
`SubViT-Reimplementation`, `MSViT-GCD-Reimplementation`, and `DART-GCD-Port`.
The DART adapter is explicitly a synthetic interface fixture, not a verified faithful
port. Production requests fail closed. No real benchmark ran and M9 remains incomplete.

## Hard gates and stop rules

- No training until per-anchor reduction, base determinism, split validation, child shape,
  and Haar reconstruction tests all pass.
- No core tier unless the diagnostic is non-negative.
- No full tier without a positive diagnostic and explicit confirmation.
- Stop the full study if any predeclared diagnostic failure condition is met.
- Missing data, checkpoints, licenses, or faithful integrations produce a visible gap,
  never an improvised replacement.

## Single-GPU execution order

After implementation gates pass: unit/smoke tests; CUB baseline seed 0; CUB detail warm-up;
small M=4 gain collection; approximately 100-image full-patch audit; decisive diagnostic;
then, only on a positive verdict, CUB gains/router/budget stages followed sequentially by
three-seed CUB, Aircraft, Cars, core baselines, ablations, efficiency, and lower-priority
work.

## M9 production implementation status (2026-08-01)

The runnable code path is implemented but no real training has been launched. It uses
manifest-only CUB/Aircraft loading, strict pinned DINOv2, M4 paired gains plus emitted M5
feature caches, M5 router training, native M3 Haar details, fixed-K adaptive training,
and the pinned GCD-v2 Hungarian evaluator. Every campaign unit is a separate resumable
dataset/method/seed command. Core and full configurations are under
`configs/publication/`; Cars remains explicitly unavailable by user choice. The full
tier is still gated by `DIAGNOSTIC_VERDICT.md` and explicit CLI confirmation; its mere
configuration does not authorize execution. No result or verdict is claimed.
