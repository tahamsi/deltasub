# Baseline status

Status date: 2026-07-29. Shared synthetic infrastructure has been implemented and tested,
but no publication baseline has been completed or run. No benchmark results are claimed.
Pinned details are recorded in `third_party/manifest.yaml`.

M1 data foundations are complete and covered by local synthetic filesystem fixtures.
No real dataset, split file, benchmark manifest, or official pretrained checkpoint is
present. The exact pinned DINOv2 source checkout is available for architecture tests.
M2 now has a real training path; only test-only CPU fixtures ran.

| Required name | Upstream status | Original protocol | Planned integration | Current status | Principal incompatibility / risk |
|---|---|---|---|---|---|
| ViT / DINOv2 + SelEx | Official DINOv2 and SelEx code available | Fine-grained GCD for SelEx; DINOv2 backbone | Pinned source, strict full-state load, exact-reference gate | M2 CPU repair validated; CUDA/real run N/A | Real inputs, pretrained checkpoint, and CUDA validation are absent |
| TransFG-GCD-Port | Official MIT code available | Supervised FGVC, ViT-B/16, typically 448 px | Faithful port if feasible | Not started | Attention part selection and supervised contrastive recipe differ from GCD |
| CFViT-GCD-Port | Official Apache-2.0 code available | ImageNet classification, coarse-to-fine dynamic inference | Faithful/approximate port to be determined | Not started | LV-ViT/DeiT-era two-stage architecture differs from DINOv2 |
| LFViT-GCD-Port | Official code/checkpoints visible | ImageNet classification, DeiT-S, localization/focus stages | Unavailable pending license | Blocked | Repository has no root license; source reuse is not authorized |
| MSViT-GCD-Reimplementation | Paper and batch-shaping utility available | ImageNet classification and segmentation | Clean-room reimplementation | Not started | Public repository does not expose a complete end-to-end MSViT model |
| DART-GCD-Port | Official code available; license conflict noted | ImageNet classification plus dense/video tasks | Faithful tokenizer port if compatible | Not started | Quantile region tokenizer changes the grid; README/license mismatch |
| SATA-GCD-Port | Paper available; cited repository unavailable | Robust image classification; token grouping before FFN | Unavailable | Blocked | No verifiable official, licensed code at cited URL |
| SpiralFovea-GCD-Reimplementation | Paper found; no verified code | Fine-grained classification; entropy-driven foveated grid | Clean-room reimplementation | Not started | Replaces grid with mixed-scale tokens; recent paper-only method |
| ARTA-Cls-Port | Paper found; no verified code | Dense semantic feature extraction/segmentation | Classification port only if defensible | Not started | Boundary allocation is designed for dense labels, not GCD |
| SubViT-Reimplementation | Paper found; no verified code | Fine-grained GCD with deletion-degradation router | Clean-room reimplementation | Not started | Must reproduce two-stage subdivision without claiming official status |
| DeltaSub | New method | Fine-grained GCD | Native implementation | M6 deterministic adaptive budget execution complete with a non-reportable CPU fixture; M7+ not started | CUDA and real pretrained checkpoint/data remain environment-dependent; central hypothesis is untested |

## M6 adaptive execution foundation

M6 retains all 256 parent tokens. Selecting parent `j` appends its three M3 Haar details;
it never replaces the parent. Canonical order is prefix, row-major parents, then selected
parents in descending router score (ascending parent index on exact ties), with details
in horizontal/vertical/diagonal order.

The implementation distinguishes selected parents, added details, effective spatial
tokens, effective total tokens, padded tokens, approximate attention-token pairs, router
multiply-add estimates, and measured latency. The fixture makes no FLOP, latency, or
wall-clock-saving claim. Padded and bucketed valid outputs are compared while padding is
excluded from meaningful outputs. The synthetic fixture uses a frozen test router and
transformer, a trainable classification head, and a bounded dual controller. No M4 gain
record or M5 cache is modified.

## Comparison policy

Only completed common-protocol runs with compatible DINOv2 initialization, 224 px input,
SelEx/SSB splits and objective, clustering/evaluation, seeds, precision, and matched
compute budgets may support direct claims. Official paper numbers and checkpoints belong
only in the table labelled **Original protocols; results are not directly comparable**.

An adapter remains `N/A` when faithful implementation is impossible. Vanilla ViT is never
used as a silent replacement. Failed experiments remain visible with their failure reason.

## M3 token foundation

M3 implements exact 14×14 parent extraction and direct TL/TR/BL/BR 7×7 subdivision.
The four child projectors copy four quadrants of the already loaded official DINOv2
ViT-B/14 projection, multiplying weights by four and copying its bias. Their mean
therefore equals the original parent projection at initialization within floating-point
tolerance. At every later step `c_q = r_q - mean(r) + p` enforces the parent mean.

Only the three fixed orthonormal Haar details (horizontal, vertical, diagonal) are
appended; the original parent is the low-pass representation and is never replaced.
Detail positions are the exact parent position plus a learned mode vector initialized
to zero. Prefixes and all 256 parents precede details; details follow ascending parent
index and mode order, with deterministic right padding for heterogeneous batches.

Generated complete official-architecture state dictionaries exercised the strict pinned
loading path with and without four register tokens. This is architecture validation,
not pretrained-checkpoint or benchmark evidence. No gains, router, or benchmark result
was produced.

## M4 gain foundation

The M4 primary label is the anchor-specific loss reduction, never the scalar batch
change. Base and counterfactual evaluations reuse identical materialized views, batch
membership/order, labels, hierarchy, pseudo-label confidence, confusion factors, model
state, precision, and RNG context. Only three parent-aware Haar details are added for
one anchor/parent candidate. Non-anchor loss changes caused by SelEx batch coupling are
stored as spillover diagnostics.

Gain records and caches bind dataset/split, official-backbone, M3 projector, SelEx gate,
configuration, batch-context, device/precision, and source provenance. Invalid anchors
remain explicitly flagged and are not converted into valid zero-gain labels. The M4
fixture uses a tiny test-only transformer and synthetic deterministic tensors; it is
diagnostic and non-reportable. M4 itself trained no router and ran no benchmark.

## M5 router foundation

The production M5 router is an O(256D) shared per-parent MLP. Its inputs are the local
pre-transformer parent embedding, the masked global mean of all parents, and normalized
row/column coordinates; optional learned parent positions are supported. It cannot accept
counterfactual tokens, transformer outputs, gain labels, or selection state as features.
It emits 256 ordered predicted gains and performs no token selection.

M4 records and router features join by exact sample, view layout, batch context, manifest,
checkpoint, and gain-configuration provenance. Deterministic SHA256 assignment keeps all
views, contexts, and candidate parents for a sample in one train/validation/test split.
Training optimizes the explicitly sampled mixture of a uniform coverage stream and an
immutable gain-informed stream. No default inverse-propensity correction is claimed:
unbiased evaluation uses the complete validation split. Bounded replay stores identifiers
and post-training residual priorities only, with deterministic sign-stratified eviction.

The loss combines valid-anchor Huber gain regression, deterministic within-anchor
logistic ranking above a target margin, and an optional positive-gain BCE auxiliary.
Validation reports regression, correlation, ranking, top-K diagnostic, sign, and quantile
calibration metrics with explicit undefined values. The M5 fixture is synthetic,
diagnostic, and non-reportable. No adaptive budget or benchmark run was executed.
