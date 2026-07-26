# DeltaSub implementation plan

Status date: 2026-07-26

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
- [ ] M2: DINOv2/SelEx baseline, per-anchor SelEx decomposition, strict scalar-equivalence
  test.
- [ ] M3: direct child projection, Haar detail representation, position encodings,
  parent consistency, reconstruction/calibration tests.
- [ ] M4: deterministic paired counterfactual engine, repeated-base acceptance checks,
  incremental compatible Parquet cache, interruption/resume.
- [ ] M5: two-stream candidate sampler, stratified replay buffer, router loss/training,
  calibration and ranking metrics.
- [ ] M6: fixed/adaptive budget control, discrete K buckets, effective/padded accounting,
  measured latency and memory.
- [ ] M7: SubViT reimplementation and decisive deletion-versus-gain diagnostic; write an
  evidence-based `DIAGNOSTIC_VERDICT.md`.
- [ ] M8: core adapters (ViT, SubViT, DeltaSub, MSViT, DART).
- [ ] M9: three-seed core experiments on CUB, Aircraft, Cars; essential ablations and
  efficiency profiling.
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

## Immediate next milestone: M2 (not started)

M2 may begin only as a separate milestone. It must integrate official DINOv2 and SelEx
components and demonstrate exact per-anchor/scalar equivalence; M1 does not claim either
integration.

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
