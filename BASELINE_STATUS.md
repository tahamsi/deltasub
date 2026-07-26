# Baseline status

Status date: 2026-07-26. No baseline has been implemented or run. No results are claimed.
Pinned details are recorded in `third_party/manifest.yaml`.

| Required name | Upstream status | Original protocol | Planned integration | Current status | Principal incompatibility / risk |
|---|---|---|---|---|---|
| ViT / DINOv2 + SelEx | Official DINOv2 and SelEx code available | Fine-grained GCD for SelEx; DINOv2 backbone | Official components in common harness | Not started | Per-anchor SelEx refactor must exactly preserve scalar reduction |
| TransFG-GCD-Port | Official MIT code available | Supervised FGVC, ViT-B/16, typically 448 px | Faithful port if feasible | Not started | Attention part selection and supervised contrastive recipe differ from GCD |
| CFViT-GCD-Port | Official Apache-2.0 code available | ImageNet classification, coarse-to-fine dynamic inference | Faithful/approximate port to be determined | Not started | LV-ViT/DeiT-era two-stage architecture differs from DINOv2 |
| LFViT-GCD-Port | Official code/checkpoints visible | ImageNet classification, DeiT-S, localization/focus stages | Unavailable pending license | Blocked | Repository has no root license; source reuse is not authorized |
| MSViT-GCD-Reimplementation | Paper and batch-shaping utility available | ImageNet classification and segmentation | Clean-room reimplementation | Not started | Public repository does not expose a complete end-to-end MSViT model |
| DART-GCD-Port | Official code available; license conflict noted | ImageNet classification plus dense/video tasks | Faithful tokenizer port if compatible | Not started | Quantile region tokenizer changes the grid; README/license mismatch |
| SATA-GCD-Port | Paper available; cited repository unavailable | Robust image classification; token grouping before FFN | Unavailable | Blocked | No verifiable official, licensed code at cited URL |
| SpiralFovea-GCD-Reimplementation | Paper found; no verified code | Fine-grained classification; entropy-driven foveated grid | Clean-room reimplementation | Not started | Replaces grid with mixed-scale tokens; recent paper-only method |
| ARTA-Cls-Port | Paper found; no verified code | Dense semantic feature extraction/segmentation | Classification port only if defensible | Not started | Boundary allocation is designed for dense labels, not GCD |
| SubViT-Reimplementation | Paper found; no verified code | Fine-grained GCD with deletion-degradation router | Clean-room reimplementation | Not started | Must reproduce two-stage subdivision without claiming official status |
| DeltaSub | New method | Fine-grained GCD | Native implementation | Not started | Central hypothesis must survive deterministic paired diagnostic |

## Comparison policy

Only completed common-protocol runs with compatible DINOv2 initialization, 224 px input,
SelEx/SSB splits and objective, clustering/evaluation, seeds, precision, and matched
compute budgets may support direct claims. Official paper numbers and checkpoints belong
only in the table labelled **Original protocols; results are not directly comparable**.

An adapter remains `N/A` when faithful implementation is impossible. Vanilla ViT is never
used as a silent replacement. Failed experiments remain visible with their failure reason.
