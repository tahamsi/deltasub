from __future__ import annotations

import math
import torch

from .attention import deterministic_topk


def metric(value: float | None, reason: str | None = None) -> dict:
    if value is None or not math.isfinite(float(value)):
        return {"defined": False, "value": None, "reason": reason or "undefined"}
    return {"defined": True, "value": float(value), "reason": None}


def compare_maps(a: torch.Tensor, b: torch.Tensor, k: int,
                 gains: torch.Tensor | None = None) -> dict:
    if a.shape != b.shape or a.ndim != 1 or a.numel() != 256:
        raise ValueError("maps must be finite [256]")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("maps must be finite")
    ia, ib = set(deterministic_topk(a[None], k)[0]), set(deterministic_topk(b[None], k)[0])
    union = ia | ib
    ra = torch.argsort(torch.argsort(a, stable=True), stable=True).float()
    rb = torch.argsort(torch.argsort(b, stable=True), stable=True).float()
    spearman = None if ra.std() == 0 or rb.std() == 0 else float(torch.corrcoef(torch.stack((ra, rb)))[0, 1])
    pairs = [(i, j) for i in range(256) for j in range(i + 1, 256)
             if a[i] != a[j] and b[i] != b[j]]
    agreement = (sum(bool((a[i] > a[j]) == (b[i] > b[j])) for i, j in pairs) / len(pairs)
                 if pairs else None)
    p = torch.softmax(a.float(), 0)
    positive = set(torch.where(gains > 0)[0].tolist()) if gains is not None else set()
    return {
        "top_k_overlap": len(ia & ib), "top_k_jaccard": metric(len(ia & ib) / len(union)
                                                               if union else 1.0),
        "spearman": metric(spearman, "constant rank"), "rank_agreement": metric(agreement, "no strict pairs"),
        "positive_gain_recall": metric(len(ia & positive) / len(positive), "no positive gains")
        if gains is not None else metric(None, "gain oracle unavailable"),
        "map_entropy": metric(float(-(p * p.clamp_min(1e-30).log()).sum())),
        "top_k_concentration": metric(float(p[list(ia)].sum()) if ia else 0.0),
        "center_selection_frequency": metric(sum(1 for x in ia if 4 <= x // 16 < 12 and 4 <= x % 16 < 12) / len(ia),
                                             "K=0"),
        "border_selection_frequency": metric(sum(1 for x in ia if not (4 <= x // 16 < 12 and 4 <= x % 16 < 12)) / len(ia),
                                             "K=0"),
    }
