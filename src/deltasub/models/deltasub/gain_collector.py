from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ...utils.reproducibility import preserve_rng_state


@dataclass
class DeterminismReport:
    loss_difference: float
    cosine_distance: float


def normalized_gain(base_loss: torch.Tensor, split_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = base_loss - split_loss
    return raw, raw / (base_loss.abs() + 1e-6)


@torch.no_grad()
def repeated_base_check(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    loss_tolerance: float = 1e-7,
    feature_tolerance: float = 1e-7,
) -> DeterminismReport:
    model.eval()
    with preserve_rng_state():
        first = model(images)
    with preserve_rng_state():
        second = model(images)
    first_loss = F.cross_entropy(first, labels, reduction="none")
    second_loss = F.cross_entropy(second, labels, reduction="none")
    loss_difference = (first_loss - second_loss).abs().max().item()
    cosine_distance = (1 - F.cosine_similarity(first, second, dim=-1)).abs().max().item()
    report = DeterminismReport(loss_difference, cosine_distance)
    if loss_difference >= loss_tolerance or cosine_distance >= feature_tolerance:
        raise RuntimeError(f"gain determinism check failed: {report}")
    return report


@torch.no_grad()
def collect_tiny_gains(model, images, labels, candidates: torch.Tensor) -> list[dict]:
    model.eval()
    base_logits = model(images)
    base_losses = F.cross_entropy(base_logits, labels, reduction="none")
    records = []
    for row in range(len(images)):
        for patch in candidates[row]:
            selections = [torch.empty(0, dtype=torch.long, device=images.device) for _ in images]
            selections[row] = patch.reshape(1)
            split = model(images, selections)
            split_loss = F.cross_entropy(split, labels, reduction="none")[row]
            raw, gain = normalized_gain(base_losses[row], split_loss)
            records.append(
                {
                    "image_id": row,
                    "patch_index": int(patch),
                    "base_loss": float(base_losses[row]),
                    "split_loss": float(split_loss),
                    "raw_gain": float(raw),
                    "normalized_gain": float(gain),
                }
            )
    return records
