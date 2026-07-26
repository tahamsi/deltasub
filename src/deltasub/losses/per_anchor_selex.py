from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class PerAnchorSelEx:
    unsupervised: torch.Tensor
    supervised: torch.Tensor
    hierarchical: torch.Tensor
    total: torch.Tensor
    pseudo_label_confidence: torch.Tensor
    labelled: torch.Tensor


def _masked_logsumexp(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask, float("-inf"))
    return torch.logsumexp(masked, dim=-1)


def masked_contrastive_per_anchor(
    logits: torch.Tensor, positive_mask: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    positive_mask = positive_mask & valid_mask
    valid_count = valid_mask.sum(-1)
    positive_count = positive_mask.sum(-1)
    safe_valid = valid_count > 0
    safe_positive = positive_count > 0
    denominator = _masked_logsumexp(logits, valid_mask)
    log_prob = logits - denominator.unsqueeze(-1)
    value = -(log_prob.masked_fill(~positive_mask, 0).sum(-1) / positive_count.clamp_min(1))
    return torch.where(safe_valid & safe_positive, value, torch.zeros_like(value))


def per_anchor_selex(
    logits: torch.Tensor,
    unsupervised_positive_mask: torch.Tensor,
    supervised_positive_mask: torch.Tensor,
    hierarchical_positive_masks: list[torch.Tensor],
    valid_mask: torch.Tensor,
    labelled: torch.Tensor,
    pseudo_label_confidence: torch.Tensor,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PerAnchorSelEx:
    unsupervised = masked_contrastive_per_anchor(logits, unsupervised_positive_mask, valid_mask)
    supervised = masked_contrastive_per_anchor(logits, supervised_positive_mask, valid_mask)
    supervised = supervised * labelled.to(logits.dtype)
    if hierarchical_positive_masks:
        levels = [
            masked_contrastive_per_anchor(logits, mask, valid_mask)
            for mask in hierarchical_positive_masks
        ]
        hierarchical = torch.stack(levels).mean(0)
    else:
        hierarchical = torch.zeros_like(unsupervised)
    total = (
        weights[0] * unsupervised
        + weights[1] * supervised
        + weights[2] * hierarchical
    )
    return PerAnchorSelEx(
        unsupervised=unsupervised,
        supervised=supervised,
        hierarchical=hierarchical,
        total=total,
        pseudo_label_confidence=pseudo_label_confidence,
        labelled=labelled,
    )
