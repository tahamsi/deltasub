"""Per-anchor decomposition of the pinned SelEx objective.

Protocol source: SarahRastegar/SelEx commit 569ee7085e779999502bd73ea92240f3d32fc84d,
``methods/contrastive_training/contrastive_training.py`` (MIT).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class PerAnchorSelEx:
    unsupervised: torch.Tensor
    supervised: torch.Tensor
    hierarchical_levels: tuple[torch.Tensor, ...]
    hierarchical: torch.Tensor
    total: torch.Tensor
    pseudo_label_confidence: torch.Tensor
    labelled: torch.Tensor
    valid: torch.Tensor


def euclidean_cdist(features: torch.Tensor) -> torch.Tensor:
    """Compute pinned SelEx Euclidean distances under the mixed-precision policy.

    SelEx uses ``torch.cdist(..., p=2)``: this helper deliberately preserves that
    definition.  FP32 stays FP32.  CUDA BF16 is promoted to an FP32 numerical
    island because CUDA ``cdist`` does not implement BF16 on supported PyTorch
    builds; the cast remains in autograd.  Other dtype/device combinations are
    rejected rather than being given different mathematics.
    """
    if features.dtype == torch.float32:
        return torch.cdist(features, features, p=2)
    if features.is_cuda and features.dtype == torch.bfloat16:
        promoted = features.float()
        return torch.cdist(promoted, promoted, p=2)
    raise ValueError(
        "SelEx Euclidean distance supports FP32 on CPU/CUDA and BF16 inputs "
        f"on CUDA via an FP32 distance island; got {features.device}/{features.dtype}"
    )


def _fp32_loss_inputs(
    features: torch.Tensor, confusion_factor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep normalization, logits, log-softmax, and reductions in FP32.

    BF16 is an input/storage policy only: the returned loss is intentionally
    FP32 and is never cast back merely to advertise mixed-precision execution.
    """
    if features.dtype == torch.float32 and confusion_factor.dtype == torch.float32:
        return features, confusion_factor
    if (
        features.is_cuda
        and confusion_factor.is_cuda
        and features.dtype == torch.bfloat16
        and confusion_factor.dtype == torch.bfloat16
    ):
        return features.float(), confusion_factor.float()
    raise ValueError(
        "SelEx floating inputs must both be FP32, or both CUDA BF16 under the "
        "BF16-input/FP32-distance policy"
    )


def _supcon(features: torch.Tensor, labels: torch.Tensor, temperature: float = .07, base_temperature: float = .07):
    batch, views = features.shape[:2]
    contrast = torch.cat(torch.unbind(features, 1), 0)
    logits = -euclidean_cdist(contrast) / temperature
    logits = logits - logits.max(1, keepdim=True).values.detach()
    same = labels[:, None].eq(labels[None]).to(features.dtype).repeat(views, views)
    not_self = ~torch.eye(batch * views, dtype=torch.bool, device=features.device)
    same = same * not_self
    log_prob = logits - torch.log((logits.exp() * not_self).sum(1, keepdim=True))
    positives = same.sum(1)
    valid = positives > 0
    values = -(temperature / base_temperature) * (same * log_prob).sum(1) / positives.clamp_min(1)
    return values.view(views, batch).mean(0), valid.view(views, batch).all(0)


def selex_per_anchor(
    features: torch.Tensor,
    labels: torch.Tensor,
    labelled: torch.Tensor,
    hierarchy_labels: tuple[torch.Tensor, ...],
    confusion_factor: torch.Tensor,
    *,
    temperature: float = 1.0,
    sup_con_weight: float = .35,
    unsupervised_smoothing: float = 1.0,
    pseudo_label_confidence: torch.Tensor | None = None,
) -> PerAnchorSelEx:
    """Exact vector decomposition whose mean follows the pinned training expression.

    CUDA BF16 features and confusion values are accepted as quantized inputs,
    then promoted for every numerically sensitive loss operation.  Consequently
    all component tensors and the final scalar reduction remain FP32.
    """
    batch, views, dim = features.shape
    features, confusion_factor = _fp32_loss_inputs(features, confusion_factor)
    normalized = F.normalize(features, dim=-1)
    flat = torch.cat(torch.unbind(normalized, 1), 0)
    pair = -euclidean_cdist(flat)
    same_view = torch.arange(batch, device=flat.device).repeat(views)
    positive = same_view[:, None].eq(same_view[None])
    not_self = ~torch.eye(batch * views, dtype=torch.bool, device=flat.device)
    logits = (pair[not_self].view(batch * views, -1) / temperature)
    positive = positive[not_self].view(batch * views, -1)
    confusion = confusion_factor[not_self].view(batch * views, -1)
    ordered = torch.cat((logits[positive].view(batch * views, -1), logits[~positive].view(batch * views, -1)), 1)
    soft = torch.cat((confusion[positive].view(batch * views, -1), confusion[~positive].view(batch * views, -1)), 1)
    target = F.one_hot(torch.zeros(batch * views, dtype=torch.long, device=flat.device), ordered.shape[1]).to(flat.dtype)
    target = target * (1 - unsupervised_smoothing) + soft * unsupervised_smoothing
    unsup = -(target * F.log_softmax(ordered, 1)).sum(1).view(views, batch).mean(0)

    if labelled.any():
        labelled_values, labelled_valid = _supcon(normalized[labelled], labels[labelled])
        supervised = torch.zeros(batch, dtype=features.dtype, device=features.device)
        # Upstream averages this term over the labelled subset.  Scaling embeds that
        # denominator in a length-B contribution vector whose batch mean is identical.
        supervised[labelled] = labelled_values * (batch / int(labelled.sum()))
    else:
        supervised = torch.zeros(batch, dtype=features.dtype, device=features.device)
        labelled_valid = torch.empty(0, dtype=torch.bool, device=features.device)
    levels = []
    level_valid = []
    for index, level_labels in enumerate(hierarchy_labels):
        value, valid = _supcon(normalized[..., : dim // (2 ** (index + 1))], level_labels)
        levels.append(value / (2 ** (index + 1)))
        level_valid.append(valid)
    hierarchical = sum(levels, torch.zeros_like(unsup))
    supervised_total = supervised + hierarchical
    total = (1 - sup_con_weight) * unsup + sup_con_weight * supervised_total / 2
    valid = torch.ones(batch, dtype=torch.bool, device=features.device)
    if labelled.any():
        valid[labelled] &= labelled_valid
    for item in level_valid:
        valid &= item
    confidence = pseudo_label_confidence if pseudo_label_confidence is not None else torch.ones_like(unsup)
    return PerAnchorSelEx(unsup, supervised, tuple(levels), hierarchical, total, confidence, labelled, valid)


# Compatibility surface retained for the M0 synthetic scaffold.
def masked_contrastive_per_anchor(logits, positive_mask, valid_mask):
    mask = positive_mask & valid_mask
    log_prob = logits - torch.logsumexp(logits.masked_fill(~valid_mask, -torch.inf), -1, keepdim=True)
    count = mask.sum(-1)
    return torch.where(count > 0, -(log_prob.masked_fill(~mask, 0).sum(-1) / count.clamp_min(1)), 0)


def per_anchor_selex(logits, unsupervised_positive_mask, supervised_positive_mask, hierarchical_positive_masks, valid_mask, labelled, pseudo_label_confidence, weights=(1., 1., 1.)):
    unsup = masked_contrastive_per_anchor(logits, unsupervised_positive_mask, valid_mask)
    sup = masked_contrastive_per_anchor(logits, supervised_positive_mask, valid_mask) * labelled
    levels = tuple(masked_contrastive_per_anchor(logits, x, valid_mask) for x in hierarchical_positive_masks)
    hier = torch.stack(levels).mean(0) if levels else torch.zeros_like(unsup)
    total = weights[0] * unsup + weights[1] * sup + weights[2] * hier
    valid = (unsupervised_positive_mask & valid_mask).any(-1)
    return PerAnchorSelEx(unsup, sup, levels, hier, total, pseudo_label_confidence, labelled, valid)
