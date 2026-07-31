from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class RouterLossOutput:
    total: torch.Tensor
    regression: torch.Tensor
    ranking: torch.Tensor
    sign: torch.Tensor
    regularization: torch.Tensor
    valid_count: int
    pair_count: int


def deterministic_pairs(
    targets: torch.Tensor, group_ids: torch.Tensor, valid: torch.Tensor,
    *, target_margin: float, maximum_pairs_per_anchor: int,
) -> list[tuple[int, int]]:
    if target_margin < 0 or maximum_pairs_per_anchor < 0:
        raise ValueError("invalid ranking configuration")
    pairs = []
    for group in sorted(set(group_ids.detach().cpu().tolist())):
        indices = [i for i in range(len(targets)) if bool(valid[i]) and int(group_ids[i]) == group]
        candidates = []
        for left in indices:
            for right in indices:
                difference = float(targets[left].detach().float() - targets[right].detach().float())
                if difference > target_margin:
                    candidates.append((-difference, left, right))
        candidates.sort()
        pairs.extend((left, right) for _, left, right in candidates[:maximum_pairs_per_anchor])
    return pairs


def router_loss(
    scores: torch.Tensor, targets: torch.Tensor, group_ids: torch.Tensor,
    valid: torch.Tensor, *, regression_weight: float = 1., ranking_weight: float = 1.,
    sign_weight: float = 0., huber_delta: float = 1., target_margin: float = 0.,
    maximum_pairs_per_anchor: int = 1024, sign_epsilon: float = 0.,
    gain_clip: float | None = None, regularization: torch.Tensor | None = None,
) -> RouterLossOutput:
    if not (scores.shape == targets.shape == group_ids.shape == valid.shape):
        raise ValueError("loss inputs must have identical one-dimensional shapes")
    if any(x < 0 for x in (regression_weight, ranking_weight, sign_weight)) or (
        regression_weight + ranking_weight + sign_weight == 0
    ):
        raise ValueError("loss weights must be nonnegative and not all zero")
    work_scores, work_targets = scores.float(), targets.float()
    if gain_clip is not None:
        if gain_clip <= 0:
            raise ValueError("training gain clip must be positive")
        work_targets = work_targets.clamp(-gain_clip, gain_clip)
    selected_scores, selected_targets = work_scores[valid], work_targets[valid]
    zero = work_scores.sum() * 0
    regression = (
        F.huber_loss(selected_scores, selected_targets, delta=huber_delta, reduction="mean")
        if selected_scores.numel() else zero
    )
    pairs = deterministic_pairs(work_targets, group_ids, valid, target_margin=target_margin,
                                maximum_pairs_per_anchor=maximum_pairs_per_anchor)
    ranking = (
        torch.stack([F.softplus(-(work_scores[a] - work_scores[b])) for a, b in pairs]).mean()
        if pairs else zero
    )
    sign = (
        F.binary_cross_entropy_with_logits(selected_scores, (selected_targets > sign_epsilon).float())
        if sign_weight and selected_scores.numel() else zero
    )
    regularization = zero if regularization is None else regularization.float()
    total = regression_weight * regression + ranking_weight * ranking + sign_weight * sign + regularization
    return RouterLossOutput(total, regression, ranking, sign, regularization,
                            int(valid.sum()), len(pairs))
