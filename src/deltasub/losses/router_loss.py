from __future__ import annotations

import torch
from torch.nn import functional as F


def pairwise_ranking_loss(
    prediction: torch.Tensor, target: torch.Tensor, image_ids: torch.Tensor, margin: float
) -> torch.Tensor:
    losses = []
    for image_id in image_ids.unique():
        keep = image_ids == image_id
        pred, truth = prediction[keep], target[keep]
        if len(pred) < 2:
            continue
        diff_t = truth[:, None] - truth[None, :]
        diff_p = pred[:, None] - pred[None, :]
        mask = torch.triu(diff_t.abs() >= margin, diagonal=1)
        if mask.any():
            losses.append(F.softplus(-diff_t.sign()[mask] * diff_p[mask]).mean())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0


def calibration_loss(prediction: torch.Tensor, target: torch.Tensor, bins: int = 5) -> torch.Tensor:
    if prediction.numel() == 0:
        return prediction.sum() * 0
    boundaries = torch.quantile(prediction.detach(), torch.linspace(0, 1, bins + 1, device=prediction.device))
    values = []
    for index in range(bins):
        mask = (prediction >= boundaries[index]) & (
            prediction <= boundaries[index + 1] if index == bins - 1 else prediction < boundaries[index + 1]
        )
        if mask.any():
            values.append((prediction[mask].mean() - target[mask].mean()).square())
    return torch.stack(values).mean() if values else prediction.sum() * 0


def router_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    exploration: torch.Tensor,
    image_ids: torch.Tensor,
    beta_rank: float = 1.0,
    beta_cal: float = 0.1,
    ranking_margin: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    regression = F.huber_loss(prediction[exploration], target[exploration])
    ranking = pairwise_ranking_loss(prediction, target, image_ids, ranking_margin)
    calibration = calibration_loss(prediction[exploration], target[exploration])
    total = regression + beta_rank * ranking + beta_cal * calibration
    return total, {"huber": regression, "ranking": ranking, "calibration": calibration}
