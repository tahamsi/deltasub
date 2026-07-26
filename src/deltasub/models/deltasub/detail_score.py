from __future__ import annotations

import torch
from torch.nn import functional as F


def cheap_detail_features(images: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    gray = images.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=images.device, dtype=images.dtype
    ).reshape(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    energy = (gx.square() + gy.square()).sqrt()
    sobel = F.avg_pool2d(energy, patch_size, patch_size).flatten(2).transpose(1, 2)
    mean = F.avg_pool2d(images, patch_size, patch_size)
    mean_sq = F.avg_pool2d(images.square(), patch_size, patch_size)
    variance = (mean_sq - mean.square()).mean(1).flatten(1).unsqueeze(-1).clamp_min(0)
    features = torch.cat([sobel, variance], dim=-1)
    lo = features.amin(dim=1, keepdim=True)
    hi = features.amax(dim=1, keepdim=True)
    return (features - lo) / (hi - lo + 1e-6)
