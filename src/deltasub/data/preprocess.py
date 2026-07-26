from __future__ import annotations

import torch
from torch.nn import functional as F


DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def normalize_dinov2(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(DINO_MEAN, device=images.device, dtype=images.dtype).reshape(1, 3, 1, 1)
    std = torch.tensor(DINO_STD, device=images.device, dtype=images.dtype).reshape(1, 3, 1, 1)
    return (images - mean) / std


def deterministic_resize(images: torch.Tensor, size: int = 224) -> torch.Tensor:
    return F.interpolate(images, (size, size), mode="bicubic", align_corners=False, antialias=True)
