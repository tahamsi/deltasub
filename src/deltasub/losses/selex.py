from __future__ import annotations

import torch

from .per_anchor_selex import per_anchor_selex


def scalar_selex(*args, **kwargs) -> torch.Tensor:
    """Scalar reduction defined from the exact per-anchor decomposition."""
    return per_anchor_selex(*args, **kwargs).total.mean()
