"""Scalar SelEx loss matching the pinned upstream batch reduction."""
from __future__ import annotations
import torch
from .per_anchor_selex import per_anchor_selex, selex_per_anchor


def scalar_selex(*args, **kwargs) -> torch.Tensor:
    return per_anchor_selex(*args, **kwargs).total.mean()


def selex_loss(*args, **kwargs) -> torch.Tensor:
    result = selex_per_anchor(*args, **kwargs)
    if not result.valid.any():
        return result.total.sum() * 0
    return result.total[result.valid].mean()
