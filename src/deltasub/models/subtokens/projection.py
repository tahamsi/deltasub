"""Quadrant initialization and exact parent consistency."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ConsistentChildren:
    raw: torch.Tensor
    consistent: torch.Tensor
    raw_mean_residual: torch.Tensor
    hard_mean_residual: torch.Tensor
    max_absolute_error: torch.Tensor
    raw_consistency_loss: torch.Tensor


def enforce_parent_consistency(raw: torch.Tensor, parent: torch.Tensor) -> ConsistentChildren:
    if raw.ndim != 4 or raw.shape[2] != 4 or parent.shape != raw.shape[:2] + raw.shape[3:]:
        raise ValueError("raw must be [B, N, 4, D] and parent must be [B, N, D]")
    mean = raw.mean(dim=2)
    consistent = raw - mean.unsqueeze(2) + parent.unsqueeze(2)
    hard = consistent.mean(dim=2) - parent
    return ConsistentChildren(
        raw, consistent, mean - parent, hard, hard.abs().max(),
        F.mse_loss(mean, parent),
    )


class ChildProjector(nn.Module):
    """Four trainable 7x7 linear projections copied from one official 14x14 projection."""

    def __init__(self, parent_projection: nn.Conv2d, *, trainable: bool = True, provenance: str):
        super().__init__()
        if not provenance:
            raise ValueError("initialization provenance is required")
        if not isinstance(parent_projection, nn.Conv2d):
            raise TypeError("parent projection must be nn.Conv2d")
        if (parent_projection.in_channels, parent_projection.kernel_size,
                parent_projection.stride, parent_projection.groups,
                parent_projection.dilation, parent_projection.padding) != (
                3, (14, 14), (14, 14), 1, (1, 1), (0, 0)):
            raise ValueError("parent projection must be Conv2d(3, D, kernel=14, stride=14)")
        self.embed_dim = parent_projection.out_channels
        self.provenance = provenance
        device, dtype = parent_projection.weight.device, parent_projection.weight.dtype
        self.projections = nn.ModuleList([
            nn.Linear(3 * 7 * 7, self.embed_dim, bias=parent_projection.bias is not None,
                      device=device, dtype=dtype) for _ in range(4)
        ])
        slices = ((slice(0, 7), slice(0, 7)), (slice(0, 7), slice(7, 14)),
                  (slice(7, 14), slice(0, 7)), (slice(7, 14), slice(7, 14)))
        with torch.no_grad():
            for layer, (rows, cols) in zip(self.projections, slices):
                layer.weight.copy_(4 * parent_projection.weight[:, :, rows, cols].reshape(self.embed_dim, -1))
                if layer.bias is not None:
                    layer.bias.copy_(parent_projection.bias)
        self.register_buffer("_initial", self._flat_parameters().detach().clone(), persistent=True)
        self.requires_grad_(trainable)

    @classmethod
    def from_patch_embed(cls, patch_embed: nn.Module, **kwargs):
        norm = getattr(patch_embed, "norm", None)
        if norm is not None and not isinstance(norm, nn.Identity):
            raise ValueError("non-identity patch-embedding normalization breaks exact decomposition")
        proj = getattr(patch_embed, "proj", None)
        return cls(proj, **kwargs)

    def _flat_parameters(self):
        return torch.cat([p.reshape(-1) for layer in self.projections for p in layer.parameters()])

    @property
    def equals_initialization(self) -> bool:
        return torch.equal(self._flat_parameters().detach(), self._initial)

    @property
    def initialization_sha256(self) -> str:
        return hashlib.sha256(self._initial.detach().cpu().numpy().tobytes()).hexdigest()

    def forward(self, children: torch.Tensor) -> torch.Tensor:
        if children.ndim != 6 or children.shape[2:] != (4, 3, 7, 7):
            raise ValueError("children must have shape [B, N, 4, 3, 7, 7]")
        flat = children.reshape(*children.shape[:3], -1)
        return torch.stack([layer(flat[:, :, q]) for q, layer in enumerate(self.projections)], dim=2)
