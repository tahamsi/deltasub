"""Parent-cell positional policy for Haar modes."""
from __future__ import annotations

import torch
from torch import nn


class ParentAwareDetailPositions(nn.Module):
    modes = ("horizontal", "vertical", "diagonal")

    def __init__(self, embed_dim: int = 768, *, test_only: bool = False):
        super().__init__()
        if embed_dim != 768 and not test_only:
            raise ValueError("production detail positions require embedding dimension 768")
        self.embed_dim = embed_dim
        self.mode_embeddings = nn.Parameter(torch.zeros(3, embed_dim))

    def forward(self, parent_positions: torch.Tensor) -> torch.Tensor:
        if parent_positions.ndim not in (2, 3) or parent_positions.shape[-2:] != (256, self.embed_dim):
            raise ValueError(f"parent positions must end in [256, {self.embed_dim}]")
        return parent_positions.unsqueeze(-2) + self.mode_embeddings.to(parent_positions)

    def split_official_positions(self, prefix_tokens: torch.Tensor, parent_positions: torch.Tensor):
        if prefix_tokens.ndim != 3 or prefix_tokens.shape[-1] != self.embed_dim:
            raise ValueError("prefix positions must be [B or 1, P, D]")
        return prefix_tokens, self(parent_positions)
