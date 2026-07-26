from __future__ import annotations

import torch
from torch import nn


class DetailPosition(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.orientation = nn.Parameter(torch.zeros(3, embed_dim))
        nn.init.trunc_normal_(self.orientation, std=0.02)

    def forward(self, parent_positions: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        selected = parent_positions[indices]
        return selected.unsqueeze(-2) + self.orientation
