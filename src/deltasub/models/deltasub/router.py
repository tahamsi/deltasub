from __future__ import annotations

import torch
from torch import nn


def normalized_coordinates(grid_size: int, *, device=None, dtype=None) -> torch.Tensor:
    axis = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([rows, cols], dim=-1).reshape(-1, 2)


class GainRouter(nn.Module):
    def __init__(self, embed_dim: int, cheap_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.depthwise = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.pointwise = nn.Conv2d(embed_dim, hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + cheap_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, parents: torch.Tensor, cheap: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        b, n, d = parents.shape
        side = int(n**0.5)
        if side * side != n:
            raise ValueError("parent token count must form a square grid")
        x = self.norm(parents).transpose(1, 2).reshape(b, d, side, side)
        x = torch.nn.functional.gelu(self.pointwise(self.depthwise(x)))
        x = x.flatten(2).transpose(1, 2)
        coords = coordinates.unsqueeze(0).expand(b, -1, -1)
        return self.head(torch.cat([x, cheap, coords], dim=-1)).squeeze(-1)
