"""Fixed orthonormal Haar transform over TL, TR, BL, BR child embeddings."""
from __future__ import annotations

import torch
from torch import nn


class HaarDetails(nn.Module):
    modes = ("horizontal", "vertical", "diagonal")

    def __init__(self):
        super().__init__()
        self.register_buffer("q", 0.5 * torch.tensor(
            [[1., -1., 1., -1.], [1., 1., -1., -1.], [1., -1., -1., 1.]]
        ))
        self.register_buffer("h", 0.5 * torch.tensor(
            [[1., 1., 1., 1.], [1., -1., 1., -1.],
             [1., 1., -1., -1.], [1., -1., -1., 1.]]
        ))

    def forward(self, children: torch.Tensor) -> torch.Tensor:
        if children.ndim != 4 or children.shape[2] != 4:
            raise ValueError("children must have shape [B, N, 4, D]")
        return torch.einsum("mc,bncd->bnmd", self.q.to(children), children)

    def reconstruct(self, parent: torch.Tensor, details: torch.Tensor) -> torch.Tensor:
        if details.ndim != 4 or details.shape[2] != 3 or parent.shape != details.shape[:2] + details.shape[3:]:
            raise ValueError("parent/details shapes must be [B, N, D] and [B, N, 3, D]")
        coefficients = torch.cat((2 * parent.unsqueeze(2), details), dim=2)
        return torch.einsum("mc,bnmd->bncd", self.h.to(details), coefficients)

    def orthogonality_error(self) -> torch.Tensor:
        return (self.q @ self.q.T - torch.eye(3, device=self.q.device, dtype=self.q.dtype)).abs().max()
