from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def extract_parent_patches(images: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-2] % patch_size or images.shape[-1] % patch_size:
        raise ValueError("images must be BCHW with spatial dimensions divisible by patch_size")
    b, c, h, w = images.shape
    rows, columns = h // patch_size, w // patch_size
    patches = images.reshape(b, c, rows, patch_size, columns, patch_size)
    return patches.permute(0, 2, 4, 1, 3, 5).reshape(
        b, rows * columns, c, patch_size, patch_size
    )


def subdivide_parent_patches(parents: torch.Tensor) -> torch.Tensor:
    if parents.ndim != 5 or parents.shape[-2:] != (14, 14):
        raise ValueError("parents must have shape [B,K,C,14,14]")
    b, k, c, _, _ = parents.shape
    children = parents.reshape(b, k, c, 2, 7, 2, 7).permute(0, 1, 3, 5, 2, 4, 6)
    return children.reshape(b, k, 4, c, 7, 7)


class ChildPatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, channels: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=7, stride=7)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, children: torch.Tensor) -> torch.Tensor:
        if children.ndim != 6 or children.shape[-2:] != (7, 7):
            raise ValueError("children must have shape [B,K,4,C,7,7]")
        b, k, four, c, h, w = children.shape
        output = self.proj(children.reshape(b * k * four, c, h, w))
        return output.flatten(1).reshape(b, k, four, -1)

    @torch.no_grad()
    def initialize_from_parent(self, parent: nn.Conv2d, mode: str) -> None:
        if parent.weight.shape[-2:] != (14, 14):
            raise ValueError("parent projector must use a 14x14 kernel")
        if mode == "resized_parent_kernel":
            weight = F.interpolate(parent.weight, size=(7, 7), mode="bilinear", align_corners=False)
        elif mode == "pooled_parent_kernel":
            weight = F.avg_pool2d(parent.weight, kernel_size=2, stride=2)
        elif mode == "random_truncated_normal":
            nn.init.trunc_normal_(self.proj.weight, std=0.02)
            return
        else:
            raise ValueError(f"unknown initialization mode: {mode}")
        weight = weight * math.sqrt(parent.weight[0].numel() / weight[0].numel())
        self.proj.weight.copy_(weight)
        if parent.bias is not None:
            self.proj.bias.copy_(parent.bias)
