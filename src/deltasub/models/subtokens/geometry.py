"""Exact pixel geometry for 224px DINOv2 ViT-B/14 inputs."""
from __future__ import annotations

import torch


def extract_parent_patches(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
        raise ValueError(f"images must have shape [B, 3, 224, 224], got {tuple(images.shape)}")
    # unfold order is row-major: parent index = row * 16 + column.
    return images.unfold(2, 14, 14).unfold(3, 14, 14).permute(0, 2, 3, 1, 4, 5).reshape(
        images.shape[0], 256, 3, 14, 14
    )


def reconstruct_images(parents: torch.Tensor) -> torch.Tensor:
    if parents.ndim != 5 or tuple(parents.shape[1:]) != (256, 3, 14, 14):
        raise ValueError(f"parents must have shape [B, 256, 3, 14, 14], got {tuple(parents.shape)}")
    return parents.reshape(-1, 16, 16, 3, 14, 14).permute(0, 3, 1, 4, 2, 5).reshape(
        parents.shape[0], 3, 224, 224
    )


def subdivide_parent_patches(parents: torch.Tensor) -> torch.Tensor:
    if parents.ndim != 5 or tuple(parents.shape[1:]) != (256, 3, 14, 14):
        raise ValueError(f"parents must have shape [B, 256, 3, 14, 14], got {tuple(parents.shape)}")
    # [row quadrant, column quadrant] flattened gives TL, TR, BL, BR.
    return parents.unfold(-2, 7, 7).unfold(-2, 7, 7).permute(0, 1, 3, 4, 2, 5, 6).reshape(
        parents.shape[0], 256, 4, 3, 7, 7
    )


def reconstruct_parent_patches(children: torch.Tensor) -> torch.Tensor:
    if children.ndim != 6 or tuple(children.shape[1:]) != (256, 4, 3, 7, 7):
        raise ValueError(f"children must have shape [B, 256, 4, 3, 7, 7], got {tuple(children.shape)}")
    return children.reshape(-1, 256, 2, 2, 3, 7, 7).permute(0, 1, 4, 2, 5, 3, 6).reshape(
        children.shape[0], 256, 3, 14, 14
    )
