from __future__ import annotations

import torch
from torch import nn

from .child_patch_embed import ChildPatchEmbed, extract_parent_patches, subdivide_parent_patches
from .haar_detail import haar_details


class TinyDeltaSub(nn.Module):
    """Small deterministic model used for tests and pipeline validation.

    This is not the publication DINOv2 model and its metrics must never enter paper tables.
    """

    def __init__(self, embed_dim: int = 32, classes: int = 4) -> None:
        super().__init__()
        self.parent_proj = nn.Conv2d(3, embed_dim, 14, 14)
        self.child_proj = ChildPatchEmbed(embed_dim)
        layer = nn.TransformerEncoderLayer(
            embed_dim, 4, embed_dim * 2, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.classifier = nn.Linear(embed_dim, classes)

    def parent_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return self.parent_proj(images).flatten(2).transpose(1, 2)

    def forward(self, images: torch.Tensor, selections: list[torch.Tensor] | None = None) -> torch.Tensor:
        parents = self.parent_tokens(images)
        sequences = []
        all_patches = extract_parent_patches(images)
        if selections is None:
            selections = [torch.empty(0, dtype=torch.long, device=images.device) for _ in images]
        for row, selected in enumerate(selections):
            sequence = parents[row]
            if len(selected):
                chosen = all_patches[row, selected].unsqueeze(0)
                children = subdivide_parent_patches(chosen)
                detail = haar_details(self.child_proj(children)).reshape(-1, parents.shape[-1])
                sequence = torch.cat([sequence, detail], dim=0)
            sequences.append(sequence)
        max_length = max(len(item) for item in sequences)
        padded = torch.stack(
            [torch.nn.functional.pad(item, (0, 0, 0, max_length - len(item))) for item in sequences]
        )
        lengths = torch.tensor([len(item) for item in sequences], device=images.device)
        padding_mask = torch.arange(max_length, device=images.device).unsqueeze(0) >= lengths.unsqueeze(1)
        encoded = self.encoder(padded, src_key_padding_mask=padding_mask)
        pooled = torch.stack([encoded[i, : len(sequence)].mean(0) for i, sequence in enumerate(sequences)])
        return self.classifier(pooled)
