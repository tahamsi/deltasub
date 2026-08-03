"""Matched DINOv2 + SelEx baseline with late-block fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MatchedSelExOutput:
    features: torch.Tensor
    logits: torch.Tensor


class MatchedSelEx(nn.Module):
    """DINOv2 baseline under the same trainability policy as DeltaSub."""

    def __init__(
        self,
        backbone: nn.Module,
        class_count: int,
        *,
        trainable_blocks: int = 2,
    ) -> None:
        super().__init__()

        if class_count <= 0:
            raise ValueError("class_count must be positive")
        if trainable_blocks <= 0:
            raise ValueError("trainable_blocks must be positive")

        self.backbone = backbone
        self.trainable_blocks = int(trainable_blocks)
        self.backbone.set_trainable_blocks(self.trainable_blocks)
        self.head = nn.Linear(768, class_count)

        self._apply_training_policy()

    @property
    def frozen_blocks(self) -> tuple[nn.Module, ...]:
        return tuple(
            self.backbone.model.blocks[
                : -self.trainable_blocks
            ]
        )

    @property
    def tuned_blocks(self) -> tuple[nn.Module, ...]:
        return tuple(
            self.backbone.model.blocks[
                -self.trainable_blocks :
            ]
        )

    def _apply_training_policy(self) -> None:
        self.backbone.model.eval()

        for block in self.frozen_blocks:
            block.eval()

        for block in self.tuned_blocks:
            block.train(self.training)

        self.backbone.model.norm.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_training_policy()
        return self

    def features(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (
            3,
            224,
            224,
        ):
            raise ValueError(
                "images must have shape [B, 3, 224, 224]"
            )

        output = self.backbone.model.forward_features(images)
        features = output["x_norm_clstoken"]

        if features.shape != (images.shape[0], 768):
            raise ValueError(
                "unexpected DINOv2 CLS feature shape: "
                f"{tuple(features.shape)}"
            )

        return features

    def branches(self, images: torch.Tensor) -> MatchedSelExOutput:
        features = self.features(images)
        return MatchedSelExOutput(
            features=features,
            logits=self.head(features),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.branches(images).logits

    def parameter_report(self) -> dict[str, int]:
        total = sum(
            parameter.numel()
            for parameter in self.parameters()
        )
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
