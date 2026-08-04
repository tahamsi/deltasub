"""Dense semantic-residual correction for DeltaSub.

This implementation performs no patch selection and uses no oracle.
A lightweight convolutional branch produces one descriptor for every
existing DINOv2 patch cell. Information predictable from the semantic
token is removed, the remaining residual is projected away from the
semantic subspace, and a bounded low-rank correction is injected into
the corresponding parent token.

The transformer sequence length remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


PATCH_COUNT = 256
GRID_SIZE = 16


@dataclass(frozen=True)
class DeltaSubResidualConfig:
    feature_dim: int = 768
    insertion_block: int = 10
    trainable_blocks: int = 2
    local_channels: int = 128
    predictor_hidden_dim: int = 512
    transport_rank: int = 128
    maximum_correction_ratio: float = 0.10
    initial_scale: float = 0.05
    use_prediction_residual: bool = True
    use_semantic_projection: bool = True

    def validate(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError(
                "feature_dim must be positive"
            )
        if self.insertion_block < 0:
            raise ValueError(
                "insertion_block must be nonnegative"
            )
        if self.trainable_blocks <= 0:
            raise ValueError(
                "trainable_blocks must be positive"
            )
        if (
            self.local_channels < 16
            or self.local_channels % 16
        ):
            raise ValueError(
                "local_channels must be a positive "
                "multiple of 16"
            )
        if self.predictor_hidden_dim <= 0:
            raise ValueError(
                "predictor_hidden_dim must be positive"
            )
        if self.transport_rank <= 0:
            raise ValueError(
                "transport_rank must be positive"
            )
        if not 0 < self.maximum_correction_ratio <= 1:
            raise ValueError(
                "maximum_correction_ratio must be in "
                "(0, 1]"
            )
        if not 0 < self.initial_scale < 1:
            raise ValueError(
                "initial_scale must be in (0, 1)"
            )
        if (
            self.use_semantic_projection
            and not self.use_prediction_residual
        ):
            raise ValueError(
                "semantic projection requires the "
                "prediction residual"
            )


@dataclass(frozen=True)
class DeltaSubResidualOutput:
    fused_features: torch.Tensor
    global_features: torch.Tensor
    refined_features: torch.Tensor
    fused_logits: torch.Tensor
    global_logits: torch.Tensor
    refined_logits: torch.Tensor
    local_descriptors: torch.Tensor
    predicted_local: torch.Tensor
    residual_tokens: torch.Tensor
    corrections: torch.Tensor
    parent_tokens: torch.Tensor
    cls_tokens: torch.Tensor
    correction_ratio: torch.Tensor
    residual_norm: torch.Tensor
    injection_scale: torch.Tensor


class LocalGridEncoder(nn.Module):
    """Produce one local descriptor for each 14x14 DINO patch cell."""

    def __init__(
        self,
        *,
        feature_dim: int,
        local_channels: int,
    ) -> None:
        super().__init__()

        first = local_channels // 2

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                first,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, first),
            nn.GELU(),
            nn.Conv2d(
                first,
                local_channels,
                kernel_size=7,
                stride=7,
                padding=0,
                bias=False,
            ),
            nn.GroupNorm(8, local_channels),
            nn.GELU(),
            nn.Conv2d(
                local_channels,
                local_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=local_channels,
                bias=False,
            ),
            nn.GELU(),
        )

        self.projection = nn.Sequential(
            nn.LayerNorm(local_channels),
            nn.Linear(
                local_channels,
                feature_dim,
                bias=False,
            ),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if (
            images.ndim != 4
            or tuple(images.shape[1:])
            != (3, 224, 224)
        ):
            raise ValueError(
                "images must have shape [B, 3, 224, 224]"
            )

        values = self.stem(images)

        if values.shape[2:] != (
            GRID_SIZE,
            GRID_SIZE,
        ):
            raise RuntimeError(
                "local encoder grid contract failed: "
                f"{tuple(values.shape)}"
            )

        values = values.flatten(2).transpose(1, 2)
        values = self.projection(values)

        if values.shape[1] != PATCH_COUNT:
            raise RuntimeError(
                "local descriptor patch count failed"
            )

        return values


def remove_semantic_subspace(
    residual: torch.Tensor,
    directions: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Remove a Gram-Schmidt semantic basis from each residual."""

    if residual.ndim != 3:
        raise ValueError(
            "residual must have shape [B, N, D]"
        )

    batch, patches, width = residual.shape
    work = residual.float()
    basis: list[torch.Tensor] = []

    for direction in directions:
        if direction.ndim == 2:
            if direction.shape != (batch, width):
                raise ValueError(
                    "global semantic direction shape mismatch"
                )
            value = direction[:, None, :].expand(
                -1,
                patches,
                -1,
            )
        elif direction.ndim == 3:
            if direction.shape != residual.shape:
                raise ValueError(
                    "local semantic direction shape mismatch"
                )
            value = direction
        else:
            raise ValueError(
                "semantic direction must be [B, D] "
                "or [B, N, D]"
            )

        value = value.float()

        for existing in basis:
            value = value - (
                value * existing
            ).sum(
                dim=-1,
                keepdim=True,
            ) * existing

        normalized = F.normalize(
            value,
            dim=-1,
            eps=1e-6,
        )
        basis.append(normalized)

        work = work - (
            work * normalized
        ).sum(
            dim=-1,
            keepdim=True,
        ) * normalized

    return work.to(residual)


def bound_token_correction(
    correction: torch.Tensor,
    parent_tokens: torch.Tensor,
    maximum_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound every correction relative to its parent-token norm."""

    if correction.shape != parent_tokens.shape:
        raise ValueError(
            "correction and parent tokens must match"
        )
    if not 0 < maximum_ratio <= 1:
        raise ValueError(
            "maximum_ratio must be in (0, 1]"
        )

    parent_norm = parent_tokens.float().norm(
        dim=-1
    ).clamp_min(1e-6)

    correction_norm = correction.float().norm(
        dim=-1
    ).clamp_min(1e-12)

    permitted = float(maximum_ratio) * parent_norm

    multiplier = (
        permitted / correction_norm
    ).clamp(max=1.0)

    bounded = correction * multiplier.unsqueeze(-1).to(
        correction
    )

    ratio = (
        bounded.float().norm(dim=-1)
        / parent_norm
    )

    return bounded, ratio


class DeltaSubResidual(nn.Module):
    """Dense missing-information correction for late DINOv2 tokens."""

    def __init__(
        self,
        backbone: nn.Module,
        class_count: int,
        config: DeltaSubResidualConfig,
    ) -> None:
        super().__init__()
        config.validate()

        if class_count <= 0:
            raise ValueError(
                "class_count must be positive"
            )

        blocks = list(backbone.model.blocks)
        depth = len(blocks)

        if (
            config.insertion_block
            + config.trainable_blocks
            != depth
        ):
            raise ValueError(
                "insertion_block + trainable_blocks "
                "must equal backbone depth"
            )

        self.backbone = backbone
        self.config = config
        self.class_count = int(class_count)

        self.backbone.set_trainable_blocks(
            config.trainable_blocks
        )

        # Preserve matched baseline head initialization order.
        self.head = nn.Linear(
            config.feature_dim,
            class_count,
        )

        self.local_encoder = LocalGridEncoder(
            feature_dim=config.feature_dim,
            local_channels=config.local_channels,
        )

        if config.use_prediction_residual:
            self.local_predictor: nn.Module | None = (
                nn.Sequential(
                    nn.LayerNorm(config.feature_dim),
                    nn.Linear(
                        config.feature_dim,
                        config.predictor_hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        config.predictor_hidden_dim,
                        config.feature_dim,
                        bias=False,
                    ),
                )
            )
        else:
            self.local_predictor = None

        if config.use_semantic_projection:
            self.semantic_projection: nn.Module | None = (
                nn.Sequential(
                    nn.LayerNorm(config.feature_dim),
                    nn.Linear(
                        config.feature_dim,
                        config.feature_dim,
                        bias=False,
                    ),
                )
            )
        else:
            self.semantic_projection = None

        self.residual_norm = nn.LayerNorm(
            config.feature_dim
        )
        self.transport_down = nn.Linear(
            config.feature_dim,
            config.transport_rank,
            bias=False,
        )
        self.transport_up = nn.Linear(
            config.transport_rank,
            config.feature_dim,
            bias=False,
        )

        nn.init.normal_(
            self.transport_up.weight,
            std=1e-3,
        )

        initial_logit = math.log(
            config.initial_scale
            / (1.0 - config.initial_scale)
        )
        self.injection_scale_logit = nn.Parameter(
            torch.tensor(initial_logit)
        )

        self._apply_training_policy()

    @property
    def trunk_blocks(
        self,
    ) -> tuple[nn.Module, ...]:
        return tuple(
            self.backbone.model.blocks[
                : self.config.insertion_block
            ]
        )

    @property
    def tail_blocks(
        self,
    ) -> tuple[nn.Module, ...]:
        return tuple(
            self.backbone.model.blocks[
                self.config.insertion_block :
            ]
        )

    def _apply_training_policy(self) -> None:
        self.backbone.model.eval()

        for block in self.trunk_blocks:
            block.eval()

        for block in self.tail_blocks:
            block.train(self.training)

        self.backbone.model.norm.train(
            self.training
        )

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)
        self._apply_training_policy()
        return self

    def _run_trunk(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if (
            images.ndim != 4
            or tuple(images.shape[1:])
            != (3, 224, 224)
        ):
            raise ValueError(
                "images must have shape [B, 3, 224, 224]"
            )

        batch = images.shape[0]

        with torch.no_grad():
            parents = (
                self.backbone
                .pre_transformer_parent_embeddings(
                    images
                )
            )
            positions = (
                self.backbone
                .parent_patch_positions()
                .to(parents)
                .expand(batch, -1, -1)
            )
            prefix = (
                self.backbone
                .prefix_tokens_with_positions(batch)
                .to(parents)
            )

            tokens = torch.cat(
                (
                    prefix,
                    parents + positions,
                ),
                dim=1,
            )

            for block in self.trunk_blocks:
                tokens = block(tokens)

        return tokens

    def _tail_encode(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.tail_blocks:
            tokens = block(tokens)

        return self.backbone.model.norm(
            tokens
        )[:, 0]

    def _residual(
        self,
        *,
        images: torch.Tensor,
        parent_tokens: torch.Tensor,
        cls_token: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        local = self.local_encoder(images)

        if self.local_predictor is None:
            predicted = torch.zeros_like(local)
            residual = local
        else:
            predicted = self.local_predictor(
                parent_tokens
            )
            residual = local - predicted

        if self.semantic_projection is not None:
            cls_direction = self.semantic_projection(
                cls_token
            )
            mean_direction = (
                self.semantic_projection(
                    parent_tokens.mean(dim=1)
                )
            )

            residual = remove_semantic_subspace(
                residual,
                (
                    cls_direction,
                    mean_direction,
                    predicted,
                ),
            )

        return local, predicted, residual

    def _correction(
        self,
        residual: torch.Tensor,
        parent_tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        transported = self.transport_up(
            F.gelu(
                self.transport_down(
                    self.residual_norm(residual)
                )
            )
        )

        scale = torch.sigmoid(
            self.injection_scale_logit
        ).to(transported)

        transported = scale * transported

        correction, ratio = bound_token_correction(
            transported,
            parent_tokens,
            self.config.maximum_correction_ratio,
        )

        return correction, ratio, scale

    def branches(
        self,
        images: torch.Tensor,
    ) -> DeltaSubResidualOutput:
        trunk_tokens = self._run_trunk(images)

        prefix_count = int(
            self.backbone.prefix_token_count
        )
        parent_start = prefix_count
        parent_end = parent_start + PATCH_COUNT

        parent_tokens = trunk_tokens[
            :,
            parent_start:parent_end,
        ]
        cls_token = trunk_tokens[:, 0]

        global_features = self._tail_encode(
            trunk_tokens
        )

        local, predicted, residual = self._residual(
            images=images,
            parent_tokens=parent_tokens,
            cls_token=cls_token,
        )

        correction, ratio, scale = (
            self._correction(
                residual,
                parent_tokens,
            )
        )

        refined_tokens = torch.cat(
            (
                trunk_tokens[:, :parent_start],
                parent_tokens
                + correction.to(parent_tokens),
                trunk_tokens[:, parent_end:],
            ),
            dim=1,
        )

        devices = (
            [torch.cuda.current_device()]
            if images.is_cuda
            else []
        )

        with torch.random.fork_rng(
            devices=devices
        ):
            refined_features = self._tail_encode(
                refined_tokens
            )

        global_logits = self.head(
            global_features
        )
        refined_logits = self.head(
            refined_features
        )

        return DeltaSubResidualOutput(
            fused_features=refined_features,
            global_features=global_features,
            refined_features=refined_features,
            fused_logits=refined_logits,
            global_logits=global_logits,
            refined_logits=refined_logits,
            local_descriptors=local,
            predicted_local=predicted,
            residual_tokens=residual,
            corrections=correction,
            parent_tokens=parent_tokens,
            cls_tokens=cls_token,
            correction_ratio=ratio,
            residual_norm=(
                residual.float().norm(dim=-1)
            ),
            injection_scale=scale,
        )

    def features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.branches(
            images
        ).fused_features

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return self.branches(
            images
        ).fused_logits

    def parameter_report(
        self,
    ) -> dict[str, int]:
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
