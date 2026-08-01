"""Executable Novelty-Preserving DeltaSub model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .novelty_preserving import (
    DetailSelection,
    NoveltyGate,
    NoveltyPreservingConfig,
    detail_saliency_scores,
    protected_residual_fusion,
    prototype_novelty_score,
    select_detail_parents,
)
from .subtokens.geometry import (
    extract_parent_patches,
    subdivide_parent_patches,
)
from .subtokens.haar import HaarDetails
from .subtokens.positions import ParentAwareDetailPositions
from .subtokens.projection import enforce_parent_consistency


class KnownPrototypeBank(nn.Module):
    """Persistent EMA prototypes built only from labelled training samples."""

    def __init__(
        self,
        class_count: int,
        feature_dim: int,
        *,
        momentum: float = 0.9,
    ) -> None:
        super().__init__()

        if class_count <= 0 or feature_dim <= 0:
            raise ValueError("prototype dimensions must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("prototype momentum must be in [0, 1)")

        self.class_count = int(class_count)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)

        self.register_buffer(
            "prototypes",
            torch.zeros(class_count, feature_dim),
        )
        self.register_buffer(
            "counts",
            torch.zeros(class_count, dtype=torch.long),
        )

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if features.ndim != 2:
            raise ValueError("features must have shape [B, D]")
        if features.shape[1] != self.feature_dim:
            raise ValueError("prototype feature dimension mismatch")
        if targets.shape != (features.shape[0],):
            raise ValueError("targets must have shape [B]")
        if targets.dtype != torch.long:
            raise TypeError("targets must be torch.long")
        if not torch.isfinite(features).all():
            raise ValueError("prototype features must be finite")
        if bool(((targets < 0) | (targets >= self.class_count)).any()):
            raise ValueError("prototype target outside class range")

        for target in torch.unique(targets, sorted=True):
            class_id = int(target.item())
            mean = features[targets == target].float().mean(dim=0)

            if self.counts[class_id] == 0:
                self.prototypes[class_id].copy_(mean)
            else:
                self.prototypes[class_id].mul_(self.momentum).add_(
                    mean,
                    alpha=1.0 - self.momentum,
                )

            self.counts[class_id] += int((targets == target).sum())

    def active(self) -> torch.Tensor:
        return self.prototypes[self.counts > 0]

    @property
    def active_class_count(self) -> int:
        return int((self.counts > 0).sum().item())


@dataclass(frozen=True)
class NoveltyDeltaSubOutput:
    fused_features: torch.Tensor
    global_features: torch.Tensor
    detail_features: torch.Tensor
    novelty: torch.Tensor
    gate: torch.Tensor
    selection: DetailSelection


class NoveltyPreservingDeltaSub(nn.Module):
    """Protected global pathway with novelty-conditioned Haar detail residuals."""

    def __init__(
        self,
        backbone: nn.Module,
        class_count: int,
        config: NoveltyPreservingConfig,
        *,
        seed: int = 0,
        prototype_momentum: float = 0.9,
        cold_start_novelty: float = 0.5,
    ) -> None:
        super().__init__()
        config.validate()

        if not 0 <= cold_start_novelty <= 1:
            raise ValueError("cold_start_novelty must be in [0, 1]")

        self.backbone = backbone
        self.config = config
        self.cold_start_novelty = float(cold_start_novelty)

        self.child = backbone.build_child_projector(trainable=True)
        self.positions = ParentAwareDetailPositions(
            config.feature_dim,
            test_only=config.feature_dim != 768,
        )
        self.haar = HaarDetails()
        self.gate = NoveltyGate(
            config.feature_dim,
            config.gate_hidden_dim,
            seed=seed,
        )
        self.head = nn.Linear(config.feature_dim, class_count)
        self.prototype_bank = KnownPrototypeBank(
            class_count,
            config.feature_dim,
            momentum=prototype_momentum,
        )

        self.backbone.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.backbone.model.blocks:
            tokens = block(tokens)
        return self.backbone.model.norm(tokens)[:, 0]

    def _global_path(
        self,
        images: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        parents = self.backbone.pre_transformer_parent_embeddings(images)
        batch_size = images.shape[0]

        parent_positions = (
            self.backbone.parent_patch_positions()
            .to(parents)
            .expand(batch_size, -1, -1)
        )
        prefix = (
            self.backbone.prefix_tokens_with_positions(batch_size)
            .to(parents)
        )

        tokens = torch.cat(
            (prefix, parents + parent_positions),
            dim=1,
        )

        return (
            self._encode(tokens),
            parents,
            parent_positions,
            prefix,
        )

    def _detail_path(
        self,
        images: torch.Tensor,
        parents: torch.Tensor,
        parent_positions: torch.Tensor,
        prefix: torch.Tensor,
        selection: DetailSelection,
    ) -> torch.Tensor:
        children = self.child(
            subdivide_parent_patches(
                extract_parent_patches(images)
            )
        )
        consistent = enforce_parent_consistency(
            children,
            parents,
        ).consistent
        details = self.haar(consistent)
        detail_positions = self.positions(parent_positions)

        result = parents.new_empty(
            images.shape[0],
            self.config.feature_dim,
        )

        for k_value in torch.unique(
            selection.adaptive_k,
            sorted=True,
        ).tolist():
            rows = torch.nonzero(
                selection.adaptive_k == int(k_value),
                as_tuple=False,
            ).squeeze(1)

            selected = selection.selected_mask[rows]
            group_size = rows.numel()

            if int(k_value):
                selected_details = details[rows][selected].reshape(
                    group_size,
                    int(k_value) * 3,
                    self.config.feature_dim,
                )
                selected_positions = detail_positions[rows][
                    selected
                ].reshape(
                    group_size,
                    int(k_value) * 3,
                    self.config.feature_dim,
                )
                detail_tokens = (
                    selected_details + selected_positions
                )
            else:
                detail_tokens = parents.new_empty(
                    group_size,
                    0,
                    self.config.feature_dim,
                )

            tokens = torch.cat(
                (
                    prefix[rows],
                    parents[rows] + parent_positions[rows],
                    detail_tokens,
                ),
                dim=1,
            )

            result[rows] = self._encode(tokens)

        return result

    def update_known_prototypes(
        self,
        global_features: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self.prototype_bank.update(
            global_features.detach(),
            targets,
        )

    def features(
        self,
        images: torch.Tensor,
        *,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | NoveltyDeltaSubOutput:
        (
            global_features,
            parents,
            parent_positions,
            prefix,
        ) = self._global_path(images)

        active_prototypes = self.prototype_bank.active()

        if active_prototypes.shape[0]:
            novelty = prototype_novelty_score(
                global_features,
                active_prototypes,
                threshold=self.config.known_similarity_threshold,
                temperature=self.config.novelty_temperature,
            ).to(global_features)
        else:
            novelty = global_features.new_full(
                (images.shape[0],),
                self.cold_start_novelty,
            )

        scores = detail_saliency_scores(
            parents,
            global_features,
        )
        selection = select_detail_parents(
            scores,
            novelty,
            maximum_k=self.config.maximum_detail_parents,
        )

        detail_features = self._detail_path(
            images,
            parents,
            parent_positions,
            prefix,
            selection,
        )

        gate = self.gate(
            global_features,
            detail_features,
            novelty,
        )
        fused = protected_residual_fusion(
            global_features,
            detail_features,
            gate,
        )

        if not return_auxiliary:
            return fused

        return NoveltyDeltaSubOutput(
            fused_features=fused,
            global_features=global_features,
            detail_features=detail_features,
            novelty=novelty,
            gate=gate,
            selection=selection,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))
