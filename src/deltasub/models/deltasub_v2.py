"""DeltaSub v2: late, selective, utility-calibrated Haar refinement."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..adaptive.selection import deterministic_select
from .subtokens.geometry import (
    extract_parent_patches,
    subdivide_parent_patches,
)
from .subtokens.haar import HaarDetails
from .subtokens.projection import enforce_parent_consistency


@dataclass(frozen=True)
class DeltaSubV2Config:
    feature_dim: int = 768
    insertion_block: int = 10
    trainable_blocks: int = 2
    minimum_k: int = 1
    maximum_k: int = 16
    retained_energy_fraction: float = 0.80
    detail_adapter_hidden_dim: int = 768
    utility_hidden_dim: int = 128
    initial_detail_scale: float = 0.10
    initial_fusion_weight: float = 0.25

    def validate(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.insertion_block < 0:
            raise ValueError("insertion_block must be nonnegative")
        if self.trainable_blocks <= 0:
            raise ValueError("trainable_blocks must be positive")
        if not 0 <= self.minimum_k <= self.maximum_k <= 256:
            raise ValueError("detail budget must satisfy 0 <= min <= max <= 256")
        if not 0 < self.retained_energy_fraction <= 1:
            raise ValueError("retained_energy_fraction must be in (0, 1]")
        if self.detail_adapter_hidden_dim <= 0:
            raise ValueError("detail_adapter_hidden_dim must be positive")
        if self.utility_hidden_dim <= 0:
            raise ValueError("utility_hidden_dim must be positive")
        if not 0 < self.initial_detail_scale < 1:
            raise ValueError("initial_detail_scale must be in (0, 1)")
        if not 0 < self.initial_fusion_weight < 1:
            raise ValueError("initial_fusion_weight must be in (0, 1)")


@dataclass(frozen=True)
class AdaptiveDetailSelection:
    scores: torch.Tensor
    selected_mask: torch.Tensor
    selected_indices: tuple[tuple[int, ...], ...]
    adaptive_k: torch.Tensor
    retained_fraction: torch.Tensor


@dataclass(frozen=True)
class DeltaSubV2Output:
    fused_features: torch.Tensor
    global_features: torch.Tensor
    detail_features: torch.Tensor
    fused_logits: torch.Tensor
    global_logits: torch.Tensor
    detail_logits: torch.Tensor
    fusion_weight: torch.Tensor
    selection: AdaptiveDetailSelection


def adaptive_energy_select(
    scores: torch.Tensor,
    *,
    minimum_k: int,
    maximum_k: int,
    retained_fraction: float,
) -> AdaptiveDetailSelection:
    """Select the smallest K retaining the requested score mass."""

    if scores.ndim != 2 or scores.shape[1] != 256:
        raise ValueError("scores must have shape [B, 256]")
    if not scores.is_floating_point():
        raise TypeError("scores must be floating point")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if bool((scores < 0).any()):
        raise ValueError("scores must be nonnegative")
    if not 0 <= minimum_k <= maximum_k <= 256:
        raise ValueError("invalid detail budget")
    if not 0 < retained_fraction <= 1:
        raise ValueError("retained_fraction must be in (0, 1]")

    work = scores.float()

    if maximum_k == 0:
        adaptive_k = torch.zeros(
            scores.shape[0],
            dtype=torch.long,
            device=scores.device,
        )
        selection = deterministic_select(work, adaptive_k)

        return AdaptiveDetailSelection(
            scores=scores,
            selected_mask=selection.selected_mask,
            selected_indices=selection.selected_indices,
            adaptive_k=adaptive_k,
            retained_fraction=torch.zeros_like(
                adaptive_k,
                dtype=work.dtype,
            ),
        )

    ranked = torch.argsort(
        work,
        dim=1,
        descending=True,
        stable=True,
    )
    ranked_scores = work.gather(1, ranked)

    # Adapt K within the permitted token budget. Measuring against
    # all 256 patches forces every diffuse sample to maximum_k.
    selectable_scores = ranked_scores[:, :maximum_k]
    cumulative = selectable_scores.cumsum(dim=1)
    totals = selectable_scores.sum(dim=1)

    threshold = totals * float(retained_fraction)
    reached = cumulative >= threshold.unsqueeze(1)

    required = reached.float().argmax(dim=1).to(torch.long) + 1
    required = torch.where(
        totals > 1e-12,
        required,
        torch.full_like(required, minimum_k),
    )
    adaptive_k = required.clamp(minimum_k, maximum_k)

    selection = deterministic_select(work, adaptive_k)

    selected_mass = (
        work * selection.selected_mask.to(work.dtype)
    ).sum(dim=1)

    retained = torch.where(
        totals > 1e-12,
        selected_mass / totals,
        torch.zeros_like(totals),
    )

    return AdaptiveDetailSelection(
        scores=scores,
        selected_mask=selection.selected_mask,
        selected_indices=selection.selected_indices,
        adaptive_k=adaptive_k,
        retained_fraction=retained,
    )


class DetailAdapter(nn.Module):
    """Map raw Haar residuals into the late transformer feature space."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim, bias=False),
        )

    def forward(self, details: torch.Tensor) -> torch.Tensor:
        if details.ndim != 4 or details.shape[2] != 3:
            raise ValueError("details must have shape [B, 256, 3, D]")
        return self.network(details)


class UtilityCalibrator(nn.Module):
    """Predict whether the detail branch is preferable per sample."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        initial_weight: float,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        final = self.network[-1]
        assert isinstance(final, nn.Linear)

        nn.init.zeros_(final.weight)
        nn.init.constant_(
            final.bias,
            math.log(initial_weight / (1.0 - initial_weight)),
        )

    @staticmethod
    def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
        classes = probabilities.shape[1]
        denominator = max(math.log(classes), 1e-12)

        return -(
            probabilities
            * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1) / denominator

    def forward(
        self,
        global_logits: torch.Tensor,
        detail_logits: torch.Tensor,
        retained_fraction: torch.Tensor,
        k_fraction: torch.Tensor,
    ) -> torch.Tensor:
        if global_logits.shape != detail_logits.shape:
            raise ValueError("branch logits must have identical shapes")
        if global_logits.ndim != 2:
            raise ValueError("branch logits must have shape [B, C]")

        batch_size = global_logits.shape[0]

        if retained_fraction.shape != (batch_size,):
            raise ValueError("retained_fraction must have shape [B]")
        if k_fraction.shape != (batch_size,):
            raise ValueError("k_fraction must have shape [B]")

        global_probability = F.softmax(
            global_logits.float(),
            dim=1,
        )
        detail_probability = F.softmax(
            detail_logits.float(),
            dim=1,
        )

        midpoint = 0.5 * (
            global_probability + detail_probability
        )

        global_kl = (
            global_probability
            * (
                global_probability.clamp_min(1e-8).log()
                - midpoint.clamp_min(1e-8).log()
            )
        ).sum(dim=1)

        detail_kl = (
            detail_probability
            * (
                detail_probability.clamp_min(1e-8).log()
                - midpoint.clamp_min(1e-8).log()
            )
        ).sum(dim=1)

        disagreement = 0.5 * (global_kl + detail_kl)

        global_confidence = global_probability.amax(dim=1)
        detail_confidence = detail_probability.amax(dim=1)

        features = torch.stack(
            (
                global_confidence,
                detail_confidence,
                self._entropy(global_probability),
                self._entropy(detail_probability),
                disagreement,
                detail_confidence - global_confidence,
                retained_fraction.float(),
                k_fraction.float(),
            ),
            dim=1,
        )

        return torch.sigmoid(
            self.network(features)
        ).squeeze(1)


class DeltaSubV2(nn.Module):
    """Late dual-path DINOv2 refinement with exact Haar detail tokens."""

    def __init__(
        self,
        backbone: nn.Module,
        class_count: int,
        config: DeltaSubV2Config,
    ) -> None:
        super().__init__()
        config.validate()

        if class_count <= 0:
            raise ValueError("class_count must be positive")

        blocks = list(backbone.model.blocks)
        depth = len(blocks)

        if config.insertion_block + config.trainable_blocks != depth:
            raise ValueError(
                "insertion_block + trainable_blocks must equal "
                f"backbone depth, received "
                f"{config.insertion_block} + "
                f"{config.trainable_blocks} != {depth}"
            )

        self.backbone = backbone
        self.config = config
        self.class_count = int(class_count)

        # Freeze the early trunk and train exactly the final N blocks.
        self.backbone.set_trainable_blocks(config.trainable_blocks)

        self.child = backbone.build_child_projector(trainable=True)
        self.haar = HaarDetails()
        self.detail_adapter = DetailAdapter(
            config.feature_dim,
            config.detail_adapter_hidden_dim,
        )

        self.mode_embeddings = nn.Parameter(
            torch.zeros(3, config.feature_dim)
        )

        initial_scale_logit = math.log(
            config.initial_detail_scale
            / (1.0 - config.initial_detail_scale)
        )
        self.detail_scale_logit = nn.Parameter(
            torch.tensor(initial_scale_logit)
        )

        self.head = nn.Linear(config.feature_dim, class_count)
        self.utility = UtilityCalibrator(
            config.utility_hidden_dim,
            initial_weight=config.initial_fusion_weight,
        )

        self._apply_training_policy()

    @property
    def trunk_blocks(self) -> tuple[nn.Module, ...]:
        return tuple(
            self.backbone.model.blocks[
                : self.config.insertion_block
            ]
        )

    @property
    def tail_blocks(self) -> tuple[nn.Module, ...]:
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

        self.backbone.model.norm.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_training_policy()
        return self

    def _run_trunk(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or tuple(images.shape[1:]) != (
            3,
            224,
            224,
        ):
            raise ValueError("images must have shape [B, 3, 224, 224]")

        batch_size = images.shape[0]

        # The early trunk is frozen. Avoid storing ten blocks of activations.
        with torch.no_grad():
            parents = self.backbone.pre_transformer_parent_embeddings(
                images
            )
            parent_positions = (
                self.backbone.parent_patch_positions()
                .to(parents)
                .expand(batch_size, -1, -1)
            )
            prefix = (
                self.backbone.prefix_tokens_with_positions(
                    batch_size
                )
                .to(parents)
            )

            tokens = torch.cat(
                (
                    prefix,
                    parents + parent_positions,
                ),
                dim=1,
            )

            for block in self.trunk_blocks:
                tokens = block(tokens)

        return tokens, parents

    def _tail_encode(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.tail_blocks:
            tokens = block(tokens)

        return self.backbone.model.norm(tokens)[:, 0]

    def _haar_details(
        self,
        images: torch.Tensor,
        parents: torch.Tensor,
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

        return self.haar(consistent)

    def _routing_scores(
        self,
        trunk_tokens: torch.Tensor,
        adapted_details: torch.Tensor,
    ) -> torch.Tensor:
        prefix_count = int(self.backbone.prefix_token_count)

        cls_token = trunk_tokens[:, 0]
        parent_tokens = trunk_tokens[
            :,
            prefix_count : prefix_count + 256,
        ]

        semantic_relevance = F.cosine_similarity(
            F.normalize(parent_tokens.float(), dim=-1),
            F.normalize(cls_token.float(), dim=-1)[:, None, :],
            dim=-1,
        ).clamp_min(0)

        detail_energy = adapted_details.float().square().mean(
            dim=(2, 3)
        ).sqrt()
        detail_energy = torch.log1p(detail_energy)

        normalized_energy = detail_energy / (
            detail_energy.amax(dim=1, keepdim=True)
            .clamp_min(1e-8)
        )

        return semantic_relevance * normalized_energy

    def _selection(
        self,
        trunk_tokens: torch.Tensor,
        adapted_details: torch.Tensor,
    ) -> AdaptiveDetailSelection:
        scores = self._routing_scores(
            trunk_tokens,
            adapted_details,
        )

        return adaptive_energy_select(
            scores,
            minimum_k=self.config.minimum_k,
            maximum_k=self.config.maximum_k,
            retained_fraction=(
                self.config.retained_energy_fraction
            ),
        )

    def _detail_branch(
        self,
        trunk_tokens: torch.Tensor,
        adapted_details: torch.Tensor,
        selection: AdaptiveDetailSelection,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.maximum_k == 0:
            return global_features

        prefix_count = int(self.backbone.prefix_token_count)
        parent_tokens = trunk_tokens[
            :,
            prefix_count : prefix_count + 256,
        ]

        result = global_features.clone()
        mode_embeddings = self.mode_embeddings.to(
            adapted_details
        )
        detail_scale = torch.sigmoid(
            self.detail_scale_logit
        ).to(adapted_details)

        for k_value in torch.unique(
            selection.adaptive_k,
            sorted=True,
        ).tolist():
            k = int(k_value)

            if k == 0:
                continue

            rows = torch.nonzero(
                selection.adaptive_k == k,
                as_tuple=False,
            ).squeeze(1)

            selected = selection.selected_mask[rows]
            group_size = int(rows.numel())

            selected_details = adapted_details[rows][
                selected
            ].reshape(
                group_size,
                k,
                3,
                self.config.feature_dim,
            )

            selected_parents = parent_tokens[rows][
                selected
            ].reshape(
                group_size,
                k,
                self.config.feature_dim,
            )

            detail_tokens = (
                selected_parents.unsqueeze(2)
                + mode_embeddings[None, None, :, :]
                + detail_scale * selected_details
            ).reshape(
                group_size,
                3 * k,
                self.config.feature_dim,
            )

            branch_tokens = torch.cat(
                (
                    trunk_tokens[rows],
                    detail_tokens,
                ),
                dim=1,
            )

            encoded = self._tail_encode(branch_tokens)
            result[rows] = encoded.to(result)

        return result

    def branches(
        self,
        images: torch.Tensor,
    ) -> DeltaSubV2Output:
        trunk_tokens, parents = self._run_trunk(images)

        global_features = self._tail_encode(trunk_tokens)

        raw_details = self._haar_details(
            images,
            parents,
        )
        adapted_details = self.detail_adapter(raw_details)

        selection = self._selection(
            trunk_tokens,
            adapted_details,
        )

        detail_features = self._detail_branch(
            trunk_tokens,
            adapted_details,
            selection,
            global_features,
        )

        global_logits = self.head(global_features)
        detail_logits = self.head(detail_features)

        if self.config.maximum_k:
            k_fraction = (
                selection.adaptive_k.float()
                / float(self.config.maximum_k)
            )
        else:
            k_fraction = torch.zeros_like(
                selection.retained_fraction
            )

        fusion_weight = self.utility(
            global_logits,
            detail_logits,
            selection.retained_fraction,
            k_fraction,
        )

        fused_features = (
            global_features
            + fusion_weight.unsqueeze(1)
            * (detail_features - global_features)
        )

        fused_logits = (
            global_logits
            + fusion_weight.unsqueeze(1)
            * (detail_logits - global_logits)
        )

        return DeltaSubV2Output(
            fused_features=fused_features,
            global_features=global_features,
            detail_features=detail_features,
            fused_logits=fused_logits,
            global_logits=global_logits,
            detail_logits=detail_logits,
            fusion_weight=fusion_weight,
            selection=selection,
        )

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.branches(images).fused_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.branches(images).fused_logits

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
