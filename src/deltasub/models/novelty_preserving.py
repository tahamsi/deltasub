"""Novelty-Preserving DeltaSub core components.

The global DINOv2 representation remains an explicit protected path.
Local-detail features enter only through a bounded residual gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..adaptive.selection import deterministic_select


@dataclass(frozen=True)
class NoveltyPreservingConfig:
    feature_dim: int = 768
    maximum_detail_parents: int = 16
    gate_hidden_dim: int = 128
    novelty_temperature: float = 0.1
    known_similarity_threshold: float = 0.5
    preservation_weight: float = 1.0
    view_consistency_weight: float = 1.0
    novel_dispersion_weight: float = 0.25
    sparsity_weight: float = 0.01
    dispersion_margin: float = 0.5

    def validate(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not 0 <= self.maximum_detail_parents <= 256:
            raise ValueError("maximum_detail_parents must be in [0, 256]")
        if self.gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")
        if self.novelty_temperature <= 0:
            raise ValueError("novelty_temperature must be positive")
        if not -1 <= self.known_similarity_threshold <= 1:
            raise ValueError("known_similarity_threshold must be in [-1, 1]")
        if not -1 <= self.dispersion_margin <= 1:
            raise ValueError("dispersion_margin must be in [-1, 1]")

        weights = (
            self.preservation_weight,
            self.view_consistency_weight,
            self.novel_dispersion_weight,
            self.sparsity_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("objective weights must be nonnegative")


@dataclass(frozen=True)
class DetailSelection:
    scores: torch.Tensor
    selected_mask: torch.Tensor
    selected_indices: tuple[tuple[int, ...], ...]
    adaptive_k: torch.Tensor


@dataclass(frozen=True)
class NoveltyPreservingLoss:
    total: torch.Tensor
    preservation: torch.Tensor
    view_consistency: torch.Tensor
    novel_dispersion: torch.Tensor
    sparsity: torch.Tensor


def _require_feature_matrix(
    tensor: torch.Tensor,
    *,
    feature_dim: int,
    name: str,
) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != feature_dim:
        raise ValueError(
            f"{name} must have shape [B, {feature_dim}], "
            f"received {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")


def prototype_novelty_score(
    global_features: torch.Tensor,
    known_prototypes: torch.Tensor,
    *,
    threshold: float = 0.5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Return a bounded known-prototype distance score in [0, 1].

    Larger values indicate that a sample is less similar to every known-class
    prototype and may therefore benefit from additional local detail.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if global_features.ndim != 2 or known_prototypes.ndim != 2:
        raise ValueError("features and prototypes must be matrices")
    if global_features.shape[1] != known_prototypes.shape[1]:
        raise ValueError("feature/prototype dimensions must match")
    if known_prototypes.shape[0] == 0:
        raise ValueError("at least one known prototype is required")
    if not global_features.is_floating_point():
        raise TypeError("global_features must be floating point")
    if not known_prototypes.is_floating_point():
        raise TypeError("known_prototypes must be floating point")
    if not torch.isfinite(global_features).all():
        raise ValueError("global_features must be finite")
    if not torch.isfinite(known_prototypes).all():
        raise ValueError("known_prototypes must be finite")

    normalized_features = F.normalize(global_features.float(), dim=-1)
    normalized_prototypes = F.normalize(known_prototypes.float(), dim=-1)

    maximum_known_similarity = (
        normalized_features @ normalized_prototypes.transpose(0, 1)
    ).amax(dim=1)

    return torch.sigmoid(
        (float(threshold) - maximum_known_similarity) / float(temperature)
    )


def detail_saliency_scores(
    parent_embeddings: torch.Tensor,
    global_features: torch.Tensor,
    *,
    view_disagreement: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score parents by object relevance and optional cross-view disagreement."""

    if parent_embeddings.ndim != 3 or parent_embeddings.shape[1] != 256:
        raise ValueError("parent_embeddings must have shape [B, 256, D]")
    if global_features.shape != (
        parent_embeddings.shape[0],
        parent_embeddings.shape[2],
    ):
        raise ValueError("global_features must have shape [B, D]")
    if not parent_embeddings.is_floating_point():
        raise TypeError("parent_embeddings must be floating point")
    if not global_features.is_floating_point():
        raise TypeError("global_features must be floating point")
    if not torch.isfinite(parent_embeddings).all():
        raise ValueError("parent_embeddings must be finite")
    if not torch.isfinite(global_features).all():
        raise ValueError("global_features must be finite")

    parents = F.normalize(parent_embeddings.float(), dim=-1)
    global_direction = F.normalize(global_features.float(), dim=-1)

    object_relevance = (
        parents * global_direction[:, None, :]
    ).sum(dim=-1)

    # Convert cosine similarity from [-1, 1] to [0, 1].
    object_relevance = object_relevance.add(1.0).mul(0.5)

    if view_disagreement is None:
        return object_relevance

    if view_disagreement.shape != object_relevance.shape:
        raise ValueError("view_disagreement must have shape [B, 256]")
    if not view_disagreement.is_floating_point():
        raise TypeError("view_disagreement must be floating point")
    if not torch.isfinite(view_disagreement).all():
        raise ValueError("view_disagreement must be finite")

    disagreement = view_disagreement.float().clamp_min(0)
    denominator = disagreement.amax(dim=1, keepdim=True).clamp_min(1e-8)
    normalized_disagreement = disagreement / denominator

    return object_relevance * (1.0 + normalized_disagreement)


def select_detail_parents(
    scores: torch.Tensor,
    novelty: torch.Tensor,
    *,
    maximum_k: int,
) -> DetailSelection:
    """Select an adaptive number of parents using deterministic ranking."""

    if scores.ndim != 2 or scores.shape[1] != 256:
        raise ValueError("scores must have shape [B, 256]")
    if novelty.shape != (scores.shape[0],):
        raise ValueError("novelty must have shape [B]")
    if not scores.is_floating_point() or not novelty.is_floating_point():
        raise TypeError("scores and novelty must be floating point")
    if not torch.isfinite(scores).all() or not torch.isfinite(novelty).all():
        raise ValueError("scores and novelty must be finite")
    if not 0 <= maximum_k <= 256:
        raise ValueError("maximum_k must be in [0, 256]")

    bounded_novelty = novelty.clamp(0, 1)
    adaptive_k = torch.round(
        bounded_novelty * float(maximum_k)
    ).to(dtype=torch.long)

    selection = deterministic_select(scores, adaptive_k)

    return DetailSelection(
        scores=scores,
        selected_mask=selection.selected_mask,
        selected_indices=selection.selected_indices,
        adaptive_k=adaptive_k,
    )


class NoveltyGate(nn.Module):
    """Bounded sample gate for protected residual fusion."""

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()

        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("gate dimensions must be positive")

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        global_features: torch.Tensor,
        detail_features: torch.Tensor,
        novelty_prior: torch.Tensor,
    ) -> torch.Tensor:
        _require_feature_matrix(
            global_features,
            feature_dim=self.feature_dim,
            name="global_features",
        )
        _require_feature_matrix(
            detail_features,
            feature_dim=self.feature_dim,
            name="detail_features",
        )

        if global_features.shape != detail_features.shape:
            raise ValueError("global and detail feature shapes must match")
        if novelty_prior.shape != (global_features.shape[0],):
            raise ValueError("novelty_prior must have shape [B]")
        if not novelty_prior.is_floating_point():
            raise TypeError("novelty_prior must be floating point")
        if not torch.isfinite(novelty_prior).all():
            raise ValueError("novelty_prior must be finite")

        global_normalized = F.normalize(global_features.float(), dim=-1)
        detail_normalized = F.normalize(detail_features.float(), dim=-1)

        cosine = (
            global_normalized * detail_normalized
        ).sum(dim=-1, keepdim=True)

        residual = detail_features.float() - global_features.float()
        residual_ratio = (
            residual.norm(dim=-1)
            / global_features.float().norm(dim=-1).clamp_min(1e-6)
        ).clamp_max(10.0).unsqueeze(-1)

        bounded_prior = novelty_prior.float().clamp(0, 1).unsqueeze(-1)

        gate_input = torch.cat(
            (cosine, residual_ratio, bounded_prior),
            dim=1,
        )
        learned_gate = torch.sigmoid(self.network(gate_input)).squeeze(-1)

        # Exact safety property: a zero novelty prior disables the detail path.
        return learned_gate * bounded_prior.squeeze(-1)


def protected_residual_fusion(
    global_features: torch.Tensor,
    detail_features: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse local detail while retaining an explicit global identity path."""

    if global_features.shape != detail_features.shape:
        raise ValueError("global and detail feature shapes must match")
    if global_features.ndim != 2:
        raise ValueError("features must have shape [B, D]")
    if gate.shape != (global_features.shape[0],):
        raise ValueError("gate must have shape [B]")
    if not global_features.is_floating_point():
        raise TypeError("global_features must be floating point")
    if not detail_features.is_floating_point():
        raise TypeError("detail_features must be floating point")
    if not gate.is_floating_point():
        raise TypeError("gate must be floating point")
    if not torch.isfinite(global_features).all():
        raise ValueError("global_features must be finite")
    if not torch.isfinite(detail_features).all():
        raise ValueError("detail_features must be finite")
    if not torch.isfinite(gate).all():
        raise ValueError("gate must be finite")

    bounded_gate = gate.clamp(0, 1).unsqueeze(-1)

    fused = global_features + bounded_gate * (
        detail_features - global_features
    )

    # Preserve exact endpoint identities despite floating-point subtraction
    # and addition rounding.
    fused = torch.where(
        bounded_gate == 0,
        global_features,
        fused,
    )
    fused = torch.where(
        bounded_gate == 1,
        detail_features,
        fused,
    )

    return fused


def novelty_preserving_objective(
    *,
    fused_view_one: torch.Tensor,
    fused_view_two: torch.Tensor,
    global_view_one: torch.Tensor,
    global_view_two: torch.Tensor,
    gate_view_one: torch.Tensor,
    gate_view_two: torch.Tensor,
    labelled_mask: torch.Tensor,
    known_prototypes: torch.Tensor | None,
    config: NoveltyPreservingConfig,
) -> NoveltyPreservingLoss:
    """Auxiliary objective protecting known semantics and novel diversity."""

    config.validate()

    feature_tensors = (
        fused_view_one,
        fused_view_two,
        global_view_one,
        global_view_two,
    )

    if any(tensor.ndim != 2 for tensor in feature_tensors):
        raise ValueError("all feature tensors must have shape [B, D]")
    if len({tuple(tensor.shape) for tensor in feature_tensors}) != 1:
        raise ValueError("all feature tensors must have identical shapes")
    if fused_view_one.shape[1] != config.feature_dim:
        raise ValueError("feature dimension does not match configuration")

    batch_size = fused_view_one.shape[0]

    if gate_view_one.shape != (batch_size,):
        raise ValueError("gate_view_one must have shape [B]")
    if gate_view_two.shape != (batch_size,):
        raise ValueError("gate_view_two must have shape [B]")
    if labelled_mask.shape != (batch_size,):
        raise ValueError("labelled_mask must have shape [B]")
    if labelled_mask.dtype != torch.bool:
        raise TypeError("labelled_mask must be boolean")

    zero = fused_view_one.float().sum() * 0.0

    if labelled_mask.any():
        preservation = 0.5 * (
            F.mse_loss(
                fused_view_one[labelled_mask].float(),
                global_view_one[labelled_mask].float(),
            )
            + F.mse_loss(
                fused_view_two[labelled_mask].float(),
                global_view_two[labelled_mask].float(),
            )
        )
    else:
        preservation = zero

    view_consistency = (
        1.0
        - F.cosine_similarity(
            fused_view_one.float(),
            fused_view_two.float(),
            dim=-1,
        )
    ).mean()

    unlabelled_mask = ~labelled_mask

    if known_prototypes is not None and unlabelled_mask.any():
        if known_prototypes.ndim != 2:
            raise ValueError("known_prototypes must have shape [C, D]")
        if known_prototypes.shape[1] != config.feature_dim:
            raise ValueError("prototype feature dimension mismatch")
        if known_prototypes.shape[0] == 0:
            raise ValueError("known_prototypes cannot be empty")

        normalized_prototypes = F.normalize(
            known_prototypes.float(),
            dim=-1,
        )

        unlabelled_features = F.normalize(
            torch.cat(
                (
                    fused_view_one[unlabelled_mask],
                    fused_view_two[unlabelled_mask],
                ),
                dim=0,
            ).float(),
            dim=-1,
        )

        maximum_known_similarity = (
            unlabelled_features
            @ normalized_prototypes.transpose(0, 1)
        ).amax(dim=1)

        novel_dispersion = F.relu(
            maximum_known_similarity - config.dispersion_margin
        ).mean()
    else:
        novel_dispersion = zero

    sparsity = 0.5 * (
        gate_view_one.float().clamp(0, 1).mean()
        + gate_view_two.float().clamp(0, 1).mean()
    )

    total = (
        config.preservation_weight * preservation
        + config.view_consistency_weight * view_consistency
        + config.novel_dispersion_weight * novel_dispersion
        + config.sparsity_weight * sparsity
    )

    return NoveltyPreservingLoss(
        total=total,
        preservation=preservation,
        view_consistency=view_consistency,
        novel_dispersion=novel_dispersion,
        sparsity=sparsity,
    )
