from __future__ import annotations

import torch
from torch import nn

from deltasub.models.novelty_deltasub import (
    KnownPrototypeBank,
    NoveltyPreservingDeltaSub,
)
from deltasub.models.novelty_preserving import (
    NoveltyPreservingConfig,
)


class MeanMixBlock(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        output = tokens.clone()
        output[:, 0] = tokens.mean(dim=1)
        return output


class FakeChildProjector(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.projection = nn.Linear(3, feature_dim, bias=False)

    def forward(self, children: torch.Tensor) -> torch.Tensor:
        pooled = children.mean(dim=(-1, -2))
        return self.projection(pooled)


class FakeBackbone(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.parent_projection = nn.Linear(
            3,
            feature_dim,
            bias=False,
        )
        self.model = nn.Module()
        self.model.blocks = nn.ModuleList([MeanMixBlock()])
        self.model.norm = nn.Identity()

    def build_child_projector(self, *, trainable: bool = True):
        projector = FakeChildProjector(self.feature_dim)
        projector.requires_grad_(trainable)
        return projector

    def pre_transformer_parent_embeddings(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        patches = (
            images.unfold(2, 14, 14)
            .unfold(3, 14, 14)
            .mean(dim=(-1, -2))
            .permute(0, 2, 3, 1)
            .reshape(images.shape[0], 256, 3)
        )
        return self.parent_projection(patches)

    def parent_patch_positions(self) -> torch.Tensor:
        return torch.zeros(1, 256, self.feature_dim)

    def prefix_tokens_with_positions(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            1,
            self.feature_dim,
        )


def make_model(maximum_k: int = 4):
    config = NoveltyPreservingConfig(
        feature_dim=8,
        maximum_detail_parents=maximum_k,
        gate_hidden_dim=4,
    )
    return NoveltyPreservingDeltaSub(
        FakeBackbone(8),
        class_count=5,
        config=config,
        seed=0,
    )


def test_prototype_bank_updates_only_observed_classes() -> None:
    bank = KnownPrototypeBank(5, 3, momentum=0.5)
    features = torch.tensor([
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    targets = torch.tensor([1, 1, 4], dtype=torch.long)

    bank.update(features, targets)

    assert bank.active_class_count == 2
    assert torch.equal(
        bank.prototypes[1],
        torch.tensor([2.0, 0.0, 0.0]),
    )
    assert torch.equal(
        bank.prototypes[4],
        torch.tensor([0.0, 2.0, 0.0]),
    )
    assert bank.counts.tolist() == [0, 2, 0, 0, 1]


def test_model_produces_bounded_adaptive_selection() -> None:
    torch.manual_seed(1)
    model = make_model(maximum_k=4)
    images = torch.randn(3, 3, 224, 224)

    output = model.features(
        images,
        return_auxiliary=True,
    )

    assert output.fused_features.shape == (3, 8)
    assert output.global_features.shape == (3, 8)
    assert output.detail_features.shape == (3, 8)
    assert output.selection.selected_mask.shape == (3, 256)
    assert output.selection.adaptive_k.tolist() == [2, 2, 2]
    assert torch.all(output.gate >= 0)
    assert torch.all(output.gate <= 1)


def test_zero_detail_budget_recovers_global_features() -> None:
    torch.manual_seed(2)
    model = make_model(maximum_k=0)
    images = torch.randn(2, 3, 224, 224)

    output = model.features(
        images,
        return_auxiliary=True,
    )

    assert output.selection.adaptive_k.tolist() == [0, 0]
    assert torch.equal(
        output.fused_features,
        output.global_features,
    )


def test_model_backpropagates_only_through_new_components() -> None:
    torch.manual_seed(3)
    model = make_model(maximum_k=4)
    model.train()

    images = torch.randn(2, 3, 224, 224)
    logits = model(images)
    loss = logits.square().mean()
    loss.backward()

    assert all(
        parameter.grad is None
        for parameter in model.backbone.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.child.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.gate.parameters()
    )
    assert model.head.weight.grad is not None


def test_prototype_update_changes_cold_start_state() -> None:
    torch.manual_seed(4)
    model = make_model(maximum_k=4)
    images = torch.randn(2, 3, 224, 224)

    before = model.features(
        images,
        return_auxiliary=True,
    )
    targets = torch.tensor([0, 1], dtype=torch.long)

    model.update_known_prototypes(
        before.global_features,
        targets,
    )

    after = model.features(
        images,
        return_auxiliary=True,
    )

    assert model.prototype_bank.active_class_count == 2
    assert not torch.equal(before.novelty, after.novelty)


def test_paired_features_use_two_view_contract() -> None:
    torch.manual_seed(5)
    model = make_model(maximum_k=4)
    views = torch.randn(2, 2, 3, 224, 224)

    output = model.paired_features(views)

    assert output.fused_features.shape == (4, 8)
    assert output.selection.selected_mask.shape == (4, 256)
    assert output.selection.adaptive_k.tolist() == [2, 2, 2, 2]



def test_detail_path_normalizes_autocast_output_dtype() -> None:
    torch.manual_seed(6)
    model = make_model(maximum_k=4)
    images = torch.randn(2, 3, 224, 224)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.features(
            images,
            return_auxiliary=True,
        )

    assert (
        output.detail_features.dtype
        == output.global_features.dtype
    )
    assert output.fused_features.dtype == torch.float32
    assert torch.isfinite(output.detail_features).all()
    assert torch.isfinite(output.fused_features).all()
