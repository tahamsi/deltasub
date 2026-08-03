from __future__ import annotations

import torch
from torch import nn

from deltasub.models.deltasub_v2 import (
    DeltaSubV2,
    DeltaSubV2Config,
    adaptive_energy_select,
)


class TinyBlock(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(
            feature_dim,
            feature_dim,
            bias=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        context = self.projection(
            tokens.mean(dim=1, keepdim=True)
        )
        return tokens + 0.1 * context


class FakeChildProjector(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(
            3,
            feature_dim,
            bias=False,
        )

    def forward(self, children: torch.Tensor) -> torch.Tensor:
        pooled = children.mean(dim=(-1, -2))
        return self.projection(pooled)


class FakeBackbone(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.parent_projection = nn.Linear(
            3,
            feature_dim,
            bias=False,
        )

        self.model = nn.Module()
        self.model.blocks = nn.ModuleList(
            [
                TinyBlock(feature_dim)
                for _ in range(12)
            ]
        )
        self.model.norm = nn.LayerNorm(feature_dim)

    @property
    def prefix_token_count(self) -> int:
        return 1

    def set_trainable_blocks(self, count: int):
        for parameter in self.parameters():
            parameter.requires_grad = False

        for block in self.model.blocks[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        for parameter in self.model.norm.parameters():
            parameter.requires_grad = True

        return {}

    def build_child_projector(
        self,
        *,
        trainable: bool = True,
    ):
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
        return torch.zeros(
            1,
            256,
            self.feature_dim,
        )

    def prefix_tokens_with_positions(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            1,
            self.feature_dim,
        )


def make_model(
    *,
    minimum_k: int = 1,
    maximum_k: int = 4,
) -> DeltaSubV2:
    config = DeltaSubV2Config(
        feature_dim=8,
        insertion_block=10,
        trainable_blocks=2,
        minimum_k=minimum_k,
        maximum_k=maximum_k,
        retained_energy_fraction=0.75,
        detail_adapter_hidden_dim=16,
        utility_hidden_dim=8,
    )

    return DeltaSubV2(
        FakeBackbone(8),
        class_count=5,
        config=config,
    )


def test_adaptive_selection_uses_smallest_sufficient_k() -> None:
    scores = torch.zeros(2, 256)
    scores[0, :4] = torch.tensor(
        [4.0, 3.0, 2.0, 1.0]
    )

    selection = adaptive_energy_select(
        scores,
        minimum_k=1,
        maximum_k=4,
        retained_fraction=0.70,
    )

    assert selection.adaptive_k.tolist() == [2, 1]
    assert selection.selected_indices[0] == (0, 1)
    assert selection.selected_indices[1] == (0,)
    assert torch.allclose(
        selection.retained_fraction[0],
        torch.tensor(0.7),
        atol=1e-6,
        rtol=0,
    )
    assert selection.retained_fraction[1].item() == 0.0


def test_only_final_two_backbone_blocks_are_trainable() -> None:
    model = make_model()

    for block in model.backbone.model.blocks[:10]:
        assert not any(
            parameter.requires_grad
            for parameter in block.parameters()
        )

    for block in model.backbone.model.blocks[10:]:
        assert all(
            parameter.requires_grad
            for parameter in block.parameters()
        )

    assert all(
        parameter.requires_grad
        for parameter in model.backbone.model.norm.parameters()
    )


def test_forward_returns_dual_paths_and_bounded_fusion() -> None:
    torch.manual_seed(0)
    model = make_model()
    images = torch.randn(3, 3, 224, 224)

    output = model.branches(images)

    assert output.global_features.shape == (3, 8)
    assert output.detail_features.shape == (3, 8)
    assert output.fused_features.shape == (3, 8)
    assert output.global_logits.shape == (3, 5)
    assert output.detail_logits.shape == (3, 5)
    assert output.fused_logits.shape == (3, 5)
    assert output.selection.selected_mask.shape == (3, 256)
    assert torch.all(output.fusion_weight >= 0)
    assert torch.all(output.fusion_weight <= 1)


def test_zero_detail_budget_exactly_recovers_global_branch() -> None:
    torch.manual_seed(1)
    model = make_model(
        minimum_k=0,
        maximum_k=0,
    )
    model.eval()

    output = model.branches(
        torch.randn(2, 3, 224, 224)
    )

    assert output.selection.adaptive_k.tolist() == [0, 0]
    assert torch.equal(
        output.detail_features,
        output.global_features,
    )
    assert torch.equal(
        output.detail_logits,
        output.global_logits,
    )
    assert torch.equal(
        output.fused_logits,
        output.global_logits,
    )


def test_gradients_flow_through_tail_and_detail_path() -> None:
    torch.manual_seed(2)
    model = make_model()
    model.train()

    output = model.branches(
        torch.randn(2, 3, 224, 224)
    )

    loss = (
        output.fused_logits.square().mean()
        + output.detail_logits.square().mean()
    )
    loss.backward()

    assert all(
        parameter.grad is None
        for block in model.backbone.model.blocks[:10]
        for parameter in block.parameters()
    )

    assert any(
        parameter.grad is not None
        for block in model.backbone.model.blocks[10:]
        for parameter in block.parameters()
    )

    assert any(
        parameter.grad is not None
        for parameter in model.child.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.detail_adapter.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.utility.parameters()
    )
    assert model.head.weight.grad is not None


def test_selection_is_deterministic_under_ties() -> None:
    scores = torch.ones(4, 256)

    first = adaptive_energy_select(
        scores,
        minimum_k=1,
        maximum_k=4,
        retained_fraction=0.75,
    )
    second = adaptive_energy_select(
        scores,
        minimum_k=1,
        maximum_k=4,
        retained_fraction=0.75,
    )

    assert first.adaptive_k.tolist() == [3, 3, 3, 3]
    assert first.selected_indices == second.selected_indices
    assert first.selected_indices[0] == (0, 1, 2)
    assert torch.allclose(
        first.retained_fraction,
        torch.full((4,), 0.75),
        atol=1e-6,
        rtol=0,
    )
