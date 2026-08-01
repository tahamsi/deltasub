from __future__ import annotations

import torch

from deltasub.models.novelty_preserving import (
    NoveltyGate,
    NoveltyPreservingConfig,
    detail_saliency_scores,
    novelty_preserving_objective,
    protected_residual_fusion,
    prototype_novelty_score,
    select_detail_parents,
)


def test_zero_gate_exactly_recovers_global_path() -> None:
    torch.manual_seed(0)
    global_features = torch.randn(4, 8)
    detail_features = torch.randn(4, 8)
    gate = torch.zeros(4)

    fused = protected_residual_fusion(
        global_features,
        detail_features,
        gate,
    )

    assert torch.equal(fused, global_features)


def test_unit_gate_exactly_recovers_detail_path() -> None:
    torch.manual_seed(1)
    global_features = torch.randn(4, 8)
    detail_features = torch.randn(4, 8)
    gate = torch.ones(4)

    fused = protected_residual_fusion(
        global_features,
        detail_features,
        gate,
    )

    assert torch.equal(fused, detail_features)


def test_novelty_gate_is_bounded_and_respects_zero_prior() -> None:
    torch.manual_seed(2)
    gate = NoveltyGate(feature_dim=8, hidden_dim=4, seed=7)

    global_features = torch.randn(5, 8)
    detail_features = torch.randn(5, 8)
    prior = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])

    values = gate(global_features, detail_features, prior)

    assert values.shape == (5,)
    assert torch.all(values >= 0)
    assert torch.all(values <= 1)
    assert values[0].item() == 0.0


def test_prototype_novelty_is_higher_far_from_known_prototype() -> None:
    prototypes = torch.tensor([[1.0, 0.0]])
    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    novelty = prototype_novelty_score(
        features,
        prototypes,
        threshold=0.5,
        temperature=0.1,
    )

    assert novelty[0] < novelty[1] < novelty[2]
    assert torch.all(novelty >= 0)
    assert torch.all(novelty <= 1)


def test_detail_selection_is_deterministic_and_adaptive() -> None:
    scores = torch.arange(256, dtype=torch.float32).repeat(3, 1)
    novelty = torch.tensor([0.0, 0.5, 1.0])

    first = select_detail_parents(
        scores,
        novelty,
        maximum_k=16,
    )
    second = select_detail_parents(
        scores,
        novelty,
        maximum_k=16,
    )

    assert first.adaptive_k.tolist() == [0, 8, 16]
    assert torch.equal(first.selected_mask, second.selected_mask)
    assert first.selected_indices == second.selected_indices
    assert first.selected_indices[1] == tuple(range(255, 247, -1))


def test_disagreement_increases_saliency_without_changing_shape() -> None:
    torch.manual_seed(3)
    parents = torch.randn(2, 256, 8)
    global_features = torch.randn(2, 8)

    base = detail_saliency_scores(parents, global_features)
    disagreement = torch.zeros(2, 256)
    disagreement[:, 17] = 1.0

    enhanced = detail_saliency_scores(
        parents,
        global_features,
        view_disagreement=disagreement,
    )

    assert base.shape == enhanced.shape == (2, 256)
    assert torch.all(enhanced[:, 17] >= base[:, 17])


def test_objective_is_finite_and_backpropagates() -> None:
    torch.manual_seed(4)
    config = NoveltyPreservingConfig(
        feature_dim=8,
        maximum_detail_parents=4,
        gate_hidden_dim=4,
    )

    fused_one = torch.randn(6, 8, requires_grad=True)
    fused_two = torch.randn(6, 8, requires_grad=True)
    global_one = torch.randn(6, 8)
    global_two = torch.randn(6, 8)
    gate_one = torch.sigmoid(torch.randn(6, requires_grad=True))
    gate_two = torch.sigmoid(torch.randn(6, requires_grad=True))
    labelled = torch.tensor([True, True, False, False, False, True])
    prototypes = torch.randn(3, 8)

    output = novelty_preserving_objective(
        fused_view_one=fused_one,
        fused_view_two=fused_two,
        global_view_one=global_one,
        global_view_two=global_two,
        gate_view_one=gate_one,
        gate_view_two=gate_two,
        labelled_mask=labelled,
        known_prototypes=prototypes,
        config=config,
    )

    assert torch.isfinite(output.total)
    assert torch.isfinite(output.preservation)
    assert torch.isfinite(output.view_consistency)
    assert torch.isfinite(output.novel_dispersion)
    assert torch.isfinite(output.sparsity)

    output.total.backward()

    assert fused_one.grad is not None
    assert fused_two.grad is not None
    assert torch.isfinite(fused_one.grad).all()
    assert torch.isfinite(fused_two.grad).all()
