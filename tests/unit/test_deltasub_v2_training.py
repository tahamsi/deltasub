from __future__ import annotations

import torch

from deltasub.experiment.deltasub_v2_training import (
    _backward_decoupled,
    branch_neighborhood_margin,
    branch_view_utility,
    preservation_penalty,
    utility_target,
)


def test_consistent_confident_views_have_higher_utility() -> None:
    stable = torch.tensor(
        [[[8.0, 0.0], [8.0, 0.0]]]
    )
    unstable = torch.tensor(
        [[[8.0, 0.0], [0.0, 8.0]]]
    )

    stable_value = branch_view_utility(stable)
    unstable_value = branch_view_utility(unstable)

    assert stable_value.item() > unstable_value.item()


def test_labelled_utility_prefers_lower_loss_detail_branch() -> None:
    global_logits = torch.tensor(
        [[[0.0, 3.0], [0.0, 3.0]]]
    )
    detail_logits = torch.tensor(
        [[[4.0, 0.0], [4.0, 0.0]]]
    )

    target = utility_target(
        global_logits=global_logits,
        detail_logits=detail_logits,
        targets=torch.tensor([0]),
        labelled=torch.tensor([True]),
        temperature=0.25,
    )

    assert target.item() > 0.99


def test_unlabelled_utility_prefers_stable_detail_branch() -> None:
    global_logits = torch.tensor(
        [[[8.0, 0.0], [0.0, 8.0]]]
    )
    detail_logits = torch.tensor(
        [[[8.0, 0.0], [8.0, 0.0]]]
    )

    target = utility_target(
        global_logits=global_logits,
        detail_logits=detail_logits,
        targets=torch.tensor([0]),
        labelled=torch.tensor([False]),
        temperature=0.25,
    )

    assert target.item() > 0.5


def test_preservation_penalty_only_charges_degradation() -> None:
    global_logits = torch.tensor(
        [[[3.0, 0.0], [3.0, 0.0]]]
    )
    better_fused = torch.tensor(
        [[[5.0, 0.0], [5.0, 0.0]]]
    )
    worse_fused = torch.tensor(
        [[[0.0, 5.0], [0.0, 5.0]]]
    )
    targets = torch.tensor([0])
    labelled = torch.tensor([True])

    good = preservation_penalty(
        fused_logits=better_fused,
        global_logits=global_logits,
        targets=targets,
        labelled=labelled,
        tolerance=0.01,
    )
    bad = preservation_penalty(
        fused_logits=worse_fused,
        global_logits=global_logits,
        targets=targets,
        labelled=labelled,
        tolerance=0.01,
    )

    assert good.item() == 0.0
    assert bad.item() > 0.0


def test_unlabelled_sentinel_target_never_reaches_cross_entropy() -> None:
    global_logits = torch.tensor(
        [
            [[5.0, 0.0], [5.0, 0.0]],
            [[0.0, 5.0], [5.0, 0.0]],
        ]
    )
    detail_logits = torch.tensor(
        [
            [[6.0, 0.0], [6.0, 0.0]],
            [[5.0, 0.0], [5.0, 0.0]],
        ]
    )

    target = utility_target(
        global_logits=global_logits,
        detail_logits=detail_logits,
        targets=torch.tensor([0, -1]),
        labelled=torch.tensor([True, False]),
        temperature=0.25,
    )

    assert target.shape == (2,)
    assert torch.isfinite(target).all()
    assert torch.all(target >= 0)
    assert torch.all(target <= 1)



def test_neighborhood_margin_rewards_stable_separation() -> None:
    stable = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    collapsed = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )

    assert (
        branch_neighborhood_margin(stable).mean()
        > branch_neighborhood_margin(collapsed).mean()
    )


def test_delta_loss_cannot_modify_backbone_or_head_gradients() -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(
                2,
                2,
                bias=False,
            )
            self.head = torch.nn.Linear(
                2,
                1,
                bias=False,
            )
            self.detail_scale = torch.nn.Parameter(
                torch.tensor(1.0)
            )

    torch.manual_seed(7)
    model = ToyModel()
    inputs = torch.randn(3, 2)

    representation = model.backbone(inputs)
    global_output = model.head(representation)
    detail_output = model.head(
        representation + model.detail_scale * inputs
    )

    global_loss = global_output.square().mean()
    delta_loss = detail_output.square().mean()

    protected = [
        model.backbone.weight,
        model.head.weight,
    ]

    expected = torch.autograd.grad(
        global_loss,
        protected,
        retain_graph=True,
    )

    _backward_decoupled(
        model=model,
        global_loss=global_loss,
        delta_loss=delta_loss,
        accumulation=1,
        variant="deltasub_v2",
    )

    for parameter, gradient in zip(
        protected,
        expected,
        strict=True,
    ):
        assert torch.allclose(
            parameter.grad,
            gradient,
            atol=1e-7,
            rtol=1e-6,
        )

    assert model.detail_scale.grad is not None
    assert model.detail_scale.grad.abs().item() > 0


def test_utility_target_remains_float32_under_autocast() -> None:
    global_logits = torch.tensor(
        [
            [[4.0, 0.0], [4.0, 0.0]],
            [[0.0, 4.0], [4.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    detail_logits = torch.tensor(
        [
            [[5.0, 0.0], [5.0, 0.0]],
            [[4.0, 0.0], [4.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    global_features = torch.randn(2, 2, 8)
    detail_features = torch.randn(2, 2, 8)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        target = utility_target(
            global_logits=global_logits,
            detail_logits=detail_logits,
            global_features=global_features,
            detail_features=detail_features,
            targets=torch.tensor([0, -1]),
            labelled=torch.tensor([True, False]),
            temperature=0.25,
        )

    assert target.dtype == torch.float32
    assert target.shape == (2,)
    assert torch.isfinite(target).all()


def test_joint_backward_includes_delta_gradients() -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(
                2,
                2,
                bias=False,
            )
            self.head = torch.nn.Linear(
                2,
                1,
                bias=False,
            )
            self.detail_scale = torch.nn.Parameter(
                torch.tensor(1.0)
            )

    torch.manual_seed(17)
    model = ToyModel()
    inputs = torch.randn(3, 2)

    representation = model.backbone(inputs)
    global_output = model.head(representation)
    detail_output = model.head(
        representation + model.detail_scale * inputs
    )

    global_loss = global_output.square().mean()
    delta_loss = detail_output.square().mean()

    parameters = [
        model.backbone.weight,
        model.head.weight,
        model.detail_scale,
    ]

    expected = torch.autograd.grad(
        global_loss + delta_loss,
        parameters,
        retain_graph=True,
    )

    _backward_decoupled(
        model=model,
        global_loss=global_loss,
        delta_loss=delta_loss,
        accumulation=1,
        variant="deltasub_v2",
        decouple_delta_gradients=False,
    )

    for parameter, gradient in zip(
        parameters,
        expected,
        strict=True,
    ):
        assert torch.allclose(
            parameter.grad,
            gradient,
            atol=1e-7,
            rtol=1e-6,
        )
