from __future__ import annotations

import torch

from deltasub.experiment.deltasub_v2_training import (
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
