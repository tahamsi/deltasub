from __future__ import annotations

import numpy as np
import torch

from deltasub.experiment.old_label_utility_router import (
    partition_routing_metrics,
    sanitize_utility,
    stable_split,
    utility_router_loss,
)


def test_stable_split_is_disjoint_and_repeatable() -> None:
    sample_ids = [
        f"sample-{index}"
        for index in range(20)
    ]

    first = stable_split(
        sample_ids,
        seed=7,
    )
    second = stable_split(
        sample_ids,
        seed=7,
    )

    assert first == second

    training, validation = first

    assert training
    assert validation
    assert not (
        set(training)
        & set(validation)
    )
    assert sorted(
        training + validation
    ) == list(range(20))


def test_utility_router_loss_is_finite() -> None:
    torch.manual_seed(3)

    scores = torch.randn(4, 256)
    utility = torch.randn(4, 256)

    loss, parts = utility_router_loss(
        scores,
        utility,
    )

    assert torch.isfinite(loss)
    assert parts["total"] > 0


def test_sanitize_utility_replaces_non_candidates() -> None:
    utility = torch.tensor(
        [
            [
                1.0,
                0.5,
                float("-inf"),
            ]
        ]
    )

    sanitized = sanitize_utility(
        utility
    )

    assert torch.isfinite(
        sanitized
    ).all()
    assert sanitized[0, 2] < 0.5


def test_partition_metrics_separate_old_and_new() -> None:
    utility = np.asarray(
        [
            [3.0, 0.0],
            [0.0, 4.0],
            [2.0, 0.0],
            [0.0, 5.0],
        ],
        dtype=np.float32,
    )

    selected = np.asarray(
        [0, 0, 0, 1],
        dtype=np.int64,
    )
    old = np.asarray(
        [True, True, False, False]
    )

    metrics = partition_routing_metrics(
        selected_indices=selected,
        utility=utility,
        old=old,
    )

    assert metrics["all"][
        "top1_recall"
    ] == 0.75
    assert metrics["old"][
        "top1_recall"
    ] == 0.5
    assert metrics["new"][
        "top1_recall"
    ] == 1.0
