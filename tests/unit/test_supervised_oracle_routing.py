from __future__ import annotations

import numpy as np
import torch

from deltasub.experiment.supervised_oracle_routing import (
    direct_metrics,
    paired_bootstrap_delta,
    topk_mask,
)


def test_topk_mask_can_drop_nonpositive_candidates() -> None:
    scores = torch.tensor(
        [
            [3.0, 2.0, -1.0, -2.0],
            [-1.0, -2.0, -3.0, -4.0],
        ]
    )

    mask = topk_mask(
        scores,
        3,
        positive_only=True,
    )

    assert mask.tolist() == [
        [True, True, False, False],
        [False, False, False, False],
    ]


def test_direct_metrics_reports_old_new_hmean() -> None:
    targets = np.asarray([0, 1, 2, 3])
    predictions = np.asarray([0, 1, 0, 3])
    old = np.asarray(
        [True, True, False, False]
    )

    metrics = direct_metrics(
        targets,
        predictions,
        old,
    )

    assert metrics["all"] == 0.75
    assert metrics["old"] == 1.0
    assert metrics["new"] == 0.5
    assert abs(
        metrics["hmean"] - (2.0 / 3.0)
    ) < 1e-12


def test_paired_bootstrap_detects_identical_predictions() -> None:
    targets = np.asarray([0, 1, 2, 3])
    predictions = np.asarray([0, 1, 2, 3])
    old = np.asarray(
        [True, True, False, False]
    )

    result = paired_bootstrap_delta(
        targets=targets,
        old=old,
        baseline=predictions,
        candidate=predictions,
        draws=20,
        seed=7,
    )

    for metric in ("all", "old", "new", "hmean"):
        assert result[metric]["mean_delta"] == 0.0
        assert result[metric]["lower_95"] == 0.0
        assert result[metric]["upper_95"] == 0.0
