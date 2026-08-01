from __future__ import annotations

import pytest

from deltasub.experiment.novelty_training import (
    _comparison,
    _method_configuration,
    load_config,
)


def test_real_diagnostic_configuration_is_valid() -> None:
    config = load_config(
        "configs/novelty_preserving/cub_seed0.yaml"
    )
    method = _method_configuration(config)

    assert config["training"]["epochs"] == 20
    assert (
        config["training"]["physical_batch_size"]
        * config["training"]["gradient_accumulation"]
        == 128
    )
    assert method.maximum_detail_parents == 16


def test_baseline_comparison_uses_all_old_new() -> None:
    baseline = {
        "method": "baseline",
        "metrics": {
            "all": 0.40,
            "old": 0.50,
            "new": 0.30,
        },
    }
    ours = {
        "all": 0.45,
        "old": 0.48,
        "new": 0.42,
    }

    comparison = _comparison(baseline, ours)

    assert comparison["absolute_delta"]["all"] == pytest.approx(0.05)
    assert comparison["absolute_delta"]["old"] == pytest.approx(-0.02)
    assert comparison["absolute_delta"]["new"] == pytest.approx(0.12)
    assert comparison["verdict"] == "positive"
