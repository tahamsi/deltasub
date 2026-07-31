from __future__ import annotations

import tempfile
from pathlib import Path

from deltasub.adaptive.training import inspect_adaptive_checkpoint, run_fixture_training


def stable(value):
    return {k: v for k, v in value.items() if k not in {
        "elapsed_seconds", "router_checkpoint_hash", "final_controller_state_hash",
        "initial_controller_state_hash", "selection_plan_hash"}}


def test_fixture_twice_checkpoint_resume_and_frozen_state():
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        a = run_fixture_training(first)
        b = run_fixture_training(second)
        assert stable(a) == stable(b)
        assert a["selected_k_values"] == [0, 1, 32, 256]
        assert a["padded_versus_bucketed_maximum_valid_output_error"] < 2e-6
        assert a["parameter_change_norm"] > 0
        assert a["frozen_state_equal"]
        assert a["dual_update_count"] == 1
        root = Path(first)
        for name in ("checkpoint_last.pt", "checkpoint_best.pt", "metrics.jsonl",
                     "resolved_config.yaml", "environment.json", "fixture_diagnostic.json"):
            assert (root / name).is_file()
        inspected = inspect_adaptive_checkpoint(root / "checkpoint_last.pt")
        assert inspected["global_step"] == 1
        resumed = run_fixture_training(first, resume=True)
        assert resumed["global_step"] == 2
        assert resumed["resumed_global_step"] == 2
