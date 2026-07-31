from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from deltasub.router.fixture import run_router_fixture
from deltasub.router.training import inspect_router_checkpoint


def stable_diagnostic(value):
    ignored = {"elapsed_seconds", "m4_fixture_diagnostic"}
    return {k: v for k, v in value.items() if k not in ignored}


def test_fixture_training_artifacts_resume_and_independent_determinism():
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        a = run_router_fixture(first)
        b = run_router_fixture(second)
        assert stable_diagnostic(a) == stable_diagnostic(b)
        root = Path(first) / "training"
        for name in ("checkpoint_last.pt", "checkpoint_best.pt", "metrics.jsonl",
                     "environment.json", "resolved_config.yaml", "split_manifest.json",
                     "replay_state.json"):
            assert (root / name).is_file(), name
        inspected = inspect_router_checkpoint(root / "checkpoint_last.pt")
        assert inspected["global_step"] > 0
        assert a["parameter_change_norm"] > 0
        assert a["frozen_state_equal"]
        assert a["coverage_stream_examples"] > 0
        assert a["informative_stream_examples"] > 0
        assert a["replay_examples_used"] > 0
        resumed = run_router_fixture(first, resume=True)
        assert resumed["global_step"] > a["global_step"]
        assert resumed["resumed_global_step"] == resumed["global_step"]
