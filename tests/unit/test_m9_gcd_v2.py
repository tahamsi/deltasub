import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deltasub.evaluation.gcd_v2 import REFERENCE_ROOT, evaluate_gcd_v2


def _reference(true, pred, mask):
    # Load the pinned function while stubbing its optional tensorboard import.
    import sys, types
    tensorboard = types.ModuleType("torch.utils.tensorboard")
    tensorboard.SummaryWriter = object
    sys.modules.setdefault("torch.utils.tensorboard", tensorboard)
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda *_args, **_kwargs: None
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        path = REFERENCE_ROOT / "project_utils/cluster_and_log_utils.py"
        spec = importlib.util.spec_from_file_location("pinned_gcd_eval", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module.split_cluster_acc_v2(np.asarray(true), np.asarray(pred), np.asarray(mask, dtype=bool))
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("true,pred,mask", [
    ([0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]),
    ([0, 0, 1, 1], [5, 5, 3, 3], [1, 1, 0, 0]),
    ([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], [1, 1, 0, 0, 0, 0]),
    ([0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]),
    ([0, 0, 1, 1], [0, 4, 2, 5], [1, 1, 0, 0]),
])
def test_exact_reference_fixtures(true, pred, mask):
    observed = evaluate_gcd_v2(true, pred, np.asarray(mask, dtype=bool))
    assert (observed.all, observed.old, observed.new) == pytest.approx(_reference(true, pred, mask))


def test_deterministic_random_reference_equivalence():
    rng = np.random.default_rng(831)
    for _ in range(20):
        true = np.repeat(np.arange(6), 9); pred = rng.integers(0, 9, true.size)
        mask = true < 3
        got = evaluate_gcd_v2(true, pred, mask)
        assert (got.all, got.old, got.new) == pytest.approx(_reference(true, pred, mask))


@pytest.mark.parametrize("true,pred,mask", [
    ([], [], []), ([0, 1], [0], [True, False]), ([0, -1], [0, 1], [True, False]),
    ([0, 1], [0, 1], [True, True]), ([0, 0], [0, 1], [True, False]),
    ([0, 1], [0.5, 1], [True, False]), ([0, 1], [0, 1], [1, 0]),
])
def test_malformed_inputs_fail_closed(true, pred, mask):
    with pytest.raises(ValueError): evaluate_gcd_v2(true, pred, mask)


def test_unsupported_protocol_fails_closed():
    with pytest.raises(ValueError, match="unsupported"): evaluate_gcd_v2([0, 1], [0, 1], [True, False], protocol="v1")
