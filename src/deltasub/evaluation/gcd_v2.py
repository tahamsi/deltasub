"""Exact, fail-closed port of the pinned GCD ``split_cluster_acc_v2`` metric.

The reference uses one global Hungarian assignment and then scores old and new
ground-truth classes under that same assignment.  Values are fractions in [0, 1].
Validation here is intentionally stricter than the reference's implicit assumptions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.hashing import sha256_file

GCD_REVISION = "831a645c3d09a68ec4633a45741025765bacf7e0"
REFERENCE_ROOT = Path("/home/ubuntu/references/generalized-category-discovery-" + GCD_REVISION)
REFERENCE_FILES = (
    "project_utils/cluster_and_log_utils.py",
    "project_utils/cluster_utils.py",
)
REFERENCE_SHA256 = {
    "project_utils/cluster_and_log_utils.py": "bce84cfbd265e3e90d7a814d175e979c9077e2974aef1cb4f35303d74d24bd21",
    "project_utils/cluster_utils.py": "01dbe0d5ca04a4d30afadac0af5105f62868e387554c341568e103026a520702",
}


@dataclass(frozen=True)
class GCDV2Metrics:
    all: float
    old: float
    new: float
    protocol: str = "gcd_v2"
    unit: str = "fraction"

    def as_dict(self) -> dict:
        return asdict(self)


def _labels(value: Iterable[int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if array.dtype.kind not in "iu":
        if array.dtype.kind == "f" and np.isfinite(array).all() and np.equal(array, np.floor(array)).all():
            array = array.astype(np.int64)
        else:
            raise ValueError(f"{name} must contain finite integer labels")
    array = array.astype(np.int64, copy=False)
    if (array < 0).any():
        raise ValueError(f"{name} must contain non-negative labels")
    return array


def evaluate_gcd_v2(y_true, y_pred, old_mask, *, protocol: str = "gcd_v2") -> GCDV2Metrics:
    """Evaluate the pinned GCD v2 semantics, with explicit invalid-input handling."""
    if protocol != "gcd_v2":
        raise ValueError(f"unsupported evaluation protocol: {protocol}")
    true, pred = _labels(y_true, "y_true"), _labels(y_pred, "y_pred")
    mask = np.asarray(old_mask)
    if mask.ndim != 1 or mask.dtype.kind != "b":
        raise ValueError("old_mask must be a one-dimensional boolean array")
    if not (true.size == pred.size == mask.size):
        raise ValueError("y_true, y_pred, and old_mask lengths must match")
    if not mask.any() or mask.all():
        raise ValueError("old and new partitions must both be non-empty")
    old_classes, new_classes = set(true[mask].tolist()), set(true[~mask].tolist())
    if old_classes & new_classes:
        raise ValueError("a ground-truth class cannot occur in both old and new partitions")
    dimension = int(max(pred.max(), true.max())) + 1
    contingency = np.zeros((dimension, dimension), dtype=np.int64)
    np.add.at(contingency, (pred, true), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    mapping = {int(column): int(row) for row, column in zip(rows, columns)}
    # The square matrix guarantees every non-negative true label has a mapping.
    total = float(contingency[rows, columns].sum() / true.size)
    old_count, new_count = int(mask.sum()), int((~mask).sum())
    old = float(sum(contingency[mapping[label], label] for label in old_classes) / old_count)
    new = float(sum(contingency[mapping[label], label] for label in new_classes) / new_count)
    return GCDV2Metrics(total, old, new)


def provenance(*, implementation_path: str | Path | None = None, reference_root: str | Path = REFERENCE_ROOT) -> dict:
    root = Path(reference_root)
    hashes = {name: sha256_file(root / name) for name in REFERENCE_FILES}
    if hashes != REFERENCE_SHA256:
        raise ValueError("pinned GCD evaluator reference hash mismatch")
    import subprocess
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    if revision != GCD_REVISION:
        raise ValueError(f"pinned GCD revision mismatch: {revision}")
    result = {"revision": GCD_REVISION, "reference_root": str(root),
              "reference_files": hashes, "protocol": "gcd_v2",
              "assignment": "single global scipy.optimize.linear_sum_assignment",
              "metric_unit": "fraction"}
    if implementation_path is not None:
        result["implementation_sha256"] = sha256_file(implementation_path)
    return result
