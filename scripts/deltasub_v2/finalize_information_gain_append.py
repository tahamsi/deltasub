#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deltasub.evaluation.gcd_v2 import evaluate_gcd_v2
from deltasub.experiment.deltasub_residual_compare import (
    bootstrap_hmean_delta,
    locate_matched_baseline,
)
from deltasub.experiment.training import _atomic_json
from deltasub.utils.hashing import sha256_file


ROOT = Path("artifacts/deltasub_information_gain/cub/seed_0")
K_VALUES = (1, 4, 8, 16)


def pick(archive, names):
    for name in names:
        if name in archive.files:
            return archive[name], name
    raise KeyError(
        f"none of {names} found; available keys: {archive.files}"
    )


def score(target, prediction, old):
    raw = evaluate_gcd_v2(
        target,
        prediction,
        old,
    ).as_dict()

    old_score = float(raw["old"])
    new_score = float(raw["new"])

    hmean = (
        2.0 * old_score * new_score
        / (old_score + new_score)
        if old_score + new_score > 0
        else 0.0
    )

    return {
        "all": float(raw["all"]),
        "old": old_score,
        "new": new_score,
        "hmean": hmean,
    }


candidate_path = ROOT / "predictions.npz"

if not candidate_path.is_file():
    raise FileNotFoundError(candidate_path)

baseline_result_path, baseline_result = locate_matched_baseline()

baseline_path = (
    baseline_result_path.parent
    / "predictions.npz"
)

if not baseline_path.is_file():
    raise FileNotFoundError(baseline_path)

candidate = np.load(
    candidate_path,
    allow_pickle=False,
)
baseline = np.load(
    baseline_path,
    allow_pickle=False,
)

target, candidate_target_key = pick(
    candidate,
    ("target", "targets", "y_true"),
)
old, candidate_old_key = pick(
    candidate,
    ("old", "old_mask"),
)

reference_target, reference_target_key = pick(
    baseline,
    ("target", "targets", "y_true"),
)
reference_old, reference_old_key = pick(
    baseline,
    ("old", "old_mask"),
)
reference_prediction, reference_prediction_key = pick(
    baseline,
    (
        "prediction",
        "predictions",
        "fused_prediction",
        "global_prediction",
        "candidate_prediction",
    ),
)

target = target.astype(np.int64)
old = old.astype(bool)
reference_target = reference_target.astype(np.int64)
reference_old = reference_old.astype(bool)
reference_prediction = reference_prediction.astype(np.int64)

if not np.array_equal(target, reference_target):
    raise RuntimeError(
        "candidate and matched-baseline targets differ"
    )

if not np.array_equal(old, reference_old):
    raise RuntimeError(
        "candidate and matched-baseline Old/New masks differ"
    )

baseline_metrics = score(
    target,
    reference_prediction,
    old,
)

for metric in ("all", "old", "new"):
    expected = float(
        baseline_result["metrics"][metric]
    )
    observed = baseline_metrics[metric]

    if abs(expected - observed) > 1e-12:
        raise RuntimeError(
            f"archived baseline mismatch for {metric}: "
            f"{observed} versus {expected}"
        )

reconstructed, reconstructed_key = pick(
    candidate,
    ("baseline_prediction",),
)
reconstructed = reconstructed.astype(np.int64)

reconstruction_mismatch = (
    reconstructed != reference_prediction
)

methods = {
    "information_gain": {},
    "energy": {},
    "random": {},
}
prediction_arrays = {
    "information_gain": {},
    "energy": {},
    "random": {},
}

available_k = []

for k in K_VALUES:
    required = [
        f"information_gain_k{k}",
        f"energy_k{k}",
        f"random_k{k}",
    ]

    if not all(
        key in candidate.files
        for key in required
    ):
        continue

    available_k.append(k)

    for method in methods:
        prediction = candidate[
            f"{method}_k{k}"
        ].astype(np.int64)

        if prediction.shape != target.shape:
            raise RuntimeError(
                f"{method} K={k} has invalid shape"
            )

        prediction_arrays[method][k] = prediction
        methods[method][str(k)] = score(
            target,
            prediction,
            old,
        )

if not available_k:
    raise RuntimeError(
        "no completed candidate predictions found"
    )

gates = {}

for k in available_k:
    key = str(k)
    candidate_metrics = methods[
        "information_gain"
    ][key]

    delta = {
        metric: (
            candidate_metrics[metric]
            - baseline_metrics[metric]
        )
        for metric in (
            "all",
            "old",
            "new",
            "hmean",
        )
    }

    interval = bootstrap_hmean_delta(
        targets=target,
        old=old,
        baseline=reference_prediction,
        candidate=prediction_arrays[
            "information_gain"
        ][k],
        draws=2000,
        seed=91000 + k,
    )

    best_control_hmean = max(
        methods["energy"][key]["hmean"],
        methods["random"][key]["hmean"],
    )

    passed = (
        delta["all"] >= 0.0
        and delta["new"] >= 0.005
        and delta["hmean"] >= 0.005
        and interval["lower_95"] > 0.0
        and candidate_metrics["hmean"]
        >= best_control_hmean
    )

    gates[key] = {
        "delta": delta,
        "hmean_paired_bootstrap": interval,
        "best_control_hmean": best_control_hmean,
        "passed": passed,
    }

passing = [
    k
    for k in available_k
    if gates[str(k)]["passed"]
]

best_k = max(
    available_k,
    key=lambda k: methods[
        "information_gain"
    ][str(k)]["hmean"],
)

verdict = "follow" if passing else "abandon"

result = {
    "schema_version": (
        "deltasub.information-gain-append.v1"
    ),
    "status": "completed",
    "dataset": "cub",
    "seed": 0,
    "method": (
        "information_gain_append_subtokens"
    ),
    "oracle_used": False,
    "test_labels_used_for_training": False,
    "test_labels_used_for_selection": False,
    "original_tokens_retained": True,
    "subtokens_appended": True,
    "feature_injection": False,
    "baseline": baseline_metrics,
    "metrics": methods,
    "gates": gates,
    "passing_k_values": passing,
    "best_information_gain_k": best_k,
    "baseline_reconstruction": {
        "mismatch_count": int(
            reconstruction_mismatch.sum()
        ),
        "mismatch_fraction": float(
            reconstruction_mismatch.mean()
        ),
        "old_mismatch_count": int(
            reconstruction_mismatch[old].sum()
        ),
        "new_mismatch_count": int(
            reconstruction_mismatch[~old].sum()
        ),
        "candidate_key": reconstructed_key,
        "reference_key": reference_prediction_key,
    },
    "archives": {
        "candidate": str(candidate_path),
        "candidate_sha256": sha256_file(
            candidate_path
        ),
        "baseline": str(baseline_path),
        "baseline_sha256": sha256_file(
            baseline_path
        ),
        "candidate_target_key": (
            candidate_target_key
        ),
        "candidate_old_key": candidate_old_key,
        "reference_target_key": (
            reference_target_key
        ),
        "reference_old_key": reference_old_key,
    },
    "gate_rule": {
        "all_delta_minimum": 0.0,
        "new_delta_minimum": 0.005,
        "hmean_delta_minimum": 0.005,
        "hmean_lower_95_above_zero": True,
        "hmean_not_below_controls": True,
    },
    "verdict": verdict,
}

result_path = ROOT / "result.json"
_atomic_json(result_path, result)

print(
    "method                    "
    "K       all       old       new      hmean"
)

print(
    f"{'matched_selex':<25}"
    f"{'-':>3}"
    f"{baseline_metrics['all']:>10.6f}"
    f"{baseline_metrics['old']:>10.6f}"
    f"{baseline_metrics['new']:>10.6f}"
    f"{baseline_metrics['hmean']:>11.6f}"
)

for k in available_k:
    key = str(k)

    for label, method in (
        ("information_gain", "information_gain"),
        ("energy_control", "energy"),
        ("random_control", "random"),
    ):
        value = methods[method][key]

        print(
            f"{label:<25}"
            f"{k:>3}"
            f"{value['all']:>10.6f}"
            f"{value['old']:>10.6f}"
            f"{value['new']:>10.6f}"
            f"{value['hmean']:>11.6f}"
        )

    gate = gates[key]
    delta = gate["delta"]
    interval = gate[
        "hmean_paired_bootstrap"
    ]

    print(
        f"  K={k}: "
        f"All={delta['all']:+.6f}, "
        f"New={delta['new']:+.6f}, "
        f"H={delta['hmean']:+.6f}, "
        f"H95=[{interval['lower_95']:+.6f}, "
        f"{interval['upper_95']:+.6f}], "
        f"gate="
        f"{'PASSED' if gate['passed'] else 'FAILED'}"
    )

print()
print(
    "reconstructed/reference baseline "
    f"mismatches: {int(reconstruction_mismatch.sum())}"
)
print("best information-gain K:", best_k)
print("passing K values:", passing)
print()
print("DELTASUB VERDICT:", verdict.upper())
print("result:", result_path)
