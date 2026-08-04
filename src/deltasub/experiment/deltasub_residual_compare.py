"""Matched seed-0 comparison for practical DeltaSub residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.gcd_v2 import (
    evaluate_gcd_v2,
)


def hmean(
    old: float,
    new: float,
) -> float:
    return (
        2.0 * old * new / (old + new)
        if old + new > 0
        else 0.0
    )


def metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    old: np.ndarray,
) -> dict[str, float]:
    value = evaluate_gcd_v2(
        targets,
        predictions,
        old,
    ).as_dict()

    return {
        "all": float(value["all"]),
        "old": float(value["old"]),
        "new": float(value["new"]),
        "hmean": hmean(
            float(value["old"]),
            float(value["new"]),
        ),
    }


def locate_matched_baseline() -> tuple[
    Path,
    dict[str, Any],
]:
    candidates: list[
        tuple[tuple[int, int, float], Path, dict[str, Any]]
    ] = []

    for path in Path("artifacts").rglob(
        "result.json"
    ):
        if "supervised" in path.parts:
            continue

        try:
            value = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if value.get("method") != "selex":
            continue
        if value.get("dataset") != "cub":
            continue
        if int(value.get("seed", -1)) != 0:
            continue
        if value.get(
            "supervision_mode",
            "gcd",
        ) != "gcd":
            continue
        if not (
            path.parent
            / "predictions.npz"
        ).is_file():
            continue

        matched_parameters = int(
            value.get(
                "trainable_parameters",
                0,
            )
            == 14_334_152
        )
        in_deltasub_tree = int(
            "deltasub_v2"
            in path.parts
        )

        candidates.append(
            (
                (
                    matched_parameters,
                    in_deltasub_tree,
                    path.stat().st_mtime,
                ),
                path,
                value,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "matched CUB seed-0 SelEx result "
            "and predictions were not found"
        )

    _, path, value = max(
        candidates,
        key=lambda item: item[0],
    )

    return path, value


def bootstrap_hmean_delta(
    *,
    targets: np.ndarray,
    old: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float]:
    generator = np.random.default_rng(
        seed
    )
    count = len(targets)
    values: list[float] = []

    for _ in range(draws):
        indices = generator.integers(
            0,
            count,
            size=count,
        )

        sampled_old = old[indices]

        if (
            not bool(sampled_old.any())
            or bool(sampled_old.all())
        ):
            continue

        baseline_metrics = metrics(
            targets[indices],
            baseline[indices],
            sampled_old,
        )
        candidate_metrics = metrics(
            targets[indices],
            candidate[indices],
            sampled_old,
        )

        values.append(
            candidate_metrics["hmean"]
            - baseline_metrics["hmean"]
        )

    if not values:
        raise RuntimeError(
            "paired bootstrap produced no draws"
        )

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "draws": int(array.size),
        "mean": float(array.mean()),
        "lower_95": float(
            np.quantile(array, 0.025)
        ),
        "upper_95": float(
            np.quantile(array, 0.975)
        ),
    }


def compare(
    *,
    candidate_result_path: Path,
    output_path: Path,
    bootstrap_draws: int,
) -> dict[str, Any]:
    candidate_result = json.loads(
        candidate_result_path.read_text(
            encoding="utf-8"
        )
    )

    candidate_prediction_path = (
        candidate_result_path.parent
        / "predictions.npz"
    )

    if not candidate_prediction_path.is_file():
        raise FileNotFoundError(
            candidate_prediction_path
        )

    baseline_result_path, baseline_result = (
        locate_matched_baseline()
    )

    baseline_prediction_path = (
        baseline_result_path.parent
        / "predictions.npz"
    )

    candidate_archive = np.load(
        candidate_prediction_path,
        allow_pickle=False,
    )
    baseline_archive = np.load(
        baseline_prediction_path,
        allow_pickle=False,
    )

    targets = candidate_archive[
        "target"
    ].astype(np.int64)
    old = candidate_archive[
        "old"
    ].astype(bool)
    candidate_predictions = candidate_archive[
        "fused_prediction"
    ].astype(np.int64)

    baseline_targets = baseline_archive[
        "target"
    ].astype(np.int64)
    baseline_old = baseline_archive[
        "old"
    ].astype(bool)
    baseline_predictions = baseline_archive[
        "fused_prediction"
    ].astype(np.int64)

    if not np.array_equal(
        targets,
        baseline_targets,
    ):
        raise RuntimeError(
            "candidate and baseline targets differ"
        )
    if not np.array_equal(
        old,
        baseline_old,
    ):
        raise RuntimeError(
            "candidate and baseline partitions differ"
        )

    baseline_metrics = {
        name: float(
            baseline_result["metrics"][name]
        )
        for name in (
            "all",
            "old",
            "new",
            "hmean",
        )
    }
    candidate_metrics = {
        name: float(
            candidate_result["metrics"][name]
        )
        for name in (
            "all",
            "old",
            "new",
            "hmean",
        )
    }

    delta = {
        name: (
            candidate_metrics[name]
            - baseline_metrics[name]
        )
        for name in (
            "all",
            "old",
            "new",
            "hmean",
        )
    }

    interval = bootstrap_hmean_delta(
        targets=targets,
        old=old,
        baseline=baseline_predictions,
        candidate=candidate_predictions,
        draws=bootstrap_draws,
        seed=41_003,
    )

    passed = (
        delta["all"] >= 0.0
        and delta["new"] >= 0.005
        and delta["hmean"] >= 0.005
        and interval["lower_95"] > 0.0
    )

    result = {
        "schema_version": (
            "deltasub-residual.gate.v1"
        ),
        "status": "completed",
        "baseline_result": str(
            baseline_result_path
        ),
        "candidate_result": str(
            candidate_result_path
        ),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "hmean_paired_bootstrap": interval,
        "gate": (
            "passed"
            if passed
            else "failed"
        ),
        "gate_rule": {
            "all_delta_minimum": 0.0,
            "new_delta_minimum": 0.005,
            "hmean_delta_minimum": 0.005,
            "hmean_lower_95_above_zero": True,
        },
        "oracle_used": False,
        "patch_selection_used": False,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "method                    "
        "all       old       new      hmean"
    )

    for name, values in (
        ("matched_selex", baseline_metrics),
        (
            "deltasub_residual",
            candidate_metrics,
        ),
    ):
        print(
            f"{name:<25}"
            f"{values['all']:>9.6f}"
            f"{values['old']:>10.6f}"
            f"{values['new']:>10.6f}"
            f"{values['hmean']:>11.6f}"
        )

    print()
    print("delta versus matched SelEx")

    for name in (
        "all",
        "old",
        "new",
        "hmean",
    ):
        print(
            f"{name:<5}: "
            f"{delta[name]:+.6f}"
        )

    print()
    print(
        "hmean paired-bootstrap 95% CI: "
        f"[{interval['lower_95']:+.6f}, "
        f"{interval['upper_95']:+.6f}]"
    )
    print(
        "practical residual gate: "
        + (
            "PASSED"
            if passed
            else "FAILED"
        )
    )
    print(
        f"comparison: {output_path}"
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-result",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    compare(
        candidate_result_path=Path(
            args.candidate_result
        ),
        output_path=Path(args.output),
        bootstrap_draws=int(
            args.bootstrap_draws
        ),
    )


if __name__ == "__main__":
    main()
