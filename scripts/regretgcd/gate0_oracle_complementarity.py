#!/usr/bin/env python
"""Diagnostic upper bound for RegretGCD expert arbitration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from deltasub.evaluation.gcd_v2 import evaluate_gcd_v2
from deltasub.experiment.deltasub_residual_compare import (
    bootstrap_hmean_delta,
)
from deltasub.experiment.training import _atomic_json
from deltasub.utils.hashing import sha256_file


SCHEMA = "regretgcd.oracle-complementarity.v1"

DEFAULT_ARCHIVE = Path(
    "artifacts/deltasub_prototype_conditional/"
    "cub/seed_0/predictions.npz"
)
DEFAULT_SOURCE_RESULT = Path(
    "artifacts/deltasub_prototype_conditional/"
    "cub/seed_0/result.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/regretgcd/cub/seed_0/"
    "gate0_result.json"
)

GATE = {
    "all_delta_minimum": 0.010,
    "old_delta_minimum": 0.030,
    "new_delta_minimum": 0.000,
    "hmean_delta_minimum": 0.020,
}


def score(target, prediction, old):
    raw = evaluate_gcd_v2(
        target,
        prediction,
        old,
    ).as_dict()

    old_value = float(raw["old"])
    new_value = float(raw["new"])

    return {
        "all": float(raw["all"]),
        "old": old_value,
        "new": new_value,
        "hmean": (
            2.0 * old_value * new_value
            / (old_value + new_value)
            if old_value + new_value > 0
            else 0.0
        ),
    }


def canonicalize(target, prediction):
    """Oracle-map predicted cluster IDs into true-label space."""

    dimension = int(
        max(target.max(), prediction.max())
    ) + 1

    contingency = np.zeros(
        (dimension, dimension),
        dtype=np.int64,
    )
    np.add.at(
        contingency,
        (prediction, target),
        1,
    )

    rows, columns = linear_sum_assignment(
        contingency.max() - contingency
    )

    mapping = {
        int(row): int(column)
        for row, column in zip(rows, columns)
    }

    canonical = np.asarray(
        [mapping[int(value)] for value in prediction],
        dtype=np.int64,
    )

    return canonical, canonical == target


def subset_fraction(mask, subset):
    selected = mask[subset]

    return (
        float(selected.mean())
        if selected.size
        else 0.0
    )


def verify_source(source, computed):
    for method in (
        "matched_selex",
        "prototype_control",
    ):
        if method not in source["metrics"]:
            raise KeyError(
                f"source result lacks {method}"
            )

        for key in (
            "all",
            "old",
            "new",
            "hmean",
        ):
            expected = float(
                source["metrics"][method][key]
            )
            actual = float(
                computed[method][key]
            )

            if abs(expected - actual) > 1e-12:
                raise RuntimeError(
                    "source/archive metric mismatch: "
                    f"{method}.{key}: "
                    f"{expected} != {actual}"
                )


def run(args):
    archive_path = Path(args.archive)
    source_path = Path(args.source_result)
    output_path = Path(args.output)

    for path in (
        archive_path,
        source_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    archive = np.load(
        archive_path,
        allow_pickle=False,
    )

    required = {
        "target",
        "old",
        "matched_selex_prediction",
        "prototype_control_prediction",
    }
    missing = required - set(archive.files)

    if missing:
        raise KeyError(
            f"missing keys {sorted(missing)}; "
            f"available={archive.files}"
        )

    target = archive["target"].astype(np.int64)
    old = archive["old"].astype(bool)
    parametric = archive[
        "matched_selex_prediction"
    ].astype(np.int64)
    prototype = archive[
        "prototype_control_prediction"
    ].astype(np.int64)

    arrays = (
        target,
        old,
        parametric,
        prototype,
    )

    if any(value.ndim != 1 for value in arrays):
        raise ValueError(
            "all arrays must be one-dimensional"
        )

    if len({value.size for value in arrays}) != 1:
        raise ValueError(
            "prediction array lengths differ"
        )

    parametric_canonical, parametric_correct = (
        canonicalize(target, parametric)
    )
    prototype_canonical, prototype_correct = (
        canonicalize(target, prototype)
    )

    both_correct = (
        parametric_correct
        & prototype_correct
    )
    parametric_only = (
        parametric_correct
        & ~prototype_correct
    )
    prototype_only = (
        prototype_correct
        & ~parametric_correct
    )
    both_wrong = (
        ~parametric_correct
        & ~prototype_correct
    )
    disagreement = (
        parametric_canonical
        != prototype_canonical
    )

    # Keep the prototype decision except when the parametric
    # expert is uniquely correct. Labels are used here only to
    # measure the maximum possible arbitration opportunity.
    oracle = prototype_canonical.copy()
    oracle[parametric_only] = (
        parametric_canonical[parametric_only]
    )

    computed = {
        "matched_selex": score(
            target,
            parametric,
            old,
        ),
        "prototype_control": score(
            target,
            prototype,
            old,
        ),
        "oracle_arbitration": score(
            target,
            oracle,
            old,
        ),
    }

    source = json.loads(
        source_path.read_text(
            encoding="utf-8"
        )
    )
    verify_source(source, computed)

    reference = computed[
        "prototype_control"
    ]
    candidate = computed[
        "oracle_arbitration"
    ]

    delta = {
        key: candidate[key] - reference[key]
        for key in (
            "all",
            "old",
            "new",
            "hmean",
        )
    }

    interval = bootstrap_hmean_delta(
        targets=target,
        old=old,
        baseline=prototype_canonical,
        candidate=oracle,
        draws=int(args.bootstrap_draws),
        seed=92_711,
    )

    passed = (
        delta["all"]
        >= GATE["all_delta_minimum"]
        and delta["old"]
        >= GATE["old_delta_minimum"]
        and delta["new"]
        >= GATE["new_delta_minimum"]
        and delta["hmean"]
        >= GATE["hmean_delta_minimum"]
    )

    verdict = (
        "continue"
        if passed
        else "abandon"
    )

    subsets = {
        "all": np.ones_like(old, dtype=bool),
        "old": old,
        "new": ~old,
    }

    fractions = {}

    for name, subset in subsets.items():
        fractions[name] = {
            "both_correct": subset_fraction(
                both_correct,
                subset,
            ),
            "parametric_only_correct": (
                subset_fraction(
                    parametric_only,
                    subset,
                )
            ),
            "prototype_only_correct": (
                subset_fraction(
                    prototype_only,
                    subset,
                )
            ),
            "both_wrong": subset_fraction(
                both_wrong,
                subset,
            ),
            "canonical_disagreement": (
                subset_fraction(
                    disagreement,
                    subset,
                )
            ),
        }

    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "dataset": "cub",
        "seed": 0,
        "stage": "gate_0_oracle_complementarity",
        "diagnostic_oracle_used": True,
        "deployable_method_claimed": False,
        "test_labels_used": True,
        "metrics": computed,
        "delta_vs_prototype_control": delta,
        "hmean_paired_bootstrap": interval,
        "complementarity": {
            "counts": {
                "samples": int(target.size),
                "both_correct": int(
                    both_correct.sum()
                ),
                "parametric_only_correct": int(
                    parametric_only.sum()
                ),
                "prototype_only_correct": int(
                    prototype_only.sum()
                ),
                "both_wrong": int(
                    both_wrong.sum()
                ),
                "canonical_disagreement": int(
                    disagreement.sum()
                ),
            },
            "fractions": fractions,
        },
        "gate_rule": GATE,
        "gate": (
            "passed"
            if passed
            else "failed"
        ),
        "verdict": verdict,
        "next_stage_if_continue": {
            "method": (
                "class-holdout learned regret router"
            ),
            "required_ablations": [
                "maximum-confidence routing",
                "minimum-entropy routing",
                "knownness-only routing",
                "global score blending",
                "random matched-rate switching",
                "instance-holdout router",
                "without stability features",
                "without prototype-density features",
            ],
            "confirmation": [
                "locked CUB seed 0 gate",
                "three CUB seeds",
                "Aircraft after CUB survival",
                "paired bootstrap",
                "current SOTA comparison",
            ],
        },
        "source": {
            "archive": str(archive_path),
            "archive_sha256": sha256_file(
                archive_path
            ),
            "result": str(source_path),
            "result_sha256": sha256_file(
                source_path
            ),
        },
    }

    _atomic_json(
        output_path,
        result,
    )

    print(
        "method                    "
        "all       old       new      hmean"
    )

    for name in (
        "matched_selex",
        "prototype_control",
        "oracle_arbitration",
    ):
        value = computed[name]

        print(
            f"{name:<25}"
            f"{value['all']:>9.6f}"
            f"{value['old']:>10.6f}"
            f"{value['new']:>10.6f}"
            f"{value['hmean']:>11.6f}"
        )

    print()
    print(
        "oracle delta versus prototype control"
    )

    for key in (
        "all",
        "old",
        "new",
        "hmean",
    ):
        print(
            f"{key:<5}: "
            f"{delta[key]:+.6f}"
        )

    print(
        "hmean paired-bootstrap 95% CI: "
        f"[{interval['lower_95']:+.6f}, "
        f"{interval['upper_95']:+.6f}]"
    )

    counts = result[
        "complementarity"
    ]["counts"]

    print()
    print(
        "parametric-only correct:",
        counts["parametric_only_correct"],
    )
    print(
        "prototype-only correct:",
        counts["prototype_only_correct"],
    )
    print(
        "both wrong:",
        counts["both_wrong"],
    )
    print(
        "canonical disagreements:",
        counts["canonical_disagreement"],
    )

    print()
    print(
        "REGRETGCD GATE-0 VERDICT:",
        verdict.upper(),
    )
    print("result:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        default=str(DEFAULT_ARCHIVE),
    )
    parser.add_argument(
        "--source-result",
        default=str(DEFAULT_SOURCE_RESULT),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=5000,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
