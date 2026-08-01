#!/usr/bin/env python3
"""Compare learned router selection with the complete M4 gain oracle.

This is an additive single-candidate-gain proxy. It does not claim that the
sum of independently measured gains equals the joint K-token intervention.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from deltasub.adaptive.selection import deterministic_select
from deltasub.router.features import load_feature_cache
from deltasub.router.model import GainRouter, RouterConfig
from deltasub.utils.checkpointing import load_checkpoint
from deltasub.utils.hashing import sha256_file


KEY_COLUMNS = ["batch_context_sha256", "sample_id"]
REQUIRED_GAIN_COLUMNS = {
    *KEY_COLUMNS,
    "candidate_parent_index",
    "gain",
    "anchor_valid",
    "labelled",
    "known_or_novel",
}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_gain_frame(root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    files: list[Path] = []

    for path in sorted(root.rglob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue

        if REQUIRED_GAIN_COLUMNS.issubset(frame.columns):
            frames.append(frame)
            files.append(path)

    if not frames:
        raise FileNotFoundError(
            f"no production gain Parquet shards found below {root}"
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined, files


def scalar_summary(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("summary values must be finite and one-dimensional")

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "positive_fraction": float((values > 0).mean()),
        "negative_fraction": float((values < 0).mean()),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
    }


def subset_summary(
    values: np.ndarray,
    metadata: pd.DataFrame,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = values[mask]
    return {
        "sample_count": int(mask.sum()),
        **scalar_summary(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/diagnostic/m9/cub/seed_0"),
    )
    parser.add_argument(
        "--diagnostic-result",
        type=Path,
        default=Path("artifacts/diagnostic/m9/cub/diagnostic.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostic/m9/cub/oracle_gain_probe"),
    )
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()

    if not 1 <= args.k <= 256:
        raise ValueError("K must be in [1, 256]")

    delta_root = args.run_root / "deltasub"
    gains_root = delta_root / "stages" / "gains"
    router_root = delta_root / "stages" / "router"

    diagnostic = json.loads(
        args.diagnostic_result.read_text(encoding="utf-8")
    )
    if diagnostic.get("status") != "completed":
        raise ValueError("corrected CUB diagnostic is not completed")

    config_path = delta_root / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    gains, gain_files = load_gain_frame(gains_root)
    gains = gains[gains["anchor_valid"].astype(bool)].copy()

    if gains.empty:
        raise ValueError("gain cache contains no valid records")

    gains["candidate_parent_index"] = pd.to_numeric(
        gains["candidate_parent_index"], errors="raise"
    ).astype(int)
    gains["gain"] = pd.to_numeric(gains["gain"], errors="raise")

    if not np.isfinite(gains["gain"].to_numpy()).all():
        raise ValueError("gain cache contains non-finite values")

    duplicate_count = int(
        gains.duplicated(KEY_COLUMNS + ["candidate_parent_index"]).sum()
    )
    if duplicate_count:
        raise ValueError(f"duplicate gain keys found: {duplicate_count}")

    counts = gains.groupby(KEY_COLUMNS).size()
    if not bool((counts == 256).all()):
        bad = counts[counts != 256]
        raise ValueError(
            f"{len(bad)} samples do not have complete 256-candidate coverage"
        )

    pivot = gains.pivot(
        index=KEY_COLUMNS,
        columns="candidate_parent_index",
        values="gain",
    ).reindex(columns=range(256))

    if pivot.isna().any().any():
        raise ValueError("gain matrix has missing candidates")

    metadata = (
        gains.groupby(KEY_COLUMNS, sort=True)
        .agg(
            labelled=("labelled", "first"),
            known_or_novel=("known_or_novel", "first"),
        )
        .reindex(pivot.index)
    )

    gain_matrix = torch.tensor(
        pivot.to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    feature_paths = sorted(gains_root.rglob("router_features.pt"))
    if len(feature_paths) != 1:
        raise ValueError(
            f"expected one router feature cache, found {len(feature_paths)}"
        )

    feature_batches = load_feature_cache(feature_paths[0])
    feature_by_key: dict[tuple[str, str], torch.Tensor] = {}

    for batch in feature_batches:
        for index, sample_id in enumerate(batch.sample_ids):
            key = (batch.batch_context_sha256, sample_id)
            if key in feature_by_key:
                raise ValueError(f"duplicate router feature identity: {key}")
            feature_by_key[key] = batch.parent_embeddings[index]

    ordered_features = []
    missing_features = []

    for context, sample_id in pivot.index:
        key = (str(context), str(sample_id))
        feature = feature_by_key.get(key)
        if feature is None:
            missing_features.append(key)
        else:
            ordered_features.append(feature)

    if missing_features:
        raise ValueError(
            f"missing router features for {len(missing_features)} gain samples"
        )

    parent_features = torch.stack(ordered_features).float()
    if parent_features.shape != (len(pivot), 256, 768):
        raise ValueError(
            f"unexpected router feature shape: {tuple(parent_features.shape)}"
        )

    architecture = dict(config["deltasub"]["router_architecture"])
    router = GainRouter(
        RouterConfig(**architecture),
        seed=int(config["seed"]),
    )

    checkpoint_path = router_root / "checkpoint_best.pt"
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    required_checkpoint = {
        "router_state",
        "router_configuration_hash",
        "m4_cache_id",
        "m4_validation_hash",
        "feature_source_hash",
        "best_metric",
        "epoch",
    }
    missing = required_checkpoint - set(checkpoint)
    if missing:
        raise ValueError(
            f"router checkpoint missing fields: {sorted(missing)}"
        )

    if router.configuration_hash != checkpoint["router_configuration_hash"]:
        raise ValueError("router architecture does not match checkpoint")

    router.load_state_dict(checkpoint["router_state"], strict=True)
    router.eval()

    with torch.inference_mode():
        router_scores = router(parent_features)

    k_values = torch.full(
        (len(pivot),),
        args.k,
        dtype=torch.long,
    )

    oracle_selection = deterministic_select(gain_matrix, k_values)
    router_selection = deterministic_select(router_scores, k_values)

    oracle_mask = oracle_selection.selected_mask
    router_mask = router_selection.selected_mask

    oracle_sum = (gain_matrix * oracle_mask).sum(dim=1).numpy()
    router_sum = (gain_matrix * router_mask).sum(dim=1).numpy()

    random_expected_sum = (
        gain_matrix.sum(dim=1) * (args.k / 256.0)
    ).numpy()

    overlap = (
        (oracle_mask & router_mask).sum(dim=1).float() / args.k
    ).numpy()

    regret = oracle_sum - router_sum

    mean_oracle = float(oracle_sum.mean())
    mean_router = float(router_sum.mean())
    mean_random = float(random_expected_sum.mean())

    denominator = mean_oracle - mean_random
    utility_capture = (
        (mean_router - mean_random) / denominator
        if abs(denominator) > 1e-12
        else None
    )

    exact_match_fraction = float(
        (oracle_mask == router_mask).all(dim=1).float().mean()
    )

    known = (
        metadata["known_or_novel"].astype(str).to_numpy() == "known"
    )
    novel = (
        metadata["known_or_novel"].astype(str).to_numpy() == "novel"
    )
    labelled = metadata["labelled"].astype(bool).to_numpy()
    unlabelled = ~labelled

    if mean_oracle <= 0:
        decision = "no_positive_oracle_gain_signal"
        interpretation = (
            "Even the complete top-K oracle ranking has non-positive mean "
            "additive gain. The M4 gain mechanism does not provide a usable "
            "positive upper-bound signal on this diagnostic sample."
        )
    elif utility_capture is not None and utility_capture < 0.25:
        decision = "router_bottleneck_on_m4_gain_ranking"
        interpretation = (
            "The complete gain oracle has positive utility, but the learned "
            "router captures less than 25% of the improvement above random. "
            "The router is the primary bottleneck on the M4 gain objective."
        )
    elif utility_capture is not None and utility_capture >= 0.75:
        decision = "gain_objective_or_detail_mechanism_misalignment"
        interpretation = (
            "The router captures at least 75% of the oracle improvement above "
            "random, yet corrected GCD performance is negative. The likely "
            "problem is mismatch between the gain objective, joint K-token "
            "behavior, and downstream GCD rather than router ranking alone."
        )
    else:
        decision = "mixed_router_and_objective_failure"
        interpretation = (
            "The oracle has positive gain signal, but the router captures only "
            "part of it. Both router quality and objective-to-GCD alignment "
            "remain plausible causes."
        )

    report: dict[str, Any] = {
        "schema_version": "m9.oracle-gain-probe.v1",
        "label": (
            "DIAGNOSTIC ONLY: additive sum of independently measured "
            "single-candidate gains"
        ),
        "run_root": str(args.run_root),
        "k": args.k,
        "sample_count": int(len(pivot)),
        "candidate_count_per_sample": 256,
        "gain_record_count": int(len(gains)),
        "gain_files": [str(path) for path in gain_files],
        "gain_file_sha256": {
            str(path): sha256_file(path) for path in gain_files
        },
        "router_feature_cache": str(feature_paths[0]),
        "router_feature_cache_sha256": sha256_file(feature_paths[0]),
        "router_checkpoint": str(checkpoint_path),
        "router_checkpoint_sha256": sha256_file(checkpoint_path),
        "router_checkpoint_epoch": int(checkpoint["epoch"]),
        "router_checkpoint_best_metric": (
            float(checkpoint["best_metric"])
            if math.isfinite(float(checkpoint["best_metric"]))
            else None
        ),
        "corrected_cub_result": {
            "verdict": diagnostic["verdict"],
            "absolute_delta": float(diagnostic["absolute_delta"]),
            "baseline_gcd_all_v2": float(
                diagnostic["metrics"]["vit_dinov2_selex"]["gcd_all_v2"]
            ),
            "deltasub_gcd_all_v2": float(
                diagnostic["metrics"]["deltasub"]["gcd_all_v2"]
            ),
        },
        "oracle_selected_gain_sum": scalar_summary(oracle_sum),
        "router_selected_gain_sum": scalar_summary(router_sum),
        "random_expected_gain_sum": scalar_summary(random_expected_sum),
        "oracle_regret": scalar_summary(regret),
        "router_oracle_topk_overlap": scalar_summary(overlap),
        "router_oracle_exact_set_match_fraction": exact_match_fraction,
        "aggregate_utility_capture_above_random": utility_capture,
        "subgroups": {
            "known": {
                "oracle": subset_summary(oracle_sum, metadata, known),
                "router": subset_summary(router_sum, metadata, known),
            },
            "novel": {
                "oracle": subset_summary(oracle_sum, metadata, novel),
                "router": subset_summary(router_sum, metadata, novel),
            },
            "labelled": {
                "oracle": subset_summary(oracle_sum, metadata, labelled),
                "router": subset_summary(router_sum, metadata, labelled),
            },
            "unlabelled": {
                "oracle": subset_summary(oracle_sum, metadata, unlabelled),
                "router": subset_summary(router_sum, metadata, unlabelled),
            },
        },
        "decision": decision,
        "interpretation": interpretation,
        "limitations": [
            (
                "Candidate gains were measured independently. Their top-K sum "
                "is an additive oracle proxy, not the exact joint K-token loss."
            ),
            (
                "The probe evaluates the 128 fully measured gain samples, not "
                "the held-out GCD test partition."
            ),
            (
                "This probe diagnoses router ranking versus the M4 gain signal; "
                "it does not itself produce a publication GCD result."
            ),
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "oracle_gain_probe.json", report)

    markdown = [
        "# CUB Oracle Gain Probe",
        "",
        "**Diagnostic only.** This uses the additive sum of independently "
        "measured single-candidate gains.",
        "",
        "## Result",
        "",
        f"- Samples: `{len(pivot)}`",
        f"- Candidates per sample: `256`",
        f"- K: `{args.k}`",
        f"- Mean oracle top-K gain sum: `{mean_oracle:+.8f}`",
        f"- Mean router top-K gain sum: `{mean_router:+.8f}`",
        f"- Mean random expected gain sum: `{mean_random:+.8f}`",
        f"- Mean oracle regret: `{float(regret.mean()):+.8f}`",
        f"- Mean router/oracle top-K overlap: `{float(overlap.mean()):.4%}`",
        f"- Exact selected-set match: `{exact_match_fraction:.4%}`",
        (
            "- Utility captured above random: "
            f"`{utility_capture:.4%}`"
            if utility_capture is not None
            else "- Utility captured above random: `undefined`"
        ),
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        interpretation,
        "",
        "## Corrected CUB GCD result",
        "",
        (
            "- Baseline GCD All: "
            f"`{report['corrected_cub_result']['baseline_gcd_all_v2']:.6f}`"
        ),
        (
            "- DeltaSub GCD All: "
            f"`{report['corrected_cub_result']['deltasub_gcd_all_v2']:.6f}`"
        ),
        (
            "- Absolute delta: "
            f"`{report['corrected_cub_result']['absolute_delta']:+.6f}`"
        ),
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ]

    (args.output / "ORACLE_GAIN_PROBE.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    sample_table = pd.DataFrame({
        "batch_context_sha256": [
            str(index[0]) for index in pivot.index
        ],
        "sample_id": [
            str(index[1]) for index in pivot.index
        ],
        "labelled": metadata["labelled"].astype(bool).to_numpy(),
        "known_or_novel": metadata["known_or_novel"].astype(str).to_numpy(),
        "oracle_topk_gain_sum": oracle_sum,
        "router_topk_gain_sum": router_sum,
        "random_expected_gain_sum": random_expected_sum,
        "oracle_regret": regret,
        "topk_overlap_fraction": overlap,
    })
    sample_table.to_csv(args.output / "sample_results.csv", index=False)

    print("===== ORACLE GAIN PROBE =====")
    print(f"samples: {len(pivot)}")
    print(f"candidate coverage: 256/256")
    print(f"K: {args.k}")
    print(f"mean oracle top-K gain sum: {mean_oracle:+.8f}")
    print(f"mean router top-K gain sum: {mean_router:+.8f}")
    print(f"mean random expected sum: {mean_random:+.8f}")
    print(f"mean oracle regret: {float(regret.mean()):+.8f}")
    print(f"mean top-K overlap: {float(overlap.mean()):.4%}")
    print(f"exact set match: {exact_match_fraction:.4%}")
    if utility_capture is None:
        print("utility captured above random: undefined")
    else:
        print(f"utility captured above random: {utility_capture:.4%}")
    print(f"decision: {decision}")
    print(interpretation)
    print()
    print(f"JSON: {args.output / 'oracle_gain_probe.json'}")
    print(f"Markdown: {args.output / 'ORACLE_GAIN_PROBE.md'}")
    print(f"Samples: {args.output / 'sample_results.csv'}")


if __name__ == "__main__":
    main()
