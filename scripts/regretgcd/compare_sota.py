#!/usr/bin/env python
"""Generate an honest RegretGCD versus SOTA reference report.

The report deliberately separates matched in-repository results from paper-reported
numbers.  Humans have repeatedly discovered that putting incompatible percentages in
one table causes reviewers to develop perfectly reasonable trust issues.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def hmean(old: float, new: float) -> float:
    return 2.0 * old * new / (old + new) if old + new > 0 else 0.0


def fmt(value: float) -> str:
    return f"{value:.2f}"


def run(args: argparse.Namespace) -> None:
    result_path = Path(args.result)
    registry_path = Path(args.registry)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != "regretgcd.gate1-result.v1":
        raise ValueError("result is not a completed RegretGCD Gate-1 artifact")
    if registry.get("schema_version") != "regretgcd.sota-registry.v1":
        raise ValueError("unsupported SOTA registry")

    dataset = str(result["dataset"])
    internal_rows: list[dict[str, Any]] = []
    for method, metric in result["metrics"].items():
        internal_rows.append(
            {
                "method": method,
                "all": 100.0 * float(metric["all"]),
                "old": 100.0 * float(metric["old"]),
                "new": 100.0 * float(metric["new"]),
                "hmean": 100.0 * float(metric["hmean"]),
                "comparison_status": "matched_in_repository",
            }
        )

    reference_rows: list[dict[str, Any]] = []
    for entry in registry["entries"]:
        metric = entry.get("metrics", {}).get(dataset)
        if metric is None:
            continue
        reference_rows.append(
            {
                "method": entry["method"],
                "variant": entry.get("variant", ""),
                "year": entry["year"],
                "venue": entry["venue"],
                "backbone": entry["backbone"],
                "all": float(metric["all"]),
                "old": float(metric["old"]),
                "new": float(metric["new"]),
                "hmean": hmean(float(metric["old"]), float(metric["new"])),
                "comparison_status": entry["status"],
                "source": entry["source"],
            }
        )

    with (output / "internal_matched.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(internal_rows[0]))
        writer.writeheader()
        writer.writerows(internal_rows)
    with (output / "published_references.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)

    lines = [
        "# RegretGCD comparison report",
        "",
        f"Dataset: **{dataset.upper()}**  ",
        f"Gate-1 verdict: **{result['verdict'].upper()}**",
        "",
        "## Matched in-repository comparison",
        "",
        "These methods consume the same cached samples and are evaluated by the same GCD-v2 implementation.",
        "",
        "| Method | All | Old | New | H-mean |",
        "|---|---:|---:|---:|---:|",
    ]
    preferred = [
        "matched_selex",
        "prototype_control",
        "regretgcd",
        "maximum_confidence",
        "minimum_entropy",
        "global_probability_blend",
        "random_matched_switch_rate",
        "instance_holdout_router",
        "without_stability",
        "without_density",
        "without_knownness",
        "without_cross_expert",
        "oracle_arbitration_diagnostic",
    ]
    indexed = {row["method"]: row for row in internal_rows}
    for method in preferred:
        if method not in indexed:
            continue
        row = indexed[method]
        lines.append(
            f"| {method} | {fmt(row['all'])} | {fmt(row['old'])} | "
            f"{fmt(row['new'])} | {fmt(row['hmean'])} |"
        )

    lines.extend(
        [
            "",
            "## Paper-reported SOTA references",
            "",
            "**These values are not direct comparisons.** They remain reference-only until checkpoint, split, "
            "augmentation, optimization, class-count assumptions, and evaluator are matched.",
            "",
            "| Method | Variant | Venue | Backbone | All | Old | New | H-mean | Status |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in reference_rows:
        lines.append(
            f"| {row['method']} | {row['variant']} | {row['venue']} {row['year']} | "
            f"{row['backbone']} | {fmt(row['all'])} | {fmt(row['old'])} | "
            f"{fmt(row['new'])} | {fmt(row['hmean'])} | {row['comparison_status']} |"
        )

    lines.extend(
        [
            "",
            "## Protocol decision",
            "",
            registry["protocol_note"],
            "",
            "The current repository baseline must first reproduce a published matched baseline within a "
            "declared tolerance before any claim such as ‘beats SOTA’ is permitted. The report generator "
            "therefore emits no cross-protocol deltas or rankings.",
            "",
            "## Other current methods checked",
            "",
        ]
    )
    for entry in registry.get("related_current_methods", []):
        lines.append(
            f"- **{entry['method']}** ({entry['venue']} {entry['year']}): {entry['mechanism']}."
        )
    lines.append("")
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("comparison:", output / "comparison.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default="artifacts/regretgcd/cub/seed_0/result.json",
    )
    parser.add_argument(
        "--registry",
        default="configs/regretgcd/sota_registry.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/regretgcd/cub/seed_0/comparison",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
