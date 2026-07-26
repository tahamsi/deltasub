from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "dataset", "seed", "all", "known", "novel"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")
    grouped = frame.groupby(["method", "dataset"], sort=True)
    rows = []
    for (method, dataset), group in grouped:
        if group["seed"].nunique() != len(group):
            raise ValueError("duplicate seed records")
        row = {"method": method, "dataset": dataset, "seeds": len(group)}
        for metric in ("all", "known", "novel"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def format_best_second(values: list[float]) -> list[str]:
    unique = sorted(set(values), reverse=True)
    best = unique[0]
    second = unique[1] if len(unique) > 1 else None
    result = []
    for value in values:
        text = f"{value:.2f}"
        if value == best:
            text = f"**{text}**"
        elif second is not None and value == second:
            text = f"<u>{text}</u>"
        result.append(text)
    return result


def write_formats(frame: pd.DataFrame, stem: str | Path) -> list[Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(ext) for ext in (".csv", ".md", ".tex")]
    frame.to_csv(outputs[0], index=False)
    outputs[1].write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    outputs[2].write_text(frame.to_latex(index=False), encoding="utf-8")
    return outputs
