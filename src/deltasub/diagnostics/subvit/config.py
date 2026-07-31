from __future__ import annotations

from pathlib import Path
import yaml

TOP = {"schema_version", "mode", "seed", "model", "checkpoint", "source",
       "attention", "ats", "degradation", "router", "loss", "training", "output"}


def load_subvit_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != TOP:
        raise ValueError(f"SubViT config keys mismatch: {sorted(set(value or {}) ^ TOP)}")
    if value["schema_version"] != 1 or value["mode"] not in {"fixture", "production"}:
        raise ValueError("unsupported M7 schema or mode")
    if value["attention"]["parent_count"] != 256 or value["ats"]["factor"] < 1:
        raise ValueError("M7 requires 256 parents and positive subdivision factor")
    if value["training"]["precision"] not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if value["training"]["device"] not in {"cpu", "cuda:0"}:
        raise ValueError("device must be cpu or cuda:0")
    if value["mode"] == "production":
        if value["training"]["device"] != "cuda:0":
            raise ValueError("production requires cuda:0")
        if not value["checkpoint"]["path"] or len(value["checkpoint"]["sha256"]) != 64:
            raise ValueError("production requires explicit checkpoint provenance")
        if not value["source"]["root"] or len(value["source"]["revision"]) != 40:
            raise ValueError("production requires explicit source provenance")
        if value["model"].get("fixture", False):
            raise ValueError("fixture components are forbidden in production")
    elif not value["model"].get("fixture", False):
        raise ValueError("fixture mode requires explicit fixture model")
    return value
