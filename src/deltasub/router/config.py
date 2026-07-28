from __future__ import annotations

from pathlib import Path
import yaml

TOP_KEYS = {
    "schema_version", "test_only", "gain_cache_path", "expected_cache_id",
    "feature_cache_path", "splits",
    "router", "loss", "training", "sampling", "replay", "checkpoint_directory",
    "validation_metric", "seed",
}
SPLIT_KEYS = {"train_fraction", "validation_fraction", "test_fraction", "seed"}
ROUTER_KEYS = {
    "input_dim", "hidden_dim", "depth", "normalization", "dropout",
    "coordinate_features", "global_context_features", "learned_position_embedding",
}
LOSS_KEYS = {
    "regression", "huber_delta", "regression_weight", "ranking_weight",
    "ranking_margin", "maximum_pairs_per_anchor", "sign_weight", "sign_epsilon",
    "gain_clip",
}
TRAIN_KEYS = {
    "physical_batch_size", "gradient_accumulation", "effective_batch_size", "optimizer",
    "learning_rate", "weight_decay", "scheduler", "epochs", "gradient_clipping",
    "precision", "device",
}
SAMPLING_KEYS = {"informative_fraction"}
REPLAY_KEYS = {"capacity", "ratio", "priority_mode", "near_zero_epsilon"}


def _section(value, name, allowed):
    if not isinstance(value.get(name), dict):
        raise ValueError(f"{name} must be a mapping")
    extra, missing = set(value[name]) - allowed, allowed - set(value[name])
    if extra or missing:
        raise ValueError(f"{name} unknown={sorted(extra)} missing={sorted(missing)}")


def load_router_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != TOP_KEYS:
        raise ValueError(f"router top-level keys mismatch: {sorted(set(value or {}) ^ TOP_KEYS)}")
    if value["schema_version"] != 1:
        raise ValueError("unsupported M5 configuration schema")
    for name, keys in (("splits", SPLIT_KEYS), ("router", ROUTER_KEYS), ("loss", LOSS_KEYS),
                       ("training", TRAIN_KEYS), ("sampling", SAMPLING_KEYS), ("replay", REPLAY_KEYS)):
        _section(value, name, keys)
    splits, training, loss, replay = value["splits"], value["training"], value["loss"], value["replay"]
    fractions = [splits[x] for x in ("train_fraction", "validation_fraction", "test_fraction")]
    if any(x < 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-12 or splits["validation_fraction"] <= 0:
        raise ValueError("invalid split fractions or empty validation split")
    if training["device"] not in {"cpu", "cuda:0"} or (not value["test_only"] and training["device"] != "cuda:0"):
        raise ValueError("unsupported or non-explicit production device")
    if training["precision"] not in {"fp32", "bf16"} or (
        training["precision"] == "bf16" and training["device"] == "cpu"
    ):
        raise ValueError("unsupported precision/device combination")
    if training["physical_batch_size"] * training["gradient_accumulation"] != training["effective_batch_size"]:
        raise ValueError("effective batch size mismatch")
    if training["optimizer"] not in {"adamw", "sgd"} or training["scheduler"] not in {"none", "cosine"}:
        raise ValueError("unsupported optimizer/scheduler")
    weights = [loss[x] for x in ("regression_weight", "ranking_weight", "sign_weight")]
    if any(x < 0 for x in weights) or not any(weights):
        raise ValueError("invalid loss weights")
    if loss["regression"] not in {"huber", "smooth_l1"} or loss["ranking_margin"] < 0:
        raise ValueError("invalid router loss")
    if replay["capacity"] < training["physical_batch_size"] or not 0 <= replay["ratio"] < 1:
        raise ValueError("invalid replay capacity/ratio")
    if value["test_only"] != (value["router"]["input_dim"] != 768):
        raise ValueError("fixture/production router dimension guard failed")
    if not value["test_only"] and (not value["gain_cache_path"] or len(value["expected_cache_id"]) != 64):
        raise ValueError("missing production cache provenance")
    return value
