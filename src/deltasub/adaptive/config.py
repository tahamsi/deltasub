from __future__ import annotations

from pathlib import Path
import yaml

TOP_KEYS = {
    "schema_version", "mode", "m2_checkpoint", "m5_router_checkpoint",
    "dinov2", "child_projector_provenance", "execution", "budget", "controller",
    "training", "loss_weights", "checkpoint_directory", "validation_metric", "seed",
}
EXECUTION_KEYS = {"mode", "bucket_policy", "bucket_boundaries"}
BUDGET_KEYS = {"mode", "unit", "target", "fixed_k", "per_sample_k", "min_k", "max_k"}
CONTROLLER_KEYS = {"mode", "score_threshold", "initial_lambda", "dual_learning_rate",
                   "lambda_min", "lambda_max", "dual_update_interval"}
TRAINING_KEYS = {"precision", "device", "physical_batch_size", "gradient_accumulation",
                 "optimizer", "scheduler", "learning_rate", "weight_decay",
                 "gradient_clipping", "epochs", "trainable_components"}
CHECKPOINT_KEYS = {"path", "expected_sha256"}
DINOV2_KEYS = {"source_root", "expected_revision", "checkpoint_sha256"}


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} keys mismatch: {sorted(set(value or {}) ^ keys)}")


def load_adaptive_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _exact(value, TOP_KEYS, "adaptive top-level")
    if value["schema_version"] != 1 or value["mode"] not in {"fixture", "production"}:
        raise ValueError("unsupported M6 schema or mode")
    for key in ("m2_checkpoint", "m5_router_checkpoint"):
        _exact(value[key], CHECKPOINT_KEYS, key)
    _exact(value["dinov2"], DINOV2_KEYS, "dinov2")
    _exact(value["execution"], EXECUTION_KEYS, "execution")
    _exact(value["budget"], BUDGET_KEYS, "budget")
    _exact(value["controller"], CONTROLLER_KEYS, "controller")
    _exact(value["training"], TRAINING_KEYS, "training")
    if value["execution"]["mode"] not in {"padded", "bucketed"}:
        raise ValueError("unsupported execution mode")
    budget = value["budget"]
    if not 0 <= budget["min_k"] <= budget["max_k"] <= 256:
        raise ValueError("invalid K bounds")
    if budget["fixed_k"] is not None and not budget["min_k"] <= budget["fixed_k"] <= budget["max_k"]:
        raise ValueError("fixed K is outside bounds")
    controller = value["controller"]
    if controller["dual_learning_rate"] < 0 or controller["dual_update_interval"] < 1:
        raise ValueError("invalid dual update settings")
    if not controller["lambda_min"] <= controller["initial_lambda"] <= controller["lambda_max"]:
        raise ValueError("invalid lambda bounds")
    training = value["training"]
    if training["precision"] not in {"fp32", "bf16"} or training["device"] not in {"cpu", "cuda:0"}:
        raise ValueError("unsupported precision or device")
    if value["mode"] == "production":
        if training["device"] != "cuda:0":
            raise ValueError("production device must explicitly be cuda:0")
        for key in ("m2_checkpoint", "m5_router_checkpoint"):
            if len(value[key]["expected_sha256"]) != 64 or not value[key]["path"]:
                raise ValueError("missing production checkpoint provenance")
    if not any(float(x) > 0 for x in value["loss_weights"].values()):
        raise ValueError("all training loss weights are zero")
    if set(value["loss_weights"]) != {"classification", "selex"}:
        raise ValueError("loss weight keys mismatch")
    return value
