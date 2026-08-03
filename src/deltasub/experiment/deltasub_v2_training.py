"""Matched training and evaluation for SelEx and DeltaSub v2."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ..data.manifests import read_manifest
from ..evaluation.gcd_v2 import (
    evaluate_gcd_v2,
    provenance as gcd_provenance,
)
from ..losses.selex import selex_loss
from ..models.backbones.dinov2 import DINOv2Adapter
from ..models.deltasub_v2 import (
    DeltaSubV2,
    DeltaSubV2Config,
)
from ..models.matched_selex import MatchedSelEx
from ..utils.checkpointing import (
    atomic_torch_save,
    load_checkpoint,
)
from ..utils.hashing import sha256_file, stable_hash
from ..utils.reproducibility import seed_everything
from .training import (
    ProductionManifestDataset,
    _atomic_json,
    _atomic_text,
    _worker_seed,
)


SCHEMA = "deltasub-v2.experiment.v1"

TOP_KEYS = {
    "schema_version",
    "dataset",
    "backbone",
    "selex",
    "seed",
    "training",
    "method",
    "output_directory",
}

TRAINING_KEYS = {
    "epochs",
    "physical_batch_size",
    "gradient_accumulation",
    "evaluation_batch_size",
    "num_workers",
    "backbone_learning_rate",
    "head_learning_rate",
    "detail_learning_rate",
    "weight_decay",
    "temperature",
    "supervised_weight",
    "classification_weight",
    "selex_weight",
    "global_branch_weight",
    "detail_branch_weight",
    "fused_branch_weight",
    "utility_weight",
    "preservation_weight",
    "budget_weight",
    "utility_temperature",
    "preservation_tolerance",
    "gradient_clipping",
}

METHOD_KEYS = {
    "variant",
    "insertion_block",
    "trainable_blocks",
    "minimum_k",
    "maximum_k",
    "retained_energy_fraction",
    "detail_adapter_hidden_dim",
    "utility_hidden_dim",
    "initial_detail_scale",
    "initial_fusion_weight",
}


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(
        Path(path).read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise TypeError("configuration must be a mapping")
    if set(value) != TOP_KEYS:
        raise ValueError(
            "top-level configuration mismatch: "
            f"{sorted(set(value) ^ TOP_KEYS)}"
        )
    if value["schema_version"] != SCHEMA:
        raise ValueError("unsupported DeltaSub v2 schema")
    if set(value["training"]) != TRAINING_KEYS:
        raise ValueError("training keys mismatch")
    if set(value["method"]) != METHOD_KEYS:
        raise ValueError("method keys mismatch")

    variant = value["method"]["variant"]

    if variant not in {"selex", "deltasub_v2"}:
        raise ValueError(
            "method.variant must be selex or deltasub_v2"
        )

    training = value["training"]
    effective_batch = (
        int(training["physical_batch_size"])
        * int(training["gradient_accumulation"])
    )

    if effective_batch != 128:
        raise ValueError(
            "effective training batch must equal 128"
        )
    if int(training["epochs"]) <= 0:
        raise ValueError("epochs must be positive")
    if int(training["num_workers"]) < 0:
        raise ValueError("num_workers must be nonnegative")

    for key in (
        "backbone_learning_rate",
        "head_learning_rate",
        "detail_learning_rate",
    ):
        if float(training[key]) <= 0:
            raise ValueError(f"{key} must be positive")

    method = value["method"]

    if int(method["trainable_blocks"]) != 2:
        raise ValueError(
            "matched protocol requires final two blocks"
        )

    if variant == "deltasub_v2":
        DeltaSubV2Config(
            feature_dim=768,
            insertion_block=int(method["insertion_block"]),
            trainable_blocks=int(method["trainable_blocks"]),
            minimum_k=int(method["minimum_k"]),
            maximum_k=int(method["maximum_k"]),
            retained_energy_fraction=float(
                method["retained_energy_fraction"]
            ),
            detail_adapter_hidden_dim=int(
                method["detail_adapter_hidden_dim"]
            ),
            utility_hidden_dim=int(
                method["utility_hidden_dim"]
            ),
            initial_detail_scale=float(
                method["initial_detail_scale"]
            ),
            initial_fusion_weight=float(
                method["initial_fusion_weight"]
            ),
        ).validate()

    required_files = (
        value["dataset"]["manifest"],
        value["dataset"]["split_validation"],
        value["backbone"]["checkpoint"],
        value["selex"]["equivalence_report"],
    )

    for filename in required_files:
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)

    return value


def _class_count(config: dict[str, Any]) -> int:
    split = json.loads(
        Path(
            config["dataset"]["split_validation"]
        ).read_text(encoding="utf-8")
    )

    classes = (
        set(split["known_class_ids"])
        | set(split["novel_class_ids"])
    )

    if not classes:
        raise ValueError("class split contains no classes")

    return len(classes)


def _backbone(config: dict[str, Any]) -> DINOv2Adapter:
    value = config["backbone"]

    return DINOv2Adapter.from_official_checkpoint(
        value["checkpoint"],
        value["checkpoint_sha256"],
        source_root=value["source_root"],
        model_name=value["name"],
    )


def construct_model(
    config: dict[str, Any],
    *,
    device: torch.device,
) -> nn.Module:
    method = config["method"]
    backbone = _backbone(config)
    classes = _class_count(config)

    if method["variant"] == "selex":
        model: nn.Module = MatchedSelEx(
            backbone,
            classes,
            trainable_blocks=int(
                method["trainable_blocks"]
            ),
        )
    else:
        model = DeltaSubV2(
            backbone,
            classes,
            DeltaSubV2Config(
                feature_dim=768,
                insertion_block=int(
                    method["insertion_block"]
                ),
                trainable_blocks=int(
                    method["trainable_blocks"]
                ),
                minimum_k=int(method["minimum_k"]),
                maximum_k=int(method["maximum_k"]),
                retained_energy_fraction=float(
                    method["retained_energy_fraction"]
                ),
                detail_adapter_hidden_dim=int(
                    method["detail_adapter_hidden_dim"]
                ),
                utility_hidden_dim=int(
                    method["utility_hidden_dim"]
                ),
                initial_detail_scale=float(
                    method["initial_detail_scale"]
                ),
                initial_fusion_weight=float(
                    method["initial_fusion_weight"]
                ),
            ),
        )

    return model.to(device)


def normalized_entropy(
    probabilities: torch.Tensor,
) -> torch.Tensor:
    classes = probabilities.shape[-1]
    denominator = max(math.log(classes), 1e-12)

    return -(
        probabilities
        * probabilities.clamp_min(1e-8).log()
    ).sum(dim=-1) / denominator


def branch_view_utility(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Confidence and cross-view stability utility."""

    if logits.ndim != 3 or logits.shape[1] != 2:
        raise ValueError("logits must have shape [B, 2, C]")

    probabilities = F.softmax(logits.float(), dim=-1)
    midpoint = probabilities.mean(dim=1)

    confidence = midpoint.amax(dim=-1)
    entropy = normalized_entropy(midpoint)

    divergence = 0.5 * (
        (
            probabilities[:, 0]
            * (
                probabilities[:, 0].clamp_min(1e-8).log()
                - midpoint.clamp_min(1e-8).log()
            )
        ).sum(dim=-1)
        + (
            probabilities[:, 1]
            * (
                probabilities[:, 1].clamp_min(1e-8).log()
                - midpoint.clamp_min(1e-8).log()
            )
        ).sum(dim=-1)
    )

    return confidence - entropy - divergence


def branch_neighborhood_margin(
    features: torch.Tensor,
) -> torch.Tensor:
    """Cross-view positive similarity minus hardest batch negative."""

    if features.ndim != 3 or features.shape[1] != 2:
        raise ValueError(
            "features must have shape [B, 2, D]"
        )

    first = F.normalize(features[:, 0].float(), dim=-1)
    second = F.normalize(features[:, 1].float(), dim=-1)

    similarity = first @ second.transpose(0, 1)
    positive = similarity.diagonal()

    if features.shape[0] == 1:
        return positive

    mask = torch.eye(
        features.shape[0],
        dtype=torch.bool,
        device=features.device,
    )

    hardest_negative = similarity.masked_fill(
        mask,
        float("-inf"),
    ).amax(dim=1)

    return positive - hardest_negative


def utility_target(
    *,
    global_logits: torch.Tensor,
    detail_logits: torch.Tensor,
    targets: torch.Tensor,
    labelled: torch.Tensor,
    temperature: float,
    global_features: torch.Tensor | None = None,
    detail_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate whether Haar refinement improves each sample."""

    if temperature <= 0:
        raise ValueError("utility temperature must be positive")
    if global_logits.shape != detail_logits.shape:
        raise ValueError("branch logits must have identical shapes")
    if targets.shape != labelled.shape:
        raise ValueError("targets and labelled must have shape [B]")

    if (
        global_features is not None
        and detail_features is not None
    ):
        if global_features.shape != detail_features.shape:
            raise ValueError(
                "branch features must have identical shapes"
            )

        global_utility = branch_neighborhood_margin(
            global_features
        )
        detail_utility = branch_neighborhood_margin(
            detail_features
        )
    else:
        global_utility = branch_view_utility(global_logits)
        detail_utility = branch_view_utility(detail_logits)

    # Autocast may produce BF16 neighborhood utilities. Keep the
    # calibration target in FP32 so labelled CE targets can be inserted.
    target = torch.sigmoid(
        (detail_utility - global_utility) / temperature
    ).float()

    if bool(labelled.any()):
        labelled_targets = targets[labelled]

        global_ce = F.cross_entropy(
            global_logits[:, 0][labelled].float(),
            labelled_targets,
            reduction="none",
        )
        detail_ce = F.cross_entropy(
            detail_logits[:, 0][labelled].float(),
            labelled_targets,
            reduction="none",
        )

        target = target.clone()
        target[labelled] = torch.sigmoid(
            (global_ce - detail_ce) / temperature
        )

    return target.detach()

def preservation_penalty(
    *,
    fused_logits: torch.Tensor,
    global_logits: torch.Tensor,
    targets: torch.Tensor,
    labelled: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Penalize fused labelled loss only when worse than the global path."""

    if tolerance < 0:
        raise ValueError(
            "preservation tolerance must be nonnegative"
        )

    if not bool(labelled.any()):
        return fused_logits.float().sum() * 0.0

    fused_ce = F.cross_entropy(
        fused_logits[:, 0][labelled].float(),
        targets[labelled],
        reduction="none",
    )
    global_ce = F.cross_entropy(
        global_logits[:, 0][labelled].float(),
        targets[labelled],
        reduction="none",
    )

    return F.relu(
        fused_ce - global_ce - tolerance
    ).mean()


def _branch_losses(
    *,
    features: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    labelled: torch.Tensor,
    training: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 3 or features.shape[1] != 2:
        raise ValueError(
            "features must have shape [B, 2, D]"
        )
    if logits.ndim != 3 or logits.shape[1] != 2:
        raise ValueError("logits must have shape [B, 2, C]")

    if bool(labelled.any()):
        supervised = F.cross_entropy(
            logits[:, 0][labelled].float(),
            targets[labelled],
        )
    else:
        supervised = logits.float().sum() * 0.0

    pseudo = logits.detach().mean(dim=1).argmax(dim=1)
    batch_size = features.shape[0]

    confusion = torch.eye(
        2 * batch_size,
        device=features.device,
        dtype=features.dtype,
    )

    contrastive = selex_loss(
        features,
        targets.clamp_min(0),
        labelled,
        (pseudo,),
        confusion,
        temperature=float(training["temperature"]),
        sup_con_weight=float(
            training["supervised_weight"]
        ),
    )

    return supervised, contrastive


def _forward_pairs(
    *,
    model: nn.Module,
    views: torch.Tensor,
    variant: str,
) -> dict[str, torch.Tensor]:
    batch_size = views.shape[0]
    flat = views.reshape(
        batch_size * 2,
        3,
        224,
        224,
    )

    output = model.branches(flat)

    if variant == "selex":
        features = output.features.reshape(
            batch_size,
            2,
            -1,
        )
        logits = output.logits.reshape(
            batch_size,
            2,
            -1,
        )

        zeros = features.new_zeros(batch_size, 2)

        return {
            "fused_features": features,
            "global_features": features,
            "detail_features": features,
            "fused_logits": logits,
            "global_logits": logits,
            "detail_logits": logits,
            "fusion_weight": zeros,
            "adaptive_k": zeros,
            "retained_fraction": zeros,
        }

    return {
        "fused_features": output.fused_features.reshape(
            batch_size,
            2,
            -1,
        ),
        "global_features": output.global_features.reshape(
            batch_size,
            2,
            -1,
        ),
        "detail_features": output.detail_features.reshape(
            batch_size,
            2,
            -1,
        ),
        "fused_logits": output.fused_logits.reshape(
            batch_size,
            2,
            -1,
        ),
        "global_logits": output.global_logits.reshape(
            batch_size,
            2,
            -1,
        ),
        "detail_logits": output.detail_logits.reshape(
            batch_size,
            2,
            -1,
        ),
        "fusion_weight": output.fusion_weight.reshape(
            batch_size,
            2,
        ),
        "adaptive_k": output.selection.adaptive_k.reshape(
            batch_size,
            2,
        ),
        "retained_fraction": (
            output.selection.retained_fraction.reshape(
                batch_size,
                2,
            )
        ),
    }


def _training_batch(
    *,
    model: nn.Module,
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
]:
    """Return independently routed global and DeltaSub losses."""

    training = config["training"]
    variant = config["method"]["variant"]

    views = batch["views"].to(device)
    labelled = batch["labelled"].to(device)
    targets = batch["target"].to(device)

    branches = _forward_pairs(
        model=model,
        views=views,
        variant=variant,
    )

    global_classification, global_selex = _branch_losses(
        features=branches["global_features"],
        logits=branches["global_logits"],
        targets=targets,
        labelled=labelled,
        training=training,
    )

    global_loss = float(
        training["global_branch_weight"]
    ) * (
        float(training["classification_weight"])
        * global_classification
        + float(training["selex_weight"])
        * global_selex
    )

    branch_stats = {
        "global_classification": float(
            global_classification.detach()
        ),
        "global_selex": float(global_selex.detach()),
    }

    if variant == "selex":
        delta_loss = global_loss * 0.0

        stats = {
            "total": float(global_loss.detach()),
            "global_loss": float(global_loss.detach()),
            "delta_loss": 0.0,
            "classification": float(
                global_classification.detach()
            ),
            "selex": float(global_selex.detach()),
            "utility": 0.0,
            "preservation": 0.0,
            "budget": 0.0,
            "mean_fusion": 0.0,
            "mean_utility_target": 0.0,
            "mean_k": 0.0,
            "mean_retained_fraction": 0.0,
            **branch_stats,
        }

        return global_loss, delta_loss, stats

    detail_classification, detail_selex = _branch_losses(
        features=branches["detail_features"],
        logits=branches["detail_logits"],
        targets=targets,
        labelled=labelled,
        training=training,
    )
    fused_classification, fused_selex = _branch_losses(
        features=branches["fused_features"],
        logits=branches["fused_logits"],
        targets=targets,
        labelled=labelled,
        training=training,
    )

    detail_weight = float(
        training["detail_branch_weight"]
    )
    fused_weight = float(
        training["fused_branch_weight"]
    )
    delta_weight = detail_weight + fused_weight

    if delta_weight <= 0:
        raise ValueError(
            "detail and fused weights cannot both be zero"
        )

    delta_classification = (
        detail_weight * detail_classification
        + fused_weight * fused_classification
    ) / delta_weight

    delta_selex = (
        detail_weight * detail_selex
        + fused_weight * fused_selex
    ) / delta_weight

    target = utility_target(
        global_logits=branches["global_logits"],
        detail_logits=branches["detail_logits"],
        global_features=branches["global_features"],
        detail_features=branches["detail_features"],
        targets=targets,
        labelled=labelled,
        temperature=float(
            training["utility_temperature"]
        ),
    )

    predicted = branches["fusion_weight"].mean(dim=1)

    with torch.autocast("cuda", enabled=False):
        utility = F.binary_cross_entropy(
            predicted.float().clamp(1e-6, 1 - 1e-6),
            target.float(),
        ) + 0.25 * F.mse_loss(
            branches["fusion_weight"][:, 0].float(),
            branches["fusion_weight"][:, 1].float(),
        )

    preservation = preservation_penalty(
        fused_logits=branches["fused_logits"],
        global_logits=branches["global_logits"],
        targets=targets,
        labelled=labelled,
        tolerance=float(
            training["preservation_tolerance"]
        ),
    )

    maximum_k = int(config["method"]["maximum_k"])

    if maximum_k:
        budget = (
            branches["adaptive_k"].float()
            / float(maximum_k)
        ).mean()
    else:
        budget = global_loss * 0.0

    delta_loss = (
        float(training["classification_weight"])
        * delta_classification
        + float(training["selex_weight"])
        * delta_selex
        + float(training["utility_weight"])
        * utility
        + float(training["preservation_weight"])
        * preservation
        + float(training["budget_weight"])
        * budget
    )

    total = global_loss + delta_loss

    stats = {
        "total": float(total.detach()),
        "global_loss": float(global_loss.detach()),
        "delta_loss": float(delta_loss.detach()),
        "classification": float(
            delta_classification.detach()
        ),
        "selex": float(delta_selex.detach()),
        "utility": float(utility.detach()),
        "preservation": float(preservation.detach()),
        "budget": float(budget.detach()),
        "mean_fusion": float(
            branches["fusion_weight"]
            .detach()
            .float()
            .mean()
        ),
        "mean_utility_target": float(
            target.detach().float().mean()
        ),
        "mean_k": float(
            branches["adaptive_k"]
            .detach()
            .float()
            .mean()
        ),
        "mean_retained_fraction": float(
            branches["retained_fraction"]
            .detach()
            .float()
            .mean()
        ),
        "detail_classification": float(
            detail_classification.detach()
        ),
        "detail_selex": float(detail_selex.detach()),
        "fused_classification": float(
            fused_classification.detach()
        ),
        "fused_selex": float(fused_selex.detach()),
        **branch_stats,
    }

    return global_loss, delta_loss, stats


def _backward_decoupled(
    *,
    model: nn.Module,
    global_loss: torch.Tensor,
    delta_loss: torch.Tensor,
    accumulation: int,
    variant: str,
) -> None:
    """Prevent detail objectives from corrupting backbone/head gradients."""

    if accumulation <= 0:
        raise ValueError("accumulation must be positive")

    global_scaled = global_loss / accumulation

    if variant == "selex":
        global_scaled.backward()
        return

    global_scaled.backward(retain_graph=True)

    protected = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            name.startswith("backbone.")
            or name.startswith("head.")
        )
    ]

    preserved_gradients = {
        parameter: (
            None
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for parameter in protected
    }

    for parameter in protected:
        parameter.grad = None

    (delta_loss / accumulation).backward()

    # Discard backbone/head gradients produced by Delta losses.
    # Their gradients come exclusively from matched SelEx.
    for parameter, gradient in preserved_gradients.items():
        parameter.grad = gradient

def _summary(
    values: list[float],
    mask: np.ndarray,
) -> dict[str, float]:
    selected = np.asarray(values, dtype=np.float64)[mask]

    return {
        "count": int(selected.size),
        "mean": float(selected.mean()) if selected.size else 0.0,
        "std": float(selected.std()) if selected.size else 0.0,
    }


def _with_hmean(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    result = dict(metrics)
    old = float(result["old"])
    new = float(result["new"])

    result["hmean"] = (
        2.0 * old * new / (old + new)
        if old + new > 0
        else 0.0
    )

    return result


@torch.inference_mode()
def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    variant: str,
    output_directory: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    model.eval()

    targets: list[int] = []
    old_mask: list[bool] = []
    fused_predictions: list[int] = []
    global_predictions: list[int] = []
    detail_predictions: list[int] = []
    fusion_weights: list[float] = []
    selected_k: list[float] = []
    retained: list[float] = []

    started = time.perf_counter()

    for batch in loader:
        images = batch["image"].to(device)
        output = model.branches(images)

        if variant == "selex":
            logits = output.logits
            fused = logits
            global_logits = logits
            detail_logits = logits
            batch_fusion = torch.zeros(
                images.shape[0],
                device=device,
            )
            batch_k = torch.zeros_like(batch_fusion)
            batch_retained = torch.zeros_like(batch_fusion)
        else:
            fused = output.fused_logits
            global_logits = output.global_logits
            detail_logits = output.detail_logits
            batch_fusion = output.fusion_weight
            batch_k = output.selection.adaptive_k
            batch_retained = (
                output.selection.retained_fraction
            )

        targets.extend(batch["target"].tolist())
        old_mask.extend(batch["old"].tolist())

        fused_predictions.extend(
            fused.float().argmax(dim=1).cpu().tolist()
        )
        global_predictions.extend(
            global_logits.float().argmax(dim=1).cpu().tolist()
        )
        detail_predictions.extend(
            detail_logits.float().argmax(dim=1).cpu().tolist()
        )
        fusion_weights.extend(
            batch_fusion.float().cpu().tolist()
        )
        selected_k.extend(
            batch_k.float().cpu().tolist()
        )
        retained.extend(
            batch_retained.float().cpu().tolist()
        )

    elapsed = time.perf_counter() - started
    old = np.asarray(old_mask, dtype=bool)
    target_array = np.asarray(targets, dtype=np.int64)
    fused_array = np.asarray(
        fused_predictions,
        dtype=np.int64,
    )
    global_array = np.asarray(
        global_predictions,
        dtype=np.int64,
    )
    detail_array = np.asarray(
        detail_predictions,
        dtype=np.int64,
    )

    primary = _with_hmean(
        evaluate_gcd_v2(
            target_array,
            fused_array,
            old,
        ).as_dict()
    )
    global_metrics = _with_hmean(
        evaluate_gcd_v2(
            target_array,
            global_array,
            old,
        ).as_dict()
    )
    detail_metrics = _with_hmean(
        evaluate_gcd_v2(
            target_array,
            detail_array,
            old,
        ).as_dict()
    )

    all_mask = np.ones_like(old, dtype=bool)

    diagnostics = {
        "all": {
            "fusion_weight": _summary(
                fusion_weights,
                all_mask,
            ),
            "selected_k": _summary(
                selected_k,
                all_mask,
            ),
            "retained_fraction": _summary(
                retained,
                all_mask,
            ),
        },
        "old": {
            "fusion_weight": _summary(
                fusion_weights,
                old,
            ),
            "selected_k": _summary(
                selected_k,
                old,
            ),
            "retained_fraction": _summary(
                retained,
                old,
            ),
        },
        "new": {
            "fusion_weight": _summary(
                fusion_weights,
                ~old,
            ),
            "selected_k": _summary(
                selected_k,
                ~old,
            ),
            "retained_fraction": _summary(
                retained,
                ~old,
            ),
        },
        "evaluation_seconds": elapsed,
        "samples_per_second": (
            len(targets) / max(elapsed, 1e-12)
        ),
    }

    np.savez_compressed(
        output_directory / "predictions.npz",
        target=target_array,
        old=old,
        fused_prediction=fused_array,
        global_prediction=global_array,
        detail_prediction=detail_array,
        fusion_weight=np.asarray(
            fusion_weights,
            dtype=np.float32,
        ),
        selected_k=np.asarray(
            selected_k,
            dtype=np.float32,
        ),
        retained_fraction=np.asarray(
            retained,
            dtype=np.float32,
        ),
    )

    return (
        primary,
        {
            "global": global_metrics,
            "detail": detail_metrics,
        },
        diagnostics,
    )


def _optimizer(
    model: nn.Module,
    training: dict[str, Any],
) -> tuple[
    torch.optim.Optimizer,
    list[nn.Parameter],
    dict[str, int],
]:
    groups: dict[str, list[nn.Parameter]] = {
        "backbone": [],
        "head": [],
        "detail": [],
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("backbone."):
            groups["backbone"].append(parameter)
        elif name.startswith("head."):
            groups["head"].append(parameter)
        else:
            groups["detail"].append(parameter)

    learning_rates = {
        "backbone": float(
            training["backbone_learning_rate"]
        ),
        "head": float(training["head_learning_rate"]),
        "detail": float(
            training["detail_learning_rate"]
        ),
    }

    optimizer_groups = [
        {
            "params": parameters,
            "lr": learning_rates[name],
            "name": name,
        }
        for name, parameters in groups.items()
        if parameters
    ]

    if not optimizer_groups:
        raise RuntimeError("model has no trainable parameters")

    parameters = [
        parameter
        for group in optimizer_groups
        for parameter in group["params"]
    ]

    counts = {
        name: sum(
            parameter.numel()
            for parameter in values
        )
        for name, values in groups.items()
    }

    return (
        torch.optim.AdamW(
            optimizer_groups,
            weight_decay=float(training["weight_decay"]),
        ),
        parameters,
        counts,
    )


def _checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    generator: torch.Generator,
) -> None:
    atomic_torch_save(
        {
            "schema_version": SCHEMA,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config_hash": stable_hash(config),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
            "loader_generator": generator.get_state(),
        },
        path,
    )


def run(
    config_path: str | Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    config = json.loads(
        json.dumps(load_config(config_path))
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "exactly one visible GPU is required"
        )

    device = torch.device("cuda:0")
    seed = int(config["seed"])
    seed_everything(seed)

    output = Path(config["output_directory"])
    result_path = output / "result.json"

    if resume and result_path.is_file():
        return json.loads(
            result_path.read_text(encoding="utf-8")
        )

    output.mkdir(parents=True, exist_ok=True)

    resolved_text = yaml.safe_dump(
        config,
        sort_keys=True,
    )
    resolved_path = output / "resolved_config.yaml"

    if (
        resolved_path.is_file()
        and resolved_path.read_text(encoding="utf-8")
        != resolved_text
    ):
        raise ValueError(
            "output contains a different resolved config"
        )

    _atomic_text(resolved_path, resolved_text)

    records = read_manifest(
        config["dataset"]["manifest"],
        dataset_root=config["dataset"]["root"],
        check_images=True,
    )

    train_dataset = ProductionManifestDataset(
        records,
        config["dataset"]["root"],
        split="train",
        seed=seed,
        train=True,
    )
    test_dataset = ProductionManifestDataset(
        records,
        config["dataset"]["root"],
        split="test",
        seed=seed,
        train=False,
    )

    training = config["training"]
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            training["physical_batch_size"]
        ),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
        worker_init_fn=_worker_seed,
        persistent_workers=(
            int(training["num_workers"]) > 0
        ),
    )
    evaluation_loader = DataLoader(
        test_dataset,
        batch_size=int(
            training["evaluation_batch_size"]
        ),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        worker_init_fn=_worker_seed,
        persistent_workers=(
            int(training["num_workers"]) > 0
        ),
    )

    model = construct_model(config, device=device)

    # Model-specific module construction must not change training RNG.
    # This makes the global DeltaSub path exactly comparable to SelEx.
    seed_everything(seed)

    optimizer, parameters, group_counts = _optimizer(
        model,
        training,
    )

    epochs = int(training["epochs"])
    accumulation = int(
        training["gradient_accumulation"]
    )
    optimizer_steps = (
        epochs
        * math.ceil(len(train_loader) / accumulation)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        max(1, optimizer_steps),
    )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    provenance = {
        "schema_version": SCHEMA,
        "repository_commit": commit,
        "repository_dirty": bool(dirty),
        "dataset_manifest_sha256": sha256_file(
            config["dataset"]["manifest"]
        ),
        "split_validation_sha256": sha256_file(
            config["dataset"]["split_validation"]
        ),
        "backbone_checkpoint_sha256": sha256_file(
            config["backbone"]["checkpoint"]
        ),
        "selex_equivalence_sha256": sha256_file(
            config["selex"]["equivalence_report"]
        ),
        "gcd": gcd_provenance(
            implementation_path=(
                Path(__file__).parents[1]
                / "evaluation"
                / "gcd_v2.py"
            )
        ),
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "device": torch.cuda.get_device_name(device),
        },
    }
    _atomic_json(output / "provenance.json", provenance)

    start_epoch = 0
    global_step = 0
    resume_history: list[dict[str, Any]] = []
    checkpoint_path = output / "checkpoint_last.pt"
    metrics_path = output / "metrics.jsonl"

    if resume and checkpoint_path.is_file():
        state = load_checkpoint(
            checkpoint_path,
            map_location=device,
        )

        if state["config_hash"] != stable_hash(config):
            raise ValueError("resume configuration mismatch")

        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])

        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(
            state["rng"]["cuda"]
        )
        generator.set_state(
            state["loader_generator"]
        )

        resume_history.append(
            {
                "resumed_epoch": start_epoch,
                "checkpoint_sha256": sha256_file(
                    checkpoint_path
                ),
            }
        )

    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, epochs):
        train_dataset.epoch = epoch
        model.train()

        sums: dict[str, float] = {}
        batches = 0

        for batch_index, batch in enumerate(train_loader):
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                global_loss, delta_loss, stats = _training_batch(
                    model=model,
                    batch=batch,
                    config=config,
                    device=device,
                )

            if (
                not torch.isfinite(global_loss)
                or not torch.isfinite(delta_loss)
            ):
                raise FloatingPointError(
                    "non-finite training loss"
                )

            _backward_decoupled(
                model=model,
                global_loss=global_loss,
                delta_loss=delta_loss,
                accumulation=accumulation,
                variant=config["method"]["variant"],
            )

            for key, value in stats.items():
                sums[key] = sums.get(key, 0.0) + value

            batches += 1

            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )

            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    float(training["gradient_clipping"]),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        means = {
            key: value / max(batches, 1)
            for key, value in sums.items()
        }

        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "elapsed_seconds": (
                time.perf_counter() - started
            ),
            "learning_rates": {
                group.get("name", str(index)): group["lr"]
                for index, group in enumerate(
                    optimizer.param_groups
                )
            },
            **means,
        }

        with metrics_path.open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(
                json.dumps(record, sort_keys=True) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

        _checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
            config=config,
            generator=generator,
        )

        print(
            f"epoch={epoch + 1:03d}/{epochs} "
            f"loss={means['total']:.6f} "
            f"fusion={means['mean_fusion']:.4f} "
            f"target={means['mean_utility_target']:.4f} "
            f"k={means['mean_k']:.2f}",
            flush=True,
        )

    state = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(state["model"], strict=True)

    metrics, branch_metrics, diagnostics = evaluate(
        model=model,
        loader=evaluation_loader,
        device=device,
        variant=config["method"]["variant"],
        output_directory=output,
    )

    elapsed = time.perf_counter() - started
    report = model.parameter_report()

    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "method": config["method"]["variant"],
        "dataset": config["dataset"]["name"],
        "seed": seed,
        "metrics": metrics,
        "branch_metrics": branch_metrics,
        "diagnostics": diagnostics,
        "checkpoint_selection": (
            "fixed final epoch; test labels used only "
            "after training completed"
        ),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
        "train_samples": len(train_dataset),
        "eval_samples": len(test_dataset),
        "runtime_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated()
        ),
        "total_parameters": report["total"],
        "trainable_parameters": report["trainable"],
        "parameter_groups": group_counts,
        "resume_history": resume_history,
    }

    _atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))

    return result


def smoke(config_path: str | Path) -> None:
    config = json.loads(
        json.dumps(load_config(config_path))
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda:0")
    seed_everything(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats()

    records = read_manifest(
        config["dataset"]["manifest"],
        dataset_root=config["dataset"]["root"],
        check_images=True,
    )

    dataset = ProductionManifestDataset(
        records,
        config["dataset"]["root"],
        split="train",
        seed=int(config["seed"]),
        train=True,
    )

    labelled_index = next(
        index
        for index, record in enumerate(dataset.records)
        if record["labelled_or_unlabelled"] == "labelled"
    )
    unlabelled_index = next(
        index
        for index, record in enumerate(dataset.records)
        if record["labelled_or_unlabelled"] == "unlabelled"
    )

    items = [
        dataset[labelled_index],
        dataset[unlabelled_index],
    ]

    batch = {
        "views": torch.stack(
            [item["views"] for item in items]
        ),
        "labelled": torch.tensor(
            [item["labelled"] for item in items],
            dtype=torch.bool,
        ),
        "target": torch.tensor(
            [item["target"] for item in items],
            dtype=torch.long,
        ),
    }

    model = construct_model(config, device=device)
    model.train()

    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        global_loss, delta_loss, stats = _training_batch(
            model=model,
            batch=batch,
            config=config,
            device=device,
        )

    _backward_decoupled(
        model=model,
        global_loss=global_loss,
        delta_loss=delta_loss,
        accumulation=1,
        variant=config["method"]["variant"],
    )

    loss = global_loss + delta_loss

    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    }

    if not gradients:
        raise RuntimeError(
            "smoke test produced no gradients"
        )
    if not all(
        torch.isfinite(gradient).all()
        for gradient in gradients.values()
    ):
        raise FloatingPointError(
            "smoke test produced invalid gradients"
        )

    if not any(
        name.startswith("backbone.model.blocks.10")
        or name.startswith("backbone.model.blocks.11")
        for name in gradients
    ):
        raise RuntimeError(
            "final DINOv2 blocks received no gradients"
        )

    if config["method"]["variant"] == "deltasub_v2":
        required = (
            "child",
            "detail_adapter",
            "parent_query",
            "detail_key",
            "head",
        )

        for prefix in required:
            if not any(
                name.startswith(prefix)
                for name in gradients
            ):
                raise RuntimeError(
                    f"{prefix} received no gradients"
                )

    report = model.parameter_report()

    print("status: passed")
    print(f"variant: {config['method']['variant']}")
    print(f"loss: {float(loss.detach()):.6f}")
    print(f"fusion: {stats['mean_fusion']:.6f}")
    print(
        "utility target: "
        f"{stats['mean_utility_target']:.6f}"
    )
    print(f"mean K: {stats['mean_k']:.2f}")
    print(f"trainable parameters: {report['trainable']}")
    print(
        "peak CUDA MiB: "
        f"{torch.cuda.max_memory_allocated() / 2**20:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        smoke(args.config)
    else:
        run(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
