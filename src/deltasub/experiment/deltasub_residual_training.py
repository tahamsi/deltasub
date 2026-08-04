"""Training and evaluation for practical dense DeltaSub residuals."""

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
from ..models.backbones.dinov2 import (
    DINOv2Adapter,
)
from ..models.deltasub_residual import (
    DeltaSubResidual,
    DeltaSubResidualConfig,
)
from ..utils.checkpointing import (
    atomic_torch_save,
    load_checkpoint,
)
from ..utils.hashing import (
    sha256_file,
    stable_hash,
)
from ..utils.reproducibility import (
    seed_everything,
)
from .deltasub_v2_training import (
    _branch_losses,
    preservation_penalty,
)
from .training import (
    ProductionManifestDataset,
    _atomic_json,
    _atomic_text,
    _worker_seed,
)


SCHEMA = "deltasub-residual.experiment.v1"


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    value = yaml.safe_load(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            "configuration must be a mapping"
        )
    if value.get("schema_version") != SCHEMA:
        raise ValueError(
            "unsupported residual experiment schema"
        )

    required = {
        "dataset",
        "backbone",
        "selex",
        "seed",
        "training",
        "method",
        "output_directory",
    }

    if required - set(value):
        raise ValueError(
            "residual configuration is incomplete"
        )

    method = value["method"]

    if method.get("variant") != (
        "deltasub_residual"
    ):
        raise ValueError(
            "method.variant must be "
            "deltasub_residual"
        )

    if (
        value["training"].get(
            "supervision_mode",
            "gcd",
        )
        != "gcd"
    ):
        raise ValueError(
            "residual gate must use GCD supervision"
        )

    effective_batch = (
        int(
            value["training"][
                "physical_batch_size"
            ]
        )
        * int(
            value["training"][
                "gradient_accumulation"
            ]
        )
    )

    if effective_batch != 128:
        raise ValueError(
            "effective batch must remain 128"
        )

    DeltaSubResidualConfig(
        feature_dim=768,
        insertion_block=int(
            method["insertion_block"]
        ),
        trainable_blocks=int(
            method["trainable_blocks"]
        ),
        local_channels=int(
            method["local_channels"]
        ),
        predictor_hidden_dim=int(
            method["predictor_hidden_dim"]
        ),
        transport_rank=int(
            method["transport_rank"]
        ),
        maximum_correction_ratio=float(
            method[
                "maximum_correction_ratio"
            ]
        ),
        initial_scale=float(
            method["initial_scale"]
        ),
        use_prediction_residual=bool(
            method[
                "use_prediction_residual"
            ]
        ),
        use_semantic_projection=bool(
            method[
                "use_semantic_projection"
            ]
        ),
    ).validate()

    for filename in (
        value["dataset"]["manifest"],
        value["dataset"][
            "split_validation"
        ],
        value["backbone"]["checkpoint"],
        value["selex"][
            "equivalence_report"
        ],
    ):
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)

    return value


def class_count(
    config: dict[str, Any],
) -> int:
    split = json.loads(
        Path(
            config["dataset"][
                "split_validation"
            ]
        ).read_text(
            encoding="utf-8"
        )
    )

    classes = (
        set(split["known_class_ids"])
        | set(split["novel_class_ids"])
    )

    if not classes:
        raise ValueError(
            "class split contains no classes"
        )

    return len(classes)


def construct_model(
    config: dict[str, Any],
    *,
    device: torch.device,
) -> DeltaSubResidual:
    backbone_config = config["backbone"]

    backbone = (
        DINOv2Adapter
        .from_official_checkpoint(
            backbone_config["checkpoint"],
            backbone_config[
                "checkpoint_sha256"
            ],
            source_root=backbone_config[
                "source_root"
            ],
            model_name=backbone_config[
                "name"
            ],
        )
    )

    method = config["method"]

    model = DeltaSubResidual(
        backbone,
        class_count(config),
        DeltaSubResidualConfig(
            feature_dim=768,
            insertion_block=int(
                method["insertion_block"]
            ),
            trainable_blocks=int(
                method["trainable_blocks"]
            ),
            local_channels=int(
                method["local_channels"]
            ),
            predictor_hidden_dim=int(
                method[
                    "predictor_hidden_dim"
                ]
            ),
            transport_rank=int(
                method["transport_rank"]
            ),
            maximum_correction_ratio=float(
                method[
                    "maximum_correction_ratio"
                ]
            ),
            initial_scale=float(
                method["initial_scale"]
            ),
            use_prediction_residual=bool(
                method[
                    "use_prediction_residual"
                ]
            ),
            use_semantic_projection=bool(
                method[
                    "use_semantic_projection"
                ]
            ),
        ),
    )

    return model.to(device)


def with_hmean(
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


def forward_pairs(
    *,
    model: DeltaSubResidual,
    views: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = views.shape[0]

    flat = views.reshape(
        batch * 2,
        3,
        224,
        224,
    )

    output = model.branches(flat)

    def paired(
        value: torch.Tensor,
    ) -> torch.Tensor:
        return value.reshape(
            batch,
            2,
            *value.shape[1:],
        )

    return {
        "global_features": paired(
            output.global_features
        ),
        "refined_features": paired(
            output.refined_features
        ),
        "global_logits": paired(
            output.global_logits
        ),
        "refined_logits": paired(
            output.refined_logits
        ),
        "residual_tokens": paired(
            output.residual_tokens
        ),
        "corrections": paired(
            output.corrections
        ),
        "parent_tokens": paired(
            output.parent_tokens
        ),
        "cls_tokens": paired(
            output.cls_tokens
        ),
        "correction_ratio": paired(
            output.correction_ratio
        ),
        "residual_norm": paired(
            output.residual_norm
        ),
        "injection_scale": (
            output.injection_scale
        ),
    }


def cross_view_residual_loss(
    residuals: torch.Tensor,
) -> torch.Tensor:
    """Match view-level residual content without assuming patch alignment."""

    if (
        residuals.ndim != 4
        or residuals.shape[1] != 2
    ):
        raise ValueError(
            "residuals must have shape "
            "[B, 2, 256, D]"
        )

    pooled = residuals.float().mean(
        dim=2
    )
    pooled = F.normalize(
        pooled,
        dim=-1,
        eps=1e-6,
    )

    direction = 1.0 - (
        pooled[:, 0]
        * pooled[:, 1]
    ).sum(
        dim=-1
    ).mean()

    first_norms = torch.sort(
        residuals[:, 0]
        .float()
        .norm(dim=-1),
        dim=1,
    ).values
    second_norms = torch.sort(
        residuals[:, 1]
        .float()
        .norm(dim=-1),
        dim=1,
    ).values

    distributions = F.smooth_l1_loss(
        first_norms,
        second_norms,
    )

    return direction + 0.10 * distributions


def correction_orthogonality_loss(
    *,
    corrections: torch.Tensor,
    parents: torch.Tensor,
    cls_tokens: torch.Tensor,
) -> torch.Tensor:
    if corrections.shape != parents.shape:
        raise ValueError(
            "corrections and parents must match"
        )
    if (
        cls_tokens.shape[:2]
        != corrections.shape[:2]
        or cls_tokens.shape[-1]
        != corrections.shape[-1]
    ):
        raise ValueError(
            "CLS token shape mismatch"
        )

    correction = corrections.float()
    parent = parents.float()
    cls = cls_tokens.float().unsqueeze(2)

    parent_overlap = F.cosine_similarity(
        correction,
        parent,
        dim=-1,
        eps=1e-6,
    ).square()

    cls_overlap = F.cosine_similarity(
        correction,
        cls.expand_as(correction),
        dim=-1,
        eps=1e-6,
    ).square()

    return (
        parent_overlap.mean()
        + cls_overlap.mean()
    )


def residual_variance_loss(
    residuals: torch.Tensor,
) -> torch.Tensor:
    pooled = residuals.float().mean(
        dim=2
    ).reshape(
        -1,
        residuals.shape[-1],
    )

    standard_deviation = torch.sqrt(
        pooled.var(
            dim=0,
            unbiased=False,
        )
        + 1e-4
    )

    return F.relu(
        0.02 - standard_deviation
    ).mean()


def training_batch(
    *,
    model: DeltaSubResidual,
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    training = config["training"]

    views = batch["views"].to(device)
    targets = batch["target"].to(device)
    labelled = batch["labelled"].to(device)

    branches = forward_pairs(
        model=model,
        views=views,
    )

    global_classification, global_selex = (
        _branch_losses(
            features=branches[
                "global_features"
            ],
            logits=branches[
                "global_logits"
            ],
            targets=targets,
            labelled=labelled,
            training=training,
        )
    )

    refined_classification, refined_selex = (
        _branch_losses(
            features=branches[
                "refined_features"
            ],
            logits=branches[
                "refined_logits"
            ],
            targets=targets,
            labelled=labelled,
            training=training,
        )
    )

    classification_weight = float(
        training["classification_weight"]
    )
    selex_weight = float(
        training["selex_weight"]
    )

    global_task = (
        classification_weight
        * global_classification
        + selex_weight
        * global_selex
    )

    refined_task = (
        classification_weight
        * refined_classification
        + selex_weight
        * refined_selex
    )

    task = 0.5 * (
        global_task + refined_task
    )

    consistency = cross_view_residual_loss(
        branches["residual_tokens"]
    )

    orthogonal = (
        correction_orthogonality_loss(
            corrections=branches[
                "corrections"
            ],
            parents=branches[
                "parent_tokens"
            ],
            cls_tokens=branches[
                "cls_tokens"
            ],
        )
    )

    variance = residual_variance_loss(
        branches["residual_tokens"]
    )

    budget = branches[
        "correction_ratio"
    ].float().mean()

    preservation = preservation_penalty(
        fused_logits=branches[
            "refined_logits"
        ],
        global_logits=branches[
            "global_logits"
        ],
        targets=targets,
        labelled=labelled,
        tolerance=float(
            training[
                "preservation_tolerance"
            ]
        ),
    )

    loss = (
        task
        + float(
            training[
                "residual_consistency_weight"
            ]
        )
        * consistency
        + float(
            training[
                "orthogonal_weight"
            ]
        )
        * orthogonal
        + float(
            training[
                "variance_weight"
            ]
        )
        * variance
        + float(
            training[
                "residual_budget_weight"
            ]
        )
        * budget
        + float(
            training[
                "preservation_weight"
            ]
        )
        * preservation
    )

    stats = {
        "total": float(loss.detach()),
        "task": float(task.detach()),
        "global_classification": float(
            global_classification.detach()
        ),
        "global_selex": float(
            global_selex.detach()
        ),
        "refined_classification": float(
            refined_classification.detach()
        ),
        "refined_selex": float(
            refined_selex.detach()
        ),
        "consistency": float(
            consistency.detach()
        ),
        "orthogonal": float(
            orthogonal.detach()
        ),
        "variance": float(
            variance.detach()
        ),
        "budget": float(
            budget.detach()
        ),
        "preservation": float(
            preservation.detach()
        ),
        "mean_residual_norm": float(
            branches["residual_norm"]
            .detach()
            .float()
            .mean()
        ),
        "mean_correction_ratio": float(
            branches[
                "correction_ratio"
            ]
            .detach()
            .float()
            .mean()
        ),
        "injection_scale": float(
            branches[
                "injection_scale"
            ]
            .detach()
            .float()
        ),
    }

    return loss, stats


def optimizer_for(
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
        "residual": [],
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("backbone."):
            groups["backbone"].append(
                parameter
            )
        elif name.startswith("head."):
            groups["head"].append(
                parameter
            )
        else:
            groups["residual"].append(
                parameter
            )

    learning_rates = {
        "backbone": float(
            training[
                "backbone_learning_rate"
            ]
        ),
        "head": float(
            training["head_learning_rate"]
        ),
        "residual": float(
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

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(
            training["weight_decay"]
        ),
    )

    return optimizer, parameters, counts


def checkpoint(
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
                "cuda": (
                    torch.cuda
                    .get_rng_state_all()
                ),
            },
            "loader_generator": (
                generator.get_state()
            ),
        },
        path,
    )


def summary(
    values: list[float],
    mask: np.ndarray,
) -> dict[str, float]:
    selected = np.asarray(
        values,
        dtype=np.float64,
    )[mask]

    return {
        "count": int(selected.size),
        "mean": (
            float(selected.mean())
            if selected.size
            else 0.0
        ),
        "std": (
            float(selected.std())
            if selected.size
            else 0.0
        ),
    }


@torch.inference_mode()
def evaluate(
    *,
    model: DeltaSubResidual,
    loader: DataLoader,
    device: torch.device,
    output_directory: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    model.eval()

    targets: list[int] = []
    old_mask: list[bool] = []
    refined_predictions: list[int] = []
    global_predictions: list[int] = []
    correction_ratios: list[float] = []
    residual_norms: list[float] = []

    started = time.perf_counter()

    for batch in loader:
        images = batch["image"].to(device)
        output = model.branches(images)

        targets.extend(
            batch["target"].tolist()
        )
        old_mask.extend(
            batch["old"].tolist()
        )

        refined_predictions.extend(
            output.refined_logits
            .float()
            .argmax(dim=1)
            .cpu()
            .tolist()
        )
        global_predictions.extend(
            output.global_logits
            .float()
            .argmax(dim=1)
            .cpu()
            .tolist()
        )

        correction_ratios.extend(
            output.correction_ratio
            .float()
            .mean(dim=1)
            .cpu()
            .tolist()
        )
        residual_norms.extend(
            output.residual_norm
            .float()
            .mean(dim=1)
            .cpu()
            .tolist()
        )

    elapsed = time.perf_counter() - started

    target_array = np.asarray(
        targets,
        dtype=np.int64,
    )
    old = np.asarray(
        old_mask,
        dtype=bool,
    )
    refined = np.asarray(
        refined_predictions,
        dtype=np.int64,
    )
    global_array = np.asarray(
        global_predictions,
        dtype=np.int64,
    )

    primary = with_hmean(
        evaluate_gcd_v2(
            target_array,
            refined,
            old,
        ).as_dict()
    )

    global_metrics = with_hmean(
        evaluate_gcd_v2(
            target_array,
            global_array,
            old,
        ).as_dict()
    )

    all_mask = np.ones_like(
        old,
        dtype=bool,
    )

    diagnostics = {
        "all": {
            "correction_ratio": summary(
                correction_ratios,
                all_mask,
            ),
            "residual_norm": summary(
                residual_norms,
                all_mask,
            ),
        },
        "old": {
            "correction_ratio": summary(
                correction_ratios,
                old,
            ),
            "residual_norm": summary(
                residual_norms,
                old,
            ),
        },
        "new": {
            "correction_ratio": summary(
                correction_ratios,
                ~old,
            ),
            "residual_norm": summary(
                residual_norms,
                ~old,
            ),
        },
        "injection_scale": float(
            torch.sigmoid(
                model.injection_scale_logit
            )
            .detach()
            .cpu()
        ),
        "evaluation_seconds": elapsed,
        "samples_per_second": (
            len(targets)
            / max(elapsed, 1e-12)
        ),
    }

    np.savez_compressed(
        output_directory
        / "predictions.npz",
        target=target_array,
        old=old,
        fused_prediction=refined,
        refined_prediction=refined,
        global_prediction=global_array,
        correction_ratio=np.asarray(
            correction_ratios,
            dtype=np.float32,
        ),
        residual_norm=np.asarray(
            residual_norms,
            dtype=np.float32,
        ),
    )

    return (
        primary,
        {"global": global_metrics},
        diagnostics,
    )


def run(
    config_path: str | Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    config = json.loads(
        json.dumps(
            load_config(config_path)
        )
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "exactly one visible GPU is required"
        )

    device = torch.device("cuda:0")
    seed = int(config["seed"])
    seed_everything(seed)

    output = Path(
        config["output_directory"]
    )
    result_path = output / "result.json"

    if resume and result_path.is_file():
        result = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )
        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
        )
        return result

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_text = yaml.safe_dump(
        config,
        sort_keys=True,
    )
    resolved_path = (
        output / "resolved_config.yaml"
    )

    if (
        resolved_path.is_file()
        and resolved_path.read_text(
            encoding="utf-8"
        )
        != resolved_text
    ):
        raise ValueError(
            "output contains a different config"
        )

    _atomic_text(
        resolved_path,
        resolved_text,
    )

    records = read_manifest(
        config["dataset"]["manifest"],
        dataset_root=(
            config["dataset"]["root"]
        ),
        check_images=True,
    )

    train_dataset = (
        ProductionManifestDataset(
            records,
            config["dataset"]["root"],
            split="train",
            seed=seed,
            train=True,
        )
    )
    test_dataset = (
        ProductionManifestDataset(
            records,
            config["dataset"]["root"],
            split="test",
            seed=seed,
            train=False,
        )
    )

    training = config["training"]
    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            training[
                "physical_batch_size"
            ]
        ),
        shuffle=True,
        generator=generator,
        num_workers=int(
            training["num_workers"]
        ),
        worker_init_fn=_worker_seed,
        persistent_workers=(
            int(
                training["num_workers"]
            )
            > 0
        ),
    )

    evaluation_loader = DataLoader(
        test_dataset,
        batch_size=int(
            training[
                "evaluation_batch_size"
            ]
        ),
        shuffle=False,
        num_workers=int(
            training["num_workers"]
        ),
        worker_init_fn=_worker_seed,
        persistent_workers=(
            int(
                training["num_workers"]
            )
            > 0
        ),
    )

    model = construct_model(
        config,
        device=device,
    )

    # Keep the matched backbone/head RNG trajectory.
    seed_everything(seed)

    optimizer, parameters, group_counts = (
        optimizer_for(
            model,
            training,
        )
    )

    epochs = int(training["epochs"])
    accumulation = int(
        training[
            "gradient_accumulation"
        ]
    )

    total_steps = (
        epochs
        * math.ceil(
            len(train_loader)
            / accumulation
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            max(1, total_steps),
        )
    )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    provenance = {
        "schema_version": SCHEMA,
        "repository_commit": commit,
        "repository_dirty": bool(dirty),
        "dataset_manifest_sha256": (
            sha256_file(
                config["dataset"][
                    "manifest"
                ]
            )
        ),
        "split_validation_sha256": (
            sha256_file(
                config["dataset"][
                    "split_validation"
                ]
            )
        ),
        "backbone_checkpoint_sha256": (
            sha256_file(
                config["backbone"][
                    "checkpoint"
                ]
            )
        ),
        "selex_equivalence_sha256": (
            sha256_file(
                config["selex"][
                    "equivalence_report"
                ]
            )
        ),
        "gcd": gcd_provenance(
            implementation_path=(
                Path(__file__).parents[1]
                / "evaluation"
                / "gcd_v2.py"
            )
        ),
        "seed": seed,
        "no_oracle": True,
        "no_patch_selection": True,
        "sequence_length_preserved": True,
        "environment": {
            "python": platform.python_version(),
            "torch": str(
                torch.__version__
            ),
            "cuda": str(
                torch.version.cuda
            ),
            "device": (
                torch.cuda
                .get_device_name(device)
            ),
        },
    }

    _atomic_json(
        output / "provenance.json",
        provenance,
    )

    checkpoint_path = (
        output / "checkpoint_last.pt"
    )
    metrics_path = (
        output / "metrics.jsonl"
    )

    start_epoch = 0
    global_step = 0
    resume_history: list[
        dict[str, Any]
    ] = []

    if resume and checkpoint_path.is_file():
        state = load_checkpoint(
            checkpoint_path,
            map_location=device,
        )

        if (
            state["config_hash"]
            != stable_hash(config)
        ):
            raise ValueError(
                "resume configuration mismatch"
            )

        model.load_state_dict(
            state["model"],
            strict=True,
        )
        optimizer.load_state_dict(
            state["optimizer"]
        )
        scheduler.load_state_dict(
            state["scheduler"]
        )

        start_epoch = int(
            state["epoch"]
        )
        global_step = int(
            state["global_step"]
        )

        random.setstate(
            state["rng"]["python"]
        )
        np.random.set_state(
            state["rng"]["numpy"]
        )
        torch.set_rng_state(
            state["rng"]["torch"]
        )
        torch.cuda.set_rng_state_all(
            state["rng"]["cuda"]
        )
        generator.set_state(
            state["loader_generator"]
        )

        resume_history.append(
            {
                "resumed_epoch": (
                    start_epoch
                ),
                "checkpoint_sha256": (
                    sha256_file(
                        checkpoint_path
                    )
                ),
            }
        )

    optimizer.zero_grad(
        set_to_none=True
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(
        start_epoch,
        epochs,
    ):
        train_dataset.epoch = epoch
        model.train()

        sums: dict[str, float] = {}
        batches = 0

        for batch_index, batch in enumerate(
            train_loader
        ):
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                loss, stats = training_batch(
                    model=model,
                    batch=batch,
                    config=config,
                    device=device,
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite residual loss"
                )

            (
                loss / accumulation
            ).backward()

            for key, value in stats.items():
                sums[key] = (
                    sums.get(key, 0.0)
                    + value
                )

            batches += 1

            should_step = (
                (
                    batch_index + 1
                )
                % accumulation
                == 0
                or batch_index + 1
                == len(train_loader)
            )

            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    float(
                        training[
                            "gradient_clipping"
                        ]
                    ),
                )
                optimizer.step()
                optimizer.zero_grad(
                    set_to_none=True
                )
                scheduler.step()
                global_step += 1

        means = {
            key: value
            / max(batches, 1)
            for key, value in sums.items()
        }

        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "elapsed_seconds": (
                time.perf_counter()
                - started
            ),
            "learning_rates": {
                group.get(
                    "name",
                    str(index),
                ): group["lr"]
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
                json.dumps(
                    record,
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

        checkpoint(
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
            f"epoch={epoch + 1:03d}/"
            f"{epochs} "
            f"loss={means['total']:.6f} "
            f"task={means['task']:.6f} "
            f"consistency="
            f"{means['consistency']:.6f} "
            f"ratio="
            f"{means['mean_correction_ratio']:.5f} "
            f"scale="
            f"{means['injection_scale']:.5f}",
            flush=True,
        )

    state = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(
        state["model"],
        strict=True,
    )

    metrics, branch_metrics, diagnostics = (
        evaluate(
            model=model,
            loader=evaluation_loader,
            device=device,
            output_directory=output,
        )
    )

    elapsed = (
        time.perf_counter()
        - started
    )
    report = model.parameter_report()

    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "method": "deltasub_residual",
        "dataset": config["dataset"]["name"],
        "seed": seed,
        "supervision_mode": "gcd",
        "metrics": metrics,
        "branch_metrics": branch_metrics,
        "diagnostics": diagnostics,
        "configuration": config["method"],
        "checkpoint_selection": (
            "fixed final epoch; test labels "
            "used only after training"
        ),
        "checkpoint_epoch": int(
            state["epoch"]
        ),
        "checkpoint_sha256": (
            sha256_file(
                checkpoint_path
            )
        ),
        "train_samples": len(
            train_dataset
        ),
        "eval_samples": len(
            test_dataset
        ),
        "runtime_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "peak_cuda_memory_bytes": (
            torch.cuda
            .max_memory_allocated()
        ),
        "total_parameters": (
            report["total"]
        ),
        "trainable_parameters": (
            report["trainable"]
        ),
        "parameter_groups": group_counts,
        "resume_history": resume_history,
    }

    _atomic_json(
        result_path,
        result,
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    return result


def smoke(
    config_path: str | Path,
) -> None:
    config = load_config(config_path)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    device = torch.device("cuda:0")
    seed_everything(
        int(config["seed"])
    )

    records = read_manifest(
        config["dataset"]["manifest"],
        dataset_root=(
            config["dataset"]["root"]
        ),
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
        for index, record in enumerate(
            dataset.records
        )
        if record[
            "labelled_or_unlabelled"
        ]
        == "labelled"
    )
    unlabelled_index = next(
        index
        for index, record in enumerate(
            dataset.records
        )
        if record[
            "labelled_or_unlabelled"
        ]
        == "unlabelled"
    )

    items = [
        dataset[labelled_index],
        dataset[unlabelled_index],
    ]

    batch = {
        "views": torch.stack(
            [
                item["views"]
                for item in items
            ]
        ),
        "labelled": torch.tensor(
            [
                item["labelled"]
                for item in items
            ],
            dtype=torch.bool,
        ),
        "target": torch.tensor(
            [
                item["target"]
                for item in items
            ],
            dtype=torch.long,
        ),
    }

    model = construct_model(
        config,
        device=device,
    )
    model.train()

    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        loss, stats = training_batch(
            model=model,
            batch=batch,
            config=config,
            device=device,
        )

    loss.backward()

    gradients = {
        name: parameter.grad
        for name, parameter
        in model.named_parameters()
        if (
            parameter.requires_grad
            and parameter.grad is not None
        )
    }

    if not gradients:
        raise RuntimeError(
            "smoke test produced no gradients"
        )

    if not all(
        torch.isfinite(
            gradient
        ).all()
        for gradient in gradients.values()
    ):
        raise FloatingPointError(
            "smoke test produced invalid gradients"
        )

    required = [
        "backbone.model.blocks.10",
        "backbone.model.blocks.11",
        "head",
        "local_encoder",
        "transport_down",
        "transport_up",
        "injection_scale_logit",
    ]

    if config["method"][
        "use_prediction_residual"
    ]:
        required.append(
            "local_predictor"
        )

    if config["method"][
        "use_semantic_projection"
    ]:
        required.append(
            "semantic_projection"
        )

    for prefix in required:
        if not any(
            name.startswith(prefix)
            for name in gradients
        ):
            raise RuntimeError(
                f"{prefix} received no gradient"
            )

    print("status: passed")
    print(
        "variant: deltasub_residual"
    )
    print(
        "no oracle: true"
    )
    print(
        "no patch selection: true"
    )
    print(
        "sequence length preserved: true"
    )
    print(
        f"loss: {float(loss.detach()):.6f}"
    )

    for key in (
        "consistency",
        "orthogonal",
        "variance",
        "budget",
        "preservation",
        "mean_correction_ratio",
        "injection_scale",
    ):
        print(
            f"{key}: {stats[key]:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    args = parser.parse_args()

    if args.smoke:
        smoke(args.config)
    else:
        run(
            args.config,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
