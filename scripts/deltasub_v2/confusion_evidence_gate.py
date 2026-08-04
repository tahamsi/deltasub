#!/usr/bin/env python
"""Practical confusion-conditioned evidence gate for DeltaSub.

No oracle or test-label selection is used. A small evidence module is
trained on genuinely labelled Old-class training images. At inference,
the frozen baseline's top-two hypotheses define a contrast query over
its existing patch tokens. Only the top-two logit margin is corrected.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from deltasub.data.manifests import read_manifest
from deltasub.evaluation.gcd_v2 import evaluate_gcd_v2
from deltasub.experiment.deltasub_residual_compare import (
    bootstrap_hmean_delta,
    locate_matched_baseline,
)
from deltasub.experiment.deltasub_v2_training import (
    construct_model,
)
from deltasub.experiment.training import (
    ProductionManifestDataset,
    _atomic_json,
    _worker_seed,
)
from deltasub.utils.checkpointing import (
    atomic_torch_save,
    load_checkpoint,
)
from deltasub.utils.hashing import sha256_file
from deltasub.utils.reproducibility import seed_everything


SCHEMA = "deltasub.confusion-evidence.v1"


@dataclass(frozen=True)
class EvidenceConfig:
    feature_dim: int = 768
    rank: int = 128
    initial_temperature: float = 0.15
    initial_beta: float = 1.0
    epochs: int = 12
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-2
    batch_size: int = 16
    num_workers: int = 4
    sign_weight: float = 0.50
    consistency_weight: float = 0.10
    entropy_weight: float = 0.01
    beta_weight: float = 0.001

    def validate(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not 0 < self.rank <= self.feature_dim:
            raise ValueError("rank must be in (0, feature_dim]")
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if self.initial_beta <= 0:
            raise ValueError("initial_beta must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class ConfusionEvidence(nn.Module):
    """Read patch evidence conditioned on a pair of class prototypes."""

    def __init__(self, config: EvidenceConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.patch_norm = nn.LayerNorm(config.feature_dim)
        self.patch_projection = nn.Linear(
            config.feature_dim,
            config.rank,
            bias=False,
        )
        self.query_projection = nn.Linear(
            config.feature_dim,
            config.rank,
            bias=False,
        )

        nn.init.orthogonal_(self.patch_projection.weight)

        with torch.no_grad():
            self.query_projection.weight.copy_(
                self.patch_projection.weight
            )

        self.temperature_parameter = nn.Parameter(
            torch.tensor(
                inverse_softplus(
                    config.initial_temperature - 0.02
                )
            )
        )
        self.beta_parameter = nn.Parameter(
            torch.tensor(
                inverse_softplus(config.initial_beta)
            )
        )

    @property
    def temperature(self) -> torch.Tensor:
        return (
            F.softplus(self.temperature_parameter)
            + 0.02
        )

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_parameter)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        first_classes: torch.Tensor,
        second_classes: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError(
                "patch_tokens must have shape [B, N, D]"
            )

        batch, _, width = patch_tokens.shape

        if width != self.config.feature_dim:
            raise ValueError("patch feature width mismatch")
        if first_classes.shape != (batch,):
            raise ValueError("first_classes must have shape [B]")
        if second_classes.shape != (batch,):
            raise ValueError("second_classes must have shape [B]")

        contrast = (
            class_weights[first_classes]
            - class_weights[second_classes]
        )

        patches = F.normalize(
            self.patch_projection(
                self.patch_norm(patch_tokens)
            ).float(),
            dim=-1,
            eps=1e-6,
        )
        query = F.normalize(
            self.query_projection(contrast).float(),
            dim=-1,
            eps=1e-6,
        )

        raw = torch.einsum(
            "bnr,br->bn",
            patches,
            query,
        )

        centered = raw - raw.mean(
            dim=1,
            keepdim=True,
        )

        attention = F.softmax(
            centered.abs()
            / self.temperature.float(),
            dim=1,
        )

        evidence = (
            attention * centered
        ).sum(dim=1)

        return evidence, attention


def correct_top_two(
    logits: torch.Tensor,
    first_classes: torch.Tensor,
    second_classes: torch.Tensor,
    evidence: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, C]")

    batch = logits.shape[0]

    if first_classes.shape != (batch,):
        raise ValueError("first_classes shape mismatch")
    if second_classes.shape != (batch,):
        raise ValueError("second_classes shape mismatch")
    if evidence.shape != (batch,):
        raise ValueError("evidence shape mismatch")

    correction = (
        0.5
        * beta.float()
        * evidence.float()
    )

    delta = torch.zeros_like(
        logits,
        dtype=torch.float32,
    )

    delta.scatter_add_(
        1,
        first_classes[:, None],
        correction[:, None],
    )
    delta.scatter_add_(
        1,
        second_classes[:, None],
        -correction[:, None],
    )

    return logits.float() + delta


def hard_competitor(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    work = logits.float().clone()

    work.scatter_(
        1,
        targets[:, None],
        float("-inf"),
    )

    return work.argmax(dim=1)


def top_two(logits: torch.Tensor) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    indices = logits.float().topk(
        k=2,
        dim=1,
    ).indices

    return indices[:, 0], indices[:, 1]


def metric_dict(
    targets: np.ndarray,
    predictions: np.ndarray,
    old: np.ndarray,
) -> dict[str, float]:
    value = evaluate_gcd_v2(
        targets,
        predictions,
        old,
    ).as_dict()

    old_value = float(value["old"])
    new_value = float(value["new"])

    return {
        "all": float(value["all"]),
        "old": old_value,
        "new": new_value,
        "hmean": (
            2.0
            * old_value
            * new_value
            / (old_value + new_value)
            if old_value + new_value > 0
            else 0.0
        ),
    }


def stable_validation_sample(
    sample_id: str,
) -> bool:
    digest = hashlib.sha256(
        sample_id.encode("utf-8")
    ).digest()

    return digest[0] % 5 == 0


def baseline_forward(
    model: nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        output = (
            model.backbone.model
            .forward_features(images)
        )
        cls_tokens = output[
            "x_norm_clstoken"
        ]
        patch_tokens = output[
            "x_norm_patchtokens"
        ]
        logits = model.head(cls_tokens)

    if patch_tokens.ndim != 3:
        raise RuntimeError(
            "DINOv2 patch-token contract failed"
        )
    if patch_tokens.shape[1:] != (
        256,
        768,
    ):
        raise RuntimeError(
            "unexpected patch-token shape: "
            f"{tuple(patch_tokens.shape)}"
        )

    return logits.detach(), patch_tokens.detach()


def load_matched_baseline(
    device: torch.device,
) -> tuple[
    nn.Module,
    dict[str, Any],
    dict[str, Any],
    Path,
]:
    result_path, result = (
        locate_matched_baseline()
    )

    config_path = (
        result_path.parent
        / "resolved_config.yaml"
    )
    checkpoint_path = (
        result_path.parent
        / "checkpoint_last.pt"
    )

    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    if config["method"]["variant"] != "selex":
        raise ValueError(
            "matched checkpoint is not SelEx"
        )

    model = construct_model(
        config,
        device=device,
    )

    state = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        state["model"],
        strict=True,
    )
    model.requires_grad_(False)
    model.eval()

    provenance = {
        "result": str(result_path),
        "resolved_config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
    }

    return model, config, result, result_path


@torch.inference_mode()
def evaluate_known_validation(
    *,
    baseline: nn.Module,
    evidence_model: ConfusionEvidence,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    evidence_model.eval()

    baseline_correct = 0
    corrected_correct = 0
    changed = 0
    count = 0
    evidence_values: list[float] = []

    class_weights = (
        baseline.head.weight.detach()
    )

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        logits, patches = baseline_forward(
            baseline,
            images,
        )

        first, second = top_two(logits)

        evidence, _ = evidence_model(
            patches,
            first,
            second,
            class_weights,
        )

        corrected = correct_top_two(
            logits,
            first,
            second,
            evidence,
            evidence_model.beta,
        )

        baseline_prediction = logits.argmax(
            dim=1
        )
        corrected_prediction = corrected.argmax(
            dim=1
        )

        baseline_correct += int(
            (
                baseline_prediction
                == targets
            ).sum()
        )
        corrected_correct += int(
            (
                corrected_prediction
                == targets
            ).sum()
        )
        changed += int(
            (
                corrected_prediction
                != baseline_prediction
            ).sum()
        )
        count += int(targets.numel())

        evidence_values.extend(
            evidence.float().cpu().tolist()
        )

    return {
        "count": count,
        "baseline_accuracy": (
            baseline_correct / count
        ),
        "corrected_accuracy": (
            corrected_correct / count
        ),
        "delta": (
            corrected_correct
            - baseline_correct
        ) / count,
        "changed_fraction": changed / count,
        "mean_evidence": float(
            np.mean(evidence_values)
        ),
        "mean_absolute_evidence": float(
            np.mean(
                np.abs(evidence_values)
            )
        ),
    }


def train_evidence_model(
    *,
    baseline: nn.Module,
    evidence_model: ConfusionEvidence,
    train_dataset: ProductionManifestDataset,
    validation_loader: DataLoader,
    device: torch.device,
    config: EvidenceConfig,
    output: Path,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(
        26_011
    )

    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        worker_init_fn=_worker_seed,
        persistent_workers=(
            config.num_workers > 0
        ),
    )

    optimizer = torch.optim.AdamW(
        evidence_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=max(
                1,
                config.epochs
                * len(loader),
            ),
        )
    )

    class_weights = (
        baseline.head.weight.detach()
    )

    best_key: tuple[float, float, float] | None = None
    best_record: dict[str, Any] | None = None
    checkpoint_path = (
        output / "checkpoint_best.pt"
    )
    metrics_path = (
        output / "training_metrics.jsonl"
    )

    started = time.perf_counter()

    for epoch in range(config.epochs):
        train_dataset.epoch = epoch
        evidence_model.train()

        sums: dict[str, float] = {}
        batches = 0

        for batch in loader:
            views = batch["views"].to(device)
            targets = batch["target"].to(device)

            batch_size = views.shape[0]

            flat = views.reshape(
                batch_size * 2,
                3,
                224,
                224,
            )

            logits, patches = baseline_forward(
                baseline,
                flat,
            )

            repeated_targets = (
                targets[:, None]
                .expand(-1, 2)
                .reshape(-1)
            )

            competitors = hard_competitor(
                logits,
                repeated_targets,
            )

            evidence, attention = evidence_model(
                patches,
                repeated_targets,
                competitors,
                class_weights,
            )

            target_logits = logits.gather(
                1,
                repeated_targets[:, None],
            ).squeeze(1)

            competitor_logits = logits.gather(
                1,
                competitors[:, None],
            ).squeeze(1)

            baseline_margin = (
                target_logits.float()
                - competitor_logits.float()
            )

            corrected_margin = (
                baseline_margin
                + evidence_model.beta.float()
                * evidence.float()
            )

            pairwise = F.softplus(
                -corrected_margin
            ).mean()

            sign = F.softplus(
                -evidence.float() / 0.10
            ).mean()

            paired_evidence = evidence.reshape(
                batch_size,
                2,
            )

            consistency = F.smooth_l1_loss(
                paired_evidence[:, 0],
                paired_evidence[:, 1],
            )

            entropy = -(
                attention.float()
                * attention.float()
                .clamp_min(1e-8)
                .log()
            ).sum(dim=1)

            entropy = (
                entropy
                / math.log(
                    attention.shape[1]
                )
            ).mean()

            beta_penalty = (
                evidence_model.beta.float()
                .square()
            )

            loss = (
                pairwise
                + config.sign_weight * sign
                + config.consistency_weight
                * consistency
                + config.entropy_weight
                * entropy
                + config.beta_weight
                * beta_penalty
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite evidence loss"
                )

            optimizer.zero_grad(
                set_to_none=True
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                evidence_model.parameters(),
                1.0,
            )
            optimizer.step()
            scheduler.step()

            values = {
                "loss": float(
                    loss.detach()
                ),
                "pairwise": float(
                    pairwise.detach()
                ),
                "sign": float(
                    sign.detach()
                ),
                "consistency": float(
                    consistency.detach()
                ),
                "entropy": float(
                    entropy.detach()
                ),
                "beta": float(
                    evidence_model.beta.detach()
                ),
                "temperature": float(
                    evidence_model
                    .temperature.detach()
                ),
                "mean_evidence": float(
                    evidence.detach()
                    .float()
                    .mean()
                ),
            }

            for name, value in values.items():
                sums[name] = (
                    sums.get(name, 0.0)
                    + value
                )

            batches += 1

        validation = (
            evaluate_known_validation(
                baseline=baseline,
                evidence_model=evidence_model,
                loader=validation_loader,
                device=device,
            )
        )

        means = {
            name: value / max(batches, 1)
            for name, value in sums.items()
        }

        record = {
            "epoch": epoch + 1,
            "elapsed_seconds": (
                time.perf_counter()
                - started
            ),
            "training": means,
            "validation": validation,
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

        selection_key = (
            validation[
                "corrected_accuracy"
            ],
            validation["delta"],
            -validation[
                "changed_fraction"
            ],
        )

        if (
            best_key is None
            or selection_key > best_key
        ):
            best_key = selection_key
            best_record = deepcopy(record)

            atomic_torch_save(
                {
                    "schema_version": SCHEMA,
                    "module": (
                        evidence_model
                        .state_dict()
                    ),
                    "configuration": (
                        asdict(config)
                    ),
                    "epoch": epoch + 1,
                    "selection": validation,
                },
                checkpoint_path,
            )

        print(
            f"epoch={epoch + 1:02d}/"
            f"{config.epochs} "
            f"loss={means['loss']:.6f} "
            f"val_base="
            f"{validation['baseline_accuracy']:.4f} "
            f"val_corrected="
            f"{validation['corrected_accuracy']:.4f} "
            f"delta={validation['delta']:+.4f} "
            f"changed="
            f"{validation['changed_fraction']:.4f} "
            f"beta={means['beta']:.4f} "
            f"tau={means['temperature']:.4f}",
            flush=True,
        )

    if best_record is None:
        raise RuntimeError(
            "no evidence checkpoint selected"
        )

    state = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    evidence_model.load_state_dict(
        state["module"],
        strict=True,
    )

    return {
        "best_epoch": int(
            state["epoch"]
        ),
        "best_validation": (
            state["selection"]
        ),
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_sha256": (
            sha256_file(checkpoint_path)
        ),
        "history_last": record,
    }


@torch.inference_mode()
def evaluate_test(
    *,
    baseline: nn.Module,
    evidence_model: ConfusionEvidence,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    evidence_model.eval()

    targets: list[int] = []
    old_values: list[bool] = []
    baseline_predictions: list[int] = []
    corrected_predictions: list[int] = []
    evidence_values: list[float] = []
    entropy_values: list[float] = []
    baseline_margins: list[float] = []
    corrected_margins: list[float] = []

    class_weights = (
        baseline.head.weight.detach()
    )

    for batch in loader:
        images = batch["image"].to(device)

        logits, patches = baseline_forward(
            baseline,
            images,
        )

        first, second = top_two(logits)

        evidence, attention = evidence_model(
            patches,
            first,
            second,
            class_weights,
        )

        corrected = correct_top_two(
            logits,
            first,
            second,
            evidence,
            evidence_model.beta,
        )

        baseline_margin = (
            logits.gather(
                1,
                first[:, None],
            ).squeeze(1)
            - logits.gather(
                1,
                second[:, None],
            ).squeeze(1)
        )

        corrected_margin = (
            corrected.gather(
                1,
                first[:, None],
            ).squeeze(1)
            - corrected.gather(
                1,
                second[:, None],
            ).squeeze(1)
        )

        entropy = -(
            attention.float()
            * attention.float()
            .clamp_min(1e-8)
            .log()
        ).sum(dim=1) / math.log(
            attention.shape[1]
        )

        targets.extend(
            batch["target"].tolist()
        )
        old_values.extend(
            batch["old"].tolist()
        )
        baseline_predictions.extend(
            logits.argmax(
                dim=1
            ).cpu().tolist()
        )
        corrected_predictions.extend(
            corrected.argmax(
                dim=1
            ).cpu().tolist()
        )
        evidence_values.extend(
            evidence.float().cpu().tolist()
        )
        entropy_values.extend(
            entropy.cpu().tolist()
        )
        baseline_margins.extend(
            baseline_margin.float()
            .cpu().tolist()
        )
        corrected_margins.extend(
            corrected_margin.float()
            .cpu().tolist()
        )

    target_array = np.asarray(
        targets,
        dtype=np.int64,
    )
    old = np.asarray(
        old_values,
        dtype=bool,
    )
    baseline_array = np.asarray(
        baseline_predictions,
        dtype=np.int64,
    )
    corrected_array = np.asarray(
        corrected_predictions,
        dtype=np.int64,
    )

    baseline_metrics = metric_dict(
        target_array,
        baseline_array,
        old,
    )
    corrected_metrics = metric_dict(
        target_array,
        corrected_array,
        old,
    )

    delta = {
        name: (
            corrected_metrics[name]
            - baseline_metrics[name]
        )
        for name in (
            "all",
            "old",
            "new",
            "hmean",
        )
    }

    changed = (
        corrected_array
        != baseline_array
    )

    diagnostics: dict[str, Any] = {
        "changed_count": int(
            changed.sum()
        ),
        "changed_fraction": float(
            changed.mean()
        ),
        "old_changed_fraction": float(
            changed[old].mean()
        ),
        "new_changed_fraction": float(
            changed[~old].mean()
        ),
        "mean_evidence": float(
            np.mean(evidence_values)
        ),
        "mean_absolute_evidence": float(
            np.mean(
                np.abs(evidence_values)
            )
        ),
        "mean_attention_entropy": float(
            np.mean(entropy_values)
        ),
        "mean_baseline_top2_margin": float(
            np.mean(baseline_margins)
        ),
        "mean_corrected_top2_margin": float(
            np.mean(corrected_margins)
        ),
        "beta": float(
            evidence_model.beta.cpu()
        ),
        "temperature": float(
            evidence_model
            .temperature.cpu()
        ),
    }

    return {
        "targets": target_array,
        "old": old,
        "baseline_predictions": (
            baseline_array
        ),
        "corrected_predictions": (
            corrected_array
        ),
        "baseline": baseline_metrics,
        "corrected": corrected_metrics,
        "delta": delta,
        "diagnostics": diagnostics,
    }


def self_test() -> None:
    torch.manual_seed(9)

    config = EvidenceConfig(
        feature_dim=16,
        rank=8,
        epochs=1,
        batch_size=2,
        num_workers=0,
    )

    model = ConfusionEvidence(config)

    patches = torch.randn(
        4,
        12,
        16,
    )
    logits = torch.randn(
        4,
        7,
    )
    weights = torch.randn(
        7,
        16,
    )

    first, second = top_two(logits)

    evidence, attention = model(
        patches,
        first,
        second,
        weights,
    )

    corrected = correct_top_two(
        logits,
        first,
        second,
        evidence,
        model.beta,
    )

    assert evidence.shape == (4,)
    assert attention.shape == (4, 12)
    assert torch.allclose(
        attention.sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )

    delta = corrected - logits

    for row in range(4):
        changed = set(
            torch.nonzero(
                delta[row].abs() > 1e-8
            ).flatten().tolist()
        )

        assert changed.issubset(
            {
                int(first[row]),
                int(second[row]),
            }
        )

    print("self-test: passed")


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "exactly one visible GPU is required"
        )

    device = torch.device("cuda:0")
    seed_everything(0)

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = output / "result.json"

    if (
        args.resume
        and result_path.is_file()
    ):
        value = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )
        print(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
            )
        )
        return

    baseline, baseline_config, baseline_result, baseline_path = (
        load_matched_baseline(device)
    )

    records = read_manifest(
        baseline_config["dataset"]["manifest"],
        dataset_root=(
            baseline_config["dataset"]["root"]
        ),
        check_images=True,
    )

    labelled_records = [
        record
        for record in records
        if (
            record["train_or_test_split"]
            == "train"
            and record[
                "labelled_or_unlabelled"
            ]
            == "labelled"
        )
    ]

    training_records = [
        record
        for record in labelled_records
        if not stable_validation_sample(
            str(record["sample_id"])
        )
    ]
    validation_records = [
        record
        for record in labelled_records
        if stable_validation_sample(
            str(record["sample_id"])
        )
    ]

    if args.limit_train:
        training_records = (
            training_records[
                : args.limit_train
            ]
        )

    if not training_records:
        raise RuntimeError(
            "empty evidence training split"
        )
    if not validation_records:
        raise RuntimeError(
            "empty evidence validation split"
        )

    training_dataset = (
        ProductionManifestDataset(
            training_records,
            baseline_config[
                "dataset"
            ]["root"],
            split="train",
            seed=0,
            train=True,
        )
    )

    validation_dataset = (
        ProductionManifestDataset(
            validation_records,
            baseline_config[
                "dataset"
            ]["root"],
            split="train",
            seed=0,
            train=False,
        )
    )

    test_dataset = (
        ProductionManifestDataset(
            records,
            baseline_config[
                "dataset"
            ]["root"],
            split="test",
            seed=0,
            train=False,
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        worker_init_fn=_worker_seed,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        worker_init_fn=_worker_seed,
    )

    configuration = EvidenceConfig(
        epochs=args.epochs,
        num_workers=4,
    )

    evidence_model = ConfusionEvidence(
        configuration
    ).to(device)

    started = time.perf_counter()

    training_report = train_evidence_model(
        baseline=baseline,
        evidence_model=evidence_model,
        train_dataset=training_dataset,
        validation_loader=validation_loader,
        device=device,
        config=configuration,
        output=output,
    )

    test = evaluate_test(
        baseline=baseline,
        evidence_model=evidence_model,
        loader=test_loader,
        device=device,
    )

    expected = baseline_result["metrics"]

    if not args.smoke:
        for name in ("all", "old", "new"):
            if abs(
                float(test["baseline"][name])
                - float(expected[name])
            ) > 1e-12:
                raise RuntimeError(
                    "recomputed matched baseline "
                    f"differs for {name}"
                )

    interval = bootstrap_hmean_delta(
        targets=test["targets"],
        old=test["old"],
        baseline=(
            test["baseline_predictions"]
        ),
        candidate=(
            test["corrected_predictions"]
        ),
        draws=args.bootstrap_draws,
        seed=92_117,
    )

    delta = test["delta"]

    follow = (
        delta["all"] >= 0.0
        and delta["new"] >= 0.005
        and delta["hmean"] >= 0.005
        and interval["lower_95"] > 0.0
    )

    verdict = (
        "follow"
        if follow
        else "abandon"
    )

    np.savez_compressed(
        output / "predictions.npz",
        target=test["targets"],
        old=test["old"],
        baseline_prediction=(
            test["baseline_predictions"]
        ),
        corrected_prediction=(
            test["corrected_predictions"]
        ),
    )

    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "dataset": "cub",
        "seed": 0,
        "method": (
            "confusion_conditioned_evidence"
        ),
        "oracle_used": False,
        "test_labels_used_for_training": False,
        "test_labels_used_for_selection": False,
        "patch_subdivision": False,
        "feature_injection": False,
        "extra_transformer_tokens": False,
        "baseline": test["baseline"],
        "candidate": test["corrected"],
        "delta": delta,
        "hmean_paired_bootstrap": interval,
        "diagnostics": test["diagnostics"],
        "training": training_report,
        "configuration": asdict(
            configuration
        ),
        "baseline_provenance": {
            "result": str(baseline_path),
            "checkpoint": (
                training_report.get(
                    "baseline_checkpoint"
                )
            ),
        },
        "training_samples": len(
            training_dataset
        ),
        "validation_samples": len(
            validation_dataset
        ),
        "test_samples": len(
            test_dataset
        ),
        "runtime_seconds": (
            time.perf_counter()
            - started
        ),
        "gate_rule": {
            "all_delta_minimum": 0.0,
            "new_delta_minimum": 0.005,
            "hmean_delta_minimum": 0.005,
            "hmean_lower_95_above_zero": True,
        },
        "verdict": verdict,
    }

    _atomic_json(
        result_path,
        result,
    )

    print()
    print("===== CONFUSION EVIDENCE GATE =====")
    print(
        "method                    "
        "all       old       new      hmean"
    )

    for name, values in (
        (
            "matched_selex",
            test["baseline"],
        ),
        (
            "confusion_evidence",
            test["corrected"],
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
        "changed predictions: "
        f"{test['diagnostics']['changed_count']} "
        f"({test['diagnostics']['changed_fraction']:.4%})"
    )
    print(
        "selected validation delta: "
        f"{training_report['best_validation']['delta']:+.6f}"
    )
    print(
        "beta: "
        f"{test['diagnostics']['beta']:.6f}"
    )
    print(
        "temperature: "
        f"{test['diagnostics']['temperature']:.6f}"
    )
    print()
    print(
        "DELTASUB VERDICT: "
        + verdict.upper()
    )
    print(f"result: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default=(
            "artifacts/deltasub_confusion/"
            "cub/seed_0"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run(args)


if __name__ == "__main__":
    main()
