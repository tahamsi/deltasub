"""Known-class supervised utility routing for DeltaSub.

The frozen fully supervised DeltaSub model supplies exact single-patch
utility targets. Only genuinely labelled known-class training samples
are used to train the router. Novel labels enter post-training analysis
only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from ..data.manifests import read_manifest
from ..evaluation.gcd_v2 import (
    evaluate_gcd_v2,
    provenance as gcd_provenance,
)
from ..models.deltasub_v2 import DeltaSubV2
from ..router.model import GainRouter, RouterConfig
from ..utils.checkpointing import (
    atomic_torch_save,
    load_checkpoint,
)
from ..utils.hashing import sha256_file, stable_hash
from ..utils.reproducibility import seed_everything
from .deltasub_v2_training import (
    ProductionManifestDataset,
    _atomic_json,
    _with_hmean,
    _worker_seed,
    construct_model,
    load_config,
)
from .supervised_oracle_routing import (
    base_parent_residuals,
    deterministic_random_scores,
    direct_metrics,
    encode_uniform_mask,
    exact_single_patch_utility,
    paired_bootstrap_delta,
    topk_mask,
)


SCHEMA = "deltasub-v2.old-label-utility-router.v1"
PATCH_COUNT = 256


class CachedUtilityDataset(Dataset):
    """Sample-level router training records."""

    def __init__(
        self,
        features: torch.Tensor,
        utilities: torch.Tensor,
        indices: list[int],
    ) -> None:
        if features.ndim != 3:
            raise ValueError(
                "features must have shape [N, 256, D]"
            )
        if utilities.shape != features.shape[:2]:
            raise ValueError(
                "utility shape must match feature samples and patches"
            )

        self.features = features
        self.utilities = utilities
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = self.indices[index]

        return (
            self.features[selected],
            self.utilities[selected],
        )


def stable_split(
    sample_ids: list[str],
    *,
    seed: int,
    validation_fraction: float = 0.20,
) -> tuple[list[int], list[int]]:
    """Deterministically split by sample identity."""

    if len(sample_ids) < 2:
        raise ValueError(
            "at least two samples are required"
        )
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be in (0, 1)"
        )

    ranked = sorted(
        range(len(sample_ids)),
        key=lambda index: hashlib.sha256(
            (
                f"{seed}:router-split:"
                f"{sample_ids[index]}"
            ).encode("utf-8")
        ).hexdigest(),
    )

    validation_count = max(
        1,
        min(
            len(ranked) - 1,
            int(
                round(
                    validation_fraction
                    * len(ranked)
                )
            ),
        ),
    )

    validation = ranked[:validation_count]
    training = ranked[validation_count:]

    if set(training) & set(validation):
        raise RuntimeError(
            "training and validation splits overlap"
        )

    return training, validation


def sanitize_utility(
    utility: torch.Tensor,
) -> torch.Tensor:
    """Replace non-candidate values used by smoke tests."""

    if utility.ndim != 2:
        raise ValueError(
            "utility must have shape [B, 256]"
        )

    finite = torch.isfinite(utility)

    if bool(finite.all()):
        return utility.float()

    finite_values = torch.where(
        finite,
        utility.float(),
        torch.full_like(
            utility.float(),
            float("inf"),
        ),
    )

    floor = (
        finite_values.amin(dim=1, keepdim=True)
        - 1.0
    )

    return torch.where(
        finite,
        utility.float(),
        floor,
    )


def utility_router_loss(
    scores: torch.Tensor,
    utility: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Top-one, listwise, and normalized regression supervision."""

    if scores.shape != utility.shape:
        raise ValueError(
            "scores and utility must have identical shape"
        )

    scores = scores.float()
    utility = sanitize_utility(utility)

    best_patch = utility.argmax(dim=1)

    top_one = F.cross_entropy(
        scores,
        best_patch,
    )

    target_distribution = F.softmax(
        utility / 0.10,
        dim=1,
    )

    listwise = -(
        target_distribution
        * F.log_softmax(scores, dim=1)
    ).sum(dim=1).mean()

    centered_utility = (
        utility
        - utility.mean(dim=1, keepdim=True)
    )
    utility_scale = centered_utility.std(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-5)

    normalized_utility = (
        centered_utility
        / utility_scale
    ).clamp(-5.0, 5.0)

    centered_scores = (
        scores
        - scores.mean(dim=1, keepdim=True)
    )
    score_scale = centered_scores.std(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-5)

    normalized_scores = (
        centered_scores
        / score_scale
    ).clamp(-5.0, 5.0)

    regression = F.smooth_l1_loss(
        normalized_scores,
        normalized_utility,
    )

    loss = (
        top_one
        + 0.50 * listwise
        + 0.10 * regression
    )

    return (
        loss,
        {
            "top_one": float(top_one.detach()),
            "listwise": float(listwise.detach()),
            "regression": float(
                regression.detach()
            ),
            "total": float(loss.detach()),
        },
    )


@torch.inference_mode()
def router_validation(
    router: GainRouter,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate utility selection without classifier labels."""

    router.eval()

    selected_gains: list[float] = []
    oracle_gains: list[float] = []
    top_one_matches: list[float] = []
    top_four_matches: list[float] = []
    positive_selections: list[float] = []

    for features, utility in loader:
        features = features.to(
            device,
            non_blocking=True,
        )
        utility = sanitize_utility(
            utility.to(
                device,
                non_blocking=True,
            )
        )

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            scores = router(features)

        predicted = scores.float().argmax(dim=1)
        oracle = utility.argmax(dim=1)

        selected = utility.gather(
            1,
            predicted[:, None],
        ).squeeze(1)

        oracle_gain = utility.gather(
            1,
            oracle[:, None],
        ).squeeze(1)

        top_four = torch.argsort(
            utility,
            dim=1,
            descending=True,
            stable=True,
        )[:, :4]

        selected_gains.extend(
            selected.cpu().tolist()
        )
        oracle_gains.extend(
            oracle_gain.cpu().tolist()
        )
        top_one_matches.extend(
            (predicted == oracle)
            .float()
            .cpu()
            .tolist()
        )
        top_four_matches.extend(
            (
                top_four
                == predicted[:, None]
            )
            .any(dim=1)
            .float()
            .cpu()
            .tolist()
        )
        positive_selections.extend(
            (selected > 0)
            .float()
            .cpu()
            .tolist()
        )

    selected_array = np.asarray(
        selected_gains,
        dtype=np.float64,
    )
    oracle_array = np.asarray(
        oracle_gains,
        dtype=np.float64,
    )

    return {
        "selected_gain": float(
            selected_array.mean()
        ),
        "oracle_gain": float(
            oracle_array.mean()
        ),
        "gain_regret": float(
            (
                oracle_array
                - selected_array
            ).mean()
        ),
        "top1_recall": float(
            np.mean(top_one_matches)
        ),
        "top4_recall": float(
            np.mean(top_four_matches)
        ),
        "positive_selection_rate": float(
            np.mean(positive_selections)
        ),
    }


def eligible_training_indices(
    dataset: ProductionManifestDataset,
) -> list[int]:
    """Return only genuinely labelled known-class samples."""

    indices = [
        index
        for index, record in enumerate(
            dataset.records
        )
        if (
            record[
                "labelled_or_unlabelled"
            ]
            == "labelled"
            and record[
                "known_or_novel"
            ]
            == "known"
        )
    ]

    if not indices:
        raise RuntimeError(
            "no labelled known-class training samples"
        )

    return indices


@torch.inference_mode()
def build_training_cache(
    *,
    model: DeltaSubV2,
    dataset: ProductionManifestDataset,
    eligible_indices: list[int],
    device: torch.device,
    output_path: Path,
    checkpoint_path: Path,
    config: dict[str, Any],
    batch_size: int,
    candidate_chunk: int,
    candidate_count: int,
) -> dict[str, Any]:
    """Build exact utility targets for known labelled samples."""

    if output_path.is_file():
        value = load_checkpoint(
            output_path,
            map_location="cpu",
        )

        required = {
            "schema_version",
            "features",
            "utilities",
            "sample_ids",
            "targets",
            "checkpoint_sha256",
            "candidate_count",
        }

        if required - set(value):
            raise ValueError(
                "router cache is incomplete"
            )

        if (
            value["checkpoint_sha256"]
            != sha256_file(checkpoint_path)
        ):
            raise ValueError(
                "router cache checkpoint mismatch"
            )

        if (
            int(value["candidate_count"])
            != candidate_count
        ):
            raise ValueError(
                "router cache candidate mismatch"
            )

        return value

    model.eval()
    model.requires_grad_(False)

    subset = Subset(
        dataset,
        eligible_indices,
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(
            config["training"]["num_workers"]
        ),
        worker_init_fn=_worker_seed,
        persistent_workers=(
            int(
                config[
                    "training"
                ]["num_workers"]
            )
            > 0
        ),
    )

    features_all: list[torch.Tensor] = []
    utilities_all: list[torch.Tensor] = []
    sample_ids: list[str] = []
    targets_all: list[int] = []

    started = time.perf_counter()
    processed = 0
    next_report = 128

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            trunk_tokens, parents = (
                model._run_trunk(images)
            )
            global_features = (
                model._tail_encode(
                    trunk_tokens
                )
            )
            global_logits = model.head(
                global_features
            )

            raw_details = model._haar_details(
                images,
                parents,
            )
            adapted_details = (
                model.detail_adapter(
                    raw_details
                )
            )
            residuals = (
                base_parent_residuals(
                    model,
                    trunk_tokens,
                    adapted_details,
                )
            )

        utilities = exact_single_patch_utility(
            model=model,
            trunk_tokens=trunk_tokens,
            residuals=residuals,
            targets=targets,
            global_logits=global_logits,
            candidate_chunk=candidate_chunk,
            candidate_count=candidate_count,
        )

        prefix_count = int(
            model.backbone.prefix_token_count
        )

        router_features = trunk_tokens[
            :,
            prefix_count : (
                prefix_count + PATCH_COUNT
            ),
        ]

        features_all.append(
            router_features.detach()
            .to(
                device="cpu",
                dtype=torch.float16,
            )
        )
        utilities_all.append(
            utilities.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
        )

        sample_ids.extend(
            str(value)
            for value in batch["sample_id"]
        )
        targets_all.extend(
            targets.cpu().tolist()
        )

        processed += images.shape[0]

        if processed >= next_report:
            elapsed = (
                time.perf_counter()
                - started
            )

            print(
                "cache "
                f"processed={processed} "
                f"elapsed_min="
                f"{elapsed / 60:.1f} "
                f"samples_per_second="
                f"{processed / max(elapsed, 1e-12):.3f}",
                flush=True,
            )

            next_report += 128

    cache = {
        "schema_version": SCHEMA,
        "features": torch.cat(
            features_all,
            dim=0,
        ),
        "utilities": torch.cat(
            utilities_all,
            dim=0,
        ),
        "sample_ids": sample_ids,
        "targets": targets_all,
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
        "manifest_sha256": sha256_file(
            config["dataset"]["manifest"]
        ),
        "candidate_count": candidate_count,
        "sample_count": len(sample_ids),
        "training_label_policy": (
            "genuinely labelled known-class "
            "training samples only"
        ),
    }

    atomic_torch_save(
        cache,
        output_path,
    )

    print(
        "cache completed: "
        f"{len(sample_ids)} samples",
        flush=True,
    )

    return cache


def train_router(
    *,
    cache: dict[str, Any],
    output_directory: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    resume: bool,
) -> tuple[GainRouter, dict[str, Any]]:
    """Train and validation-select the known-class router."""

    features = cache["features"]
    utilities = cache["utilities"]
    sample_ids = [
        str(value)
        for value in cache["sample_ids"]
    ]

    training_indices, validation_indices = (
        stable_split(
            sample_ids,
            seed=seed,
        )
    )

    train_dataset = CachedUtilityDataset(
        features,
        utilities,
        training_indices,
    )
    validation_dataset = (
        CachedUtilityDataset(
            features,
            utilities,
            validation_indices,
        )
    )

    generator = torch.Generator().manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    router_config = RouterConfig(
        input_dim=768,
        hidden_dim=256,
        depth=2,
        normalization="layernorm",
        dropout=0.0,
        coordinate_features=True,
        global_context_features=True,
        learned_position_embedding=False,
    )

    router = GainRouter(
        router_config,
        seed=seed,
    ).to(device)

    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=1.0e-4,
        weight_decay=1.0e-4,
    )
    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            max(1, epochs),
        )
    )

    checkpoint_last = (
        output_directory
        / "router_checkpoint_last.pt"
    )
    checkpoint_best = (
        output_directory
        / "router_checkpoint_best.pt"
    )
    metrics_path = (
        output_directory
        / "router_metrics.jsonl"
    )

    start_epoch = 0
    best_selected_gain = -float("inf")
    best_validation: dict[str, float] = {}

    if resume and checkpoint_last.is_file():
        state = load_checkpoint(
            checkpoint_last,
            map_location=device,
        )

        expected_hash = stable_hash(
            {
                "sample_ids": sample_ids,
                "router": router_config.__dict__,
                "seed": seed,
            }
        )

        if (
            state["training_identity_hash"]
            != expected_hash
        ):
            raise ValueError(
                "router resume identity mismatch"
            )

        router.load_state_dict(
            state["router_state"],
            strict=True,
        )
        optimizer.load_state_dict(
            state["optimizer_state"]
        )
        scheduler.load_state_dict(
            state["scheduler_state"]
        )

        start_epoch = int(state["epoch"])
        best_selected_gain = float(
            state["best_selected_gain"]
        )
        best_validation = dict(
            state["best_validation"]
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

    training_identity_hash = stable_hash(
        {
            "sample_ids": sample_ids,
            "router": router_config.__dict__,
            "seed": seed,
        }
    )

    for epoch in range(
        start_epoch,
        epochs,
    ):
        router.train()

        sums = {
            "top_one": 0.0,
            "listwise": 0.0,
            "regression": 0.0,
            "total": 0.0,
        }
        batches = 0

        for features_batch, utility_batch in train_loader:
            features_batch = (
                features_batch.to(
                    device,
                    non_blocking=True,
                )
            )
            utility_batch = (
                utility_batch.to(
                    device,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                scores = router(
                    features_batch
                )

            loss, parts = utility_router_loss(
                scores,
                utility_batch,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite router loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                router.parameters(),
                1.0,
            )

            optimizer.step()

            for name, value in parts.items():
                sums[name] += value

            batches += 1

        scheduler.step()

        validation = router_validation(
            router,
            validation_loader,
            device,
        )

        means = {
            name: value / max(
                batches,
                1,
            )
            for name, value in sums.items()
        }

        record = {
            "epoch": epoch + 1,
            "learning_rate": (
                optimizer.param_groups[0][
                    "lr"
                ]
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
            stream.flush()

        selected_gain = float(
            validation["selected_gain"]
        )

        state = {
            "schema_version": SCHEMA,
            "router_state": (
                router.state_dict()
            ),
            "optimizer_state": (
                optimizer.state_dict()
            ),
            "scheduler_state": (
                scheduler.state_dict()
            ),
            "router_config": (
                router_config.__dict__
            ),
            "epoch": epoch + 1,
            "best_selected_gain": max(
                best_selected_gain,
                selected_gain,
            ),
            "best_validation": (
                validation
                if selected_gain
                > best_selected_gain
                else best_validation
            ),
            "training_identity_hash": (
                training_identity_hash
            ),
            "training_indices": (
                training_indices
            ),
            "validation_indices": (
                validation_indices
            ),
            "training_sample_ids": [
                sample_ids[index]
                for index in training_indices
            ],
            "validation_sample_ids": [
                sample_ids[index]
                for index in validation_indices
            ],
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
        }

        atomic_torch_save(
            state,
            checkpoint_last,
        )

        if selected_gain > best_selected_gain:
            best_selected_gain = (
                selected_gain
            )
            best_validation = validation

            atomic_torch_save(
                state,
                checkpoint_best,
            )

        print(
            f"router epoch={epoch + 1:03d}/"
            f"{epochs} "
            f"loss={means['total']:.6f} "
            f"val_gain="
            f"{validation['selected_gain']:.6f} "
            f"top1="
            f"{validation['top1_recall']:.4f} "
            f"top4="
            f"{validation['top4_recall']:.4f} "
            f"positive="
            f"{validation['positive_selection_rate']:.4f}",
            flush=True,
        )

    if not checkpoint_best.is_file():
        raise RuntimeError(
            "router best checkpoint was not created"
        )

    best_state = load_checkpoint(
        checkpoint_best,
        map_location=device,
    )
    router.load_state_dict(
        best_state["router_state"],
        strict=True,
    )
    router.eval()

    summary = {
        "training_samples": len(
            training_indices
        ),
        "validation_samples": len(
            validation_indices
        ),
        "best_epoch": int(
            best_state["epoch"]
        ),
        "best_validation": (
            best_state["best_validation"]
        ),
        "router_parameters": (
            router.parameter_count
        ),
        "checkpoint": str(
            checkpoint_best
        ),
        "checkpoint_sha256": (
            sha256_file(checkpoint_best)
        ),
    }

    return router, summary


def partition_routing_metrics(
    *,
    selected_indices: np.ndarray,
    utility: np.ndarray,
    old: np.ndarray,
) -> dict[str, Any]:
    """Utility-ranking transfer metrics by Old/New partition."""

    oracle_indices = utility.argmax(axis=1)

    selected_gain = utility[
        np.arange(len(utility)),
        selected_indices,
    ]
    oracle_gain = utility[
        np.arange(len(utility)),
        oracle_indices,
    ]

    top_four = np.argsort(
        -utility,
        axis=1,
        kind="stable",
    )[:, :4]

    result: dict[str, Any] = {}

    for name, mask in (
        (
            "all",
            np.ones_like(
                old,
                dtype=bool,
            ),
        ),
        ("old", old),
        ("new", ~old),
    ):
        partition_selected = (
            selected_indices[mask]
        )
        partition_oracle = (
            oracle_indices[mask]
        )

        partition_top_four = (
            top_four[mask]
        )

        result[name] = {
            "count": int(mask.sum()),
            "top1_recall": float(
                np.mean(
                    partition_selected
                    == partition_oracle
                )
            ),
            "top4_recall": float(
                np.mean(
                    (
                        partition_top_four
                        == partition_selected[
                            :, None
                        ]
                    ).any(axis=1)
                )
            ),
            "selected_gain": float(
                selected_gain[mask].mean()
            ),
            "oracle_gain": float(
                oracle_gain[mask].mean()
            ),
            "gain_regret": float(
                (
                    oracle_gain[mask]
                    - selected_gain[mask]
                ).mean()
            ),
            "positive_selection_rate": float(
                np.mean(
                    selected_gain[mask] > 0
                )
            ),
        }

    return result


@torch.inference_mode()
def evaluate_transfer(
    *,
    model: DeltaSubV2,
    router: GainRouter,
    dataset: ProductionManifestDataset,
    oracle_predictions_path: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    bootstrap_draws: int,
    smoke: bool,
    output_directory: Path,
) -> dict[str, Any]:
    """Evaluate K=1 known-label routing on Old and New classes."""

    archive = np.load(
        oracle_predictions_path,
        allow_pickle=False,
    )

    oracle_sample_ids = archive[
        "sample_id"
    ].astype(str)
    utility = archive["utility"].astype(
        np.float32
    )
    targets = archive["target"].astype(
        np.int64
    )
    old = archive["old"].astype(bool)

    evaluation_indices = list(
        range(len(dataset))
    )

    if smoke:
        old_index = int(
            np.flatnonzero(old)[0]
        )
        new_index = int(
            np.flatnonzero(~old)[0]
        )
        evaluation_indices = [
            old_index,
            new_index,
        ]

    subset = Subset(
        dataset,
        evaluation_indices,
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=(
            0
            if smoke
            else 4
        ),
        worker_init_fn=_worker_seed,
        persistent_workers=not smoke,
    )

    model.eval()
    model.requires_grad_(False)
    router.eval()
    router.requires_grad_(False)

    predictions: list[int] = []
    selected_indices: list[int] = []
    energy_indices: list[int] = []
    random_indices: list[int] = []
    sample_ids: list[str] = []

    started = time.perf_counter()

    for batch in loader:
        images = batch["image"].to(device)
        batch_sample_ids = [
            str(value)
            for value in batch["sample_id"]
        ]

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            trunk_tokens, parents = (
                model._run_trunk(images)
            )

            raw_details = model._haar_details(
                images,
                parents,
            )
            adapted_details = (
                model.detail_adapter(
                    raw_details
                )
            )

            residuals = (
                base_parent_residuals(
                    model,
                    trunk_tokens,
                    adapted_details,
                )
            )

            prefix_count = int(
                model.backbone
                .prefix_token_count
            )

            router_features = trunk_tokens[
                :,
                prefix_count : (
                    prefix_count
                    + PATCH_COUNT
                ),
            ]

            router_scores = router(
                router_features
            )
            energy_scores = (
                model._routing_scores(
                    trunk_tokens,
                    adapted_details,
                )
            )

        router_mask = topk_mask(
            router_scores.float(),
            1,
        )

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            features = encode_uniform_mask(
                model,
                trunk_tokens,
                residuals,
                router_mask,
            )
            logits = model.head(features)

        random_scores = (
            deterministic_random_scores(
                batch_sample_ids,
                seed=seed,
                device=device,
            )
        )

        predictions.extend(
            logits.float()
            .argmax(dim=1)
            .cpu()
            .tolist()
        )
        selected_indices.extend(
            router_scores.float()
            .argmax(dim=1)
            .cpu()
            .tolist()
        )
        energy_indices.extend(
            energy_scores.float()
            .argmax(dim=1)
            .cpu()
            .tolist()
        )
        random_indices.extend(
            random_scores.argmax(dim=1)
            .cpu()
            .tolist()
        )
        sample_ids.extend(
            batch_sample_ids
        )

    selected = np.asarray(
        evaluation_indices,
        dtype=np.int64,
    )

    expected_ids = (
        oracle_sample_ids[selected]
    )

    if not np.array_equal(
        np.asarray(sample_ids),
        expected_ids,
    ):
        raise RuntimeError(
            "oracle and router evaluation "
            "sample ordering differ"
        )

    target_selected = targets[selected]
    old_selected = old[selected]
    utility_selected = utility[selected]
    router_prediction = np.asarray(
        predictions,
        dtype=np.int64,
    )

    control_prediction_keys = {
        "global": "prediction_global",
        "trained_adaptive": (
            "prediction_trained_adaptive"
        ),
        "energy_k1": (
            "prediction_energy_k1"
        ),
        "random_k1": (
            "prediction_random_k1"
        ),
        "oracle_k1": (
            "prediction_oracle_k1"
        ),
    }

    prediction_arrays = {
        name: archive[key][selected].astype(
            np.int64
        )
        for name, key in (
            control_prediction_keys.items()
        )
    }
    prediction_arrays[
        "old_label_router_k1"
    ] = router_prediction

    metrics: dict[str, Any] = {}

    for name, prediction in (
        prediction_arrays.items()
    ):
        metrics[name] = {
            "direct": direct_metrics(
                target_selected,
                prediction,
                old_selected,
            ),
            "gcd_v2": _with_hmean(
                evaluate_gcd_v2(
                    target_selected,
                    prediction,
                    old_selected,
                ).as_dict()
            ),
        }

    router_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )
    energy_index_array = np.asarray(
        energy_indices,
        dtype=np.int64,
    )
    random_index_array = np.asarray(
        random_indices,
        dtype=np.int64,
    )

    ranking = {
        "old_label_router_k1": (
            partition_routing_metrics(
                selected_indices=router_indices,
                utility=utility_selected,
                old=old_selected,
            )
        ),
        "energy_k1": (
            partition_routing_metrics(
                selected_indices=(
                    energy_index_array
                ),
                utility=utility_selected,
                old=old_selected,
            )
        ),
        "random_k1": (
            partition_routing_metrics(
                selected_indices=(
                    random_index_array
                ),
                utility=utility_selected,
                old=old_selected,
            )
        ),
    }

    bootstrap = paired_bootstrap_delta(
        targets=target_selected,
        old=old_selected,
        baseline=prediction_arrays[
            "global"
        ],
        candidate=router_prediction,
        draws=bootstrap_draws,
        seed=seed + 30_001,
    )

    baseline = metrics["global"]["direct"]
    candidate = metrics[
        "old_label_router_k1"
    ]["direct"]

    delta = {
        metric: (
            float(candidate[metric])
            - float(baseline[metric])
        )
        for metric in (
            "all",
            "old",
            "new",
            "hmean",
        )
    }

    novel_router_recall = ranking[
        "old_label_router_k1"
    ]["new"]["top1_recall"]

    novel_random_recall = ranking[
        "random_k1"
    ]["new"]["top1_recall"]

    gate_passed = (
        delta["all"] >= 0.0
        and delta["new"] >= 0.005
        and delta["hmean"] >= 0.005
        and bootstrap["hmean"][
            "lower_95"
        ] > 0.0
        and novel_router_recall
        >= novel_random_recall + 0.01
    )

    np.savez_compressed(
        output_directory
        / "transfer_predictions.npz",
        sample_id=np.asarray(sample_ids),
        target=target_selected,
        old=old_selected,
        prediction_router=(
            router_prediction
        ),
        selected_patch=router_indices,
        energy_patch=energy_index_array,
        random_patch=random_index_array,
    )

    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "training_label_policy": (
            "genuinely labelled known-class "
            "training samples only"
        ),
        "novel_label_policy": (
            "novel labels used only after "
            "router training for analysis"
        ),
        "selection_budget": 1,
        "metrics": metrics,
        "routing_transfer": ranking,
        "paired_bootstrap_delta_vs_global": (
            bootstrap
        ),
        "delta_vs_global": delta,
        "transfer_gate": (
            "passed"
            if gate_passed
            else "failed"
        ),
        "gate_rule": {
            "all_delta_minimum": 0.0,
            "new_delta_minimum": 0.005,
            "hmean_delta_minimum": 0.005,
            "hmean_bootstrap_lower_95_above_zero": True,
            "novel_top1_recall_margin_over_random": 0.01,
        },
        "evaluation_samples": int(
            len(target_selected)
        ),
        "runtime_seconds": (
            time.perf_counter()
            - started
        ),
    }

    print()
    print("===== OLD-LABEL ROUTER TRANSFER =====")
    print(
        "method                 "
        "all       old       new      hmean"
    )

    order = (
        "global",
        "trained_adaptive",
        "energy_k1",
        "random_k1",
        "old_label_router_k1",
        "oracle_k1",
    )

    for name in order:
        values = metrics[name]["direct"]

        print(
            f"{name:<22}"
            f"{float(values['all']):>9.6f}"
            f"{float(values['old']):>10.6f}"
            f"{float(values['new']):>10.6f}"
            f"{float(values['hmean']):>11.6f}"
        )

    print()
    print("===== ROUTING TRANSFER =====")

    for method in (
        "old_label_router_k1",
        "energy_k1",
        "random_k1",
    ):
        print(method)

        for partition in (
            "all",
            "old",
            "new",
        ):
            values = ranking[
                method
            ][partition]

            print(
                f"  {partition:<3} "
                f"top1={values['top1_recall']:.4f} "
                f"top4={values['top4_recall']:.4f} "
                f"gain={values['selected_gain']:.6f} "
                f"positive="
                f"{values['positive_selection_rate']:.4f}"
            )

    print()
    print("===== TRANSFER GATE =====")

    for metric, value in delta.items():
        print(
            f"{metric:<5}: {value:+.6f}"
        )

    interval = bootstrap["hmean"]

    print(
        "hmean paired-bootstrap 95% CI: "
        f"[{interval['lower_95']:+.6f}, "
        f"{interval['upper_95']:+.6f}]"
    )
    print(
        "transfer gate: "
        + (
            "PASSED"
            if gate_passed
            else "FAILED"
        )
    )

    return result


def run(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    oracle_predictions_path: str | Path,
    output_directory: str | Path,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    """Run known-label utility-router training and transfer evaluation."""

    config = load_config(config_path)

    if config["method"]["variant"] != "deltasub_v2":
        raise ValueError(
            "known-label router requires DeltaSub"
        )

    if (
        config["training"].get(
            "supervision_mode",
            "gcd",
        )
        != "fully_supervised"
    ):
        raise ValueError(
            "known-label router requires the "
            "fully supervised checkpoint"
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
    torch.cuda.reset_peak_memory_stats()

    checkpoint_path = Path(
        checkpoint_path
    )
    oracle_predictions_path = Path(
        oracle_predictions_path
    )
    output = Path(output_directory)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = output / "result.json"

    if (
        resume
        and result_path.is_file()
        and not smoke
    ):
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

    state = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    model = construct_model(
        config,
        device=device,
    )

    if not isinstance(model, DeltaSubV2):
        raise TypeError(
            "constructed model is not DeltaSubV2"
        )

    model.load_state_dict(
        state["model"],
        strict=True,
    )
    model.eval()
    model.requires_grad_(False)

    records = read_manifest(
        config["dataset"]["manifest"],
        dataset_root=(
            config["dataset"]["root"]
        ),
        check_images=True,
    )

    train_dataset = ProductionManifestDataset(
        records,
        config["dataset"]["root"],
        split="train",
        seed=seed,
        train=False,
    )

    test_dataset = ProductionManifestDataset(
        records,
        config["dataset"]["root"],
        split="test",
        seed=seed,
        train=False,
    )

    eligible = eligible_training_indices(
        train_dataset
    )

    candidate_count = PATCH_COUNT
    epochs = 25
    router_batch_size = 16
    cache_batch_size = 4
    candidate_chunk = 128
    bootstrap_draws = 2000

    if smoke:
        eligible = eligible[:12]
        candidate_count = 16
        epochs = 2
        router_batch_size = 4
        cache_batch_size = 2
        candidate_chunk = 8
        bootstrap_draws = 20

    cache_path = (
        output / "known_label_utility_cache.pt"
    )

    cache = build_training_cache(
        model=model,
        dataset=train_dataset,
        eligible_indices=eligible,
        device=device,
        output_path=cache_path,
        checkpoint_path=checkpoint_path,
        config=config,
        batch_size=cache_batch_size,
        candidate_chunk=candidate_chunk,
        candidate_count=candidate_count,
    )

    router, training_summary = train_router(
        cache=cache,
        output_directory=output,
        device=device,
        seed=seed,
        epochs=epochs,
        batch_size=router_batch_size,
        resume=resume,
    )

    transfer = evaluate_transfer(
        model=model,
        router=router,
        dataset=test_dataset,
        oracle_predictions_path=(
            oracle_predictions_path
        ),
        device=device,
        seed=seed,
        batch_size=(
            2
            if smoke
            else 32
        ),
        bootstrap_draws=bootstrap_draws,
        smoke=smoke,
        output_directory=output,
    )

    provenance = {
        "schema_version": SCHEMA,
        "repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "repository_dirty": bool(
            subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(
            device
        ),
        "seed": seed,
        "supervised_checkpoint": str(
            checkpoint_path
        ),
        "supervised_checkpoint_sha256": (
            sha256_file(checkpoint_path)
        ),
        "oracle_predictions": str(
            oracle_predictions_path
        ),
        "oracle_predictions_sha256": (
            sha256_file(
                oracle_predictions_path
            )
        ),
        "manifest_sha256": sha256_file(
            config["dataset"]["manifest"]
        ),
        "split_sha256": sha256_file(
            config[
                "dataset"
            ]["split_validation"]
        ),
        "gcd": gcd_provenance(
            implementation_path=(
                Path(__file__).parents[1]
                / "evaluation"
                / "gcd_v2.py"
            )
        ),
        "smoke": smoke,
    }

    final_result = {
        **transfer,
        "training": training_summary,
        "cache": {
            "sample_count": int(
                cache["sample_count"]
            ),
            "candidate_count": int(
                cache["candidate_count"]
            ),
            "path": str(cache_path),
            "sha256": sha256_file(
                cache_path
            ),
        },
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated()
        ),
    }

    _atomic_json(
        output / "provenance.json",
        provenance,
    )
    _atomic_json(
        result_path,
        final_result,
    )

    print()
    print(
        json.dumps(
            {
                "status": (
                    final_result["status"]
                ),
                "transfer_gate": (
                    final_result[
                        "transfer_gate"
                    ]
                ),
                "training": (
                    training_summary
                ),
                "delta_vs_global": (
                    final_result[
                        "delta_vs_global"
                    ]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )

    return final_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--oracle-predictions",
        required=True,
    )
    parser.add_argument(
        "--output",
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

    run(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        oracle_predictions_path=(
            args.oracle_predictions
        ),
        output_directory=args.output,
        resume=args.resume,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
