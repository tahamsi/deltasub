#!/usr/bin/env python
"""Information-gain selection with appended DeltaSub tokens.

The experiment tests the original DeltaSub hypothesis:

1. retain every original DINOv2 token;
2. subdivide selected parent patches;
3. append their three parent-consistent Haar detail tokens;
4. select candidates using label-free predictive information gain.

No oracle, ground-truth test labels, residual injection, or learned router
is used during candidate selection.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
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

from deltasub.data.manifests import read_manifest
from deltasub.evaluation.gcd_v2 import evaluate_gcd_v2
from deltasub.experiment.deltasub_residual_compare import (
    bootstrap_hmean_delta,
    locate_matched_baseline,
)
from deltasub.experiment.deltasub_v2_training import (
    _branch_losses,
    construct_model,
)
from deltasub.experiment.training import (
    ProductionManifestDataset,
    _atomic_json,
    _worker_seed,
)
from deltasub.models.subtokens.geometry import (
    extract_parent_patches,
    subdivide_parent_patches,
)
from deltasub.models.subtokens.haar import HaarDetails
from deltasub.models.subtokens.positions import (
    ParentAwareDetailPositions,
)
from deltasub.models.subtokens.projection import (
    enforce_parent_consistency,
)
from deltasub.utils.checkpointing import (
    atomic_torch_save,
    load_checkpoint,
)
from deltasub.utils.hashing import sha256_file
from deltasub.utils.reproducibility import seed_everything


SCHEMA = "deltasub.information-gain-append.v1"
PATCH_COUNT = 256
DETAIL_MODES = 3
K_VALUES = (1, 4, 8, 16)


@dataclass(frozen=True)
class ExperimentConfig:
    insertion_block: int = 10
    epochs: int = 30
    physical_batch_size: int = 8
    gradient_accumulation: int = 16
    evaluation_batch_size: int = 8
    candidate_chunk_size: int = 32
    tail_learning_rate: float = 1.0e-5
    detail_learning_rate: float = 1.0e-3
    weight_decay: float = 0.05
    distillation_weight: float = 0.05
    gradient_clipping: float = 1.0
    num_workers: int = 4

    def validate(self) -> None:
        if self.insertion_block != 10:
            raise ValueError(
                "the matched protocol inserts before block 10"
            )
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if (
            self.physical_batch_size
            * self.gradient_accumulation
            != 128
        ):
            raise ValueError(
                "effective batch size must remain 128"
            )
        if self.evaluation_batch_size <= 0:
            raise ValueError(
                "evaluation_batch_size must be positive"
            )
        if not 1 <= self.candidate_chunk_size <= 256:
            raise ValueError(
                "candidate_chunk_size must be in [1, 256]"
            )
        if self.tail_learning_rate <= 0:
            raise ValueError(
                "tail_learning_rate must be positive"
            )
        if self.detail_learning_rate <= 0:
            raise ValueError(
                "detail_learning_rate must be positive"
            )
        if self.distillation_weight < 0:
            raise ValueError(
                "distillation_weight must be nonnegative"
            )


class AppendSubtokenModel(nn.Module):
    """Append selected Haar subtokens after the original token sequence."""

    def __init__(
        self,
        matched_model: nn.Module,
        *,
        insertion_block: int,
    ) -> None:
        super().__init__()

        self.backbone = matched_model.backbone
        self.head = matched_model.head
        self.insertion_block = int(insertion_block)

        blocks = list(self.backbone.model.blocks)

        if self.insertion_block >= len(blocks):
            raise ValueError(
                "insertion block must precede the final block"
            )

        # The matched global model remains completely frozen.
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(False)

        # A separately adapted copy receives original tokens plus subtokens.
        self.detail_tail = nn.ModuleList(
            deepcopy(blocks[self.insertion_block :])
        )
        self.detail_norm = deepcopy(
            self.backbone.model.norm
        )

        self.child = self.backbone.build_child_projector(
            trainable=True
        )
        self.positions = ParentAwareDetailPositions()
        self.haar = HaarDetails()

        self._apply_policy()

    @property
    def trunk_blocks(
        self,
    ) -> tuple[nn.Module, ...]:
        return tuple(
            self.backbone.model.blocks[
                : self.insertion_block
            ]
        )

    @property
    def original_tail(
        self,
    ) -> tuple[nn.Module, ...]:
        return tuple(
            self.backbone.model.blocks[
                self.insertion_block :
            ]
        )

    def _apply_policy(self) -> None:
        self.backbone.eval()
        self.head.eval()

        for block in self.detail_tail:
            block.train(self.training)

        self.detail_norm.train(self.training)
        self.child.train(self.training)
        self.positions.train(self.training)

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)
        self._apply_policy()
        return self

    def trunk_and_details(
        self,
        images: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if (
            images.ndim != 4
            or tuple(images.shape[1:])
            != (3, 224, 224)
        ):
            raise ValueError(
                "images must have shape [B, 3, 224, 224]"
            )

        batch = images.shape[0]

        with torch.no_grad():
            parents = (
                self.backbone
                .pre_transformer_parent_embeddings(images)
            )

            parent_positions = (
                self.backbone
                .parent_patch_positions()
                .to(parents)
                .expand(batch, -1, -1)
            )

            prefix = (
                self.backbone
                .prefix_tokens_with_positions(batch)
                .to(parents)
            )

            trunk_tokens = torch.cat(
                (
                    prefix,
                    parents + parent_positions,
                ),
                dim=1,
            )

            for block in self.trunk_blocks:
                trunk_tokens = block(trunk_tokens)

        children = self.child(
            subdivide_parent_patches(
                extract_parent_patches(images)
            )
        )

        consistent = enforce_parent_consistency(
            children,
            parents,
        ).consistent

        raw_details = self.haar(consistent)

        if raw_details.shape != (
            batch,
            PATCH_COUNT,
            DETAIL_MODES,
            768,
        ):
            raise RuntimeError(
                "unexpected Haar detail shape: "
                f"{tuple(raw_details.shape)}"
            )

        detail_positions = self.positions(
            parent_positions
        )

        detail_tokens = (
            raw_details + detail_positions
        )

        return (
            trunk_tokens,
            raw_details,
            detail_tokens,
        )

    def _detail_features(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.detail_tail:
            sequence = block(sequence)

        return self.detail_norm(sequence)[:, 0]

    @torch.no_grad()
    def global_logits_from_trunk(
        self,
        trunk_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sequence = trunk_tokens

        for block in self.original_tail:
            sequence = block(sequence)

        features = (
            self.backbone.model.norm(sequence)
            [:, 0]
        )

        return self.head(features)

    def encode_selected(
        self,
        trunk_tokens: torch.Tensor,
        detail_tokens: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if indices.ndim != 2:
            raise ValueError(
                "indices must have shape [B, K]"
            )

        batch, selected_count = indices.shape

        if batch != trunk_tokens.shape[0]:
            raise ValueError(
                "selection batch does not match tokens"
            )
        if selected_count <= 0:
            raise ValueError(
                "at least one parent must be selected"
            )

        gather_indices = indices[
            :,
            :,
            None,
            None,
        ].expand(
            -1,
            -1,
            DETAIL_MODES,
            detail_tokens.shape[-1],
        )

        selected = detail_tokens.gather(
            1,
            gather_indices,
        ).reshape(
            batch,
            selected_count * DETAIL_MODES,
            detail_tokens.shape[-1],
        )

        # Every original token is retained. Subtokens are appended.
        sequence = torch.cat(
            (
                trunk_tokens,
                selected,
            ),
            dim=1,
        )

        features = self._detail_features(
            sequence
        )

        return features, self.head(features)

    def single_patch_logits(
        self,
        trunk_tokens: torch.Tensor,
        detail_tokens: torch.Tensor,
        *,
        candidate_count: int,
        chunk_size: int,
    ) -> torch.Tensor:
        if not 1 <= candidate_count <= PATCH_COUNT:
            raise ValueError(
                "candidate_count must be in [1, 256]"
            )

        batch = trunk_tokens.shape[0]
        token_count = trunk_tokens.shape[1]
        width = trunk_tokens.shape[2]

        values: list[torch.Tensor] = []

        for start in range(
            0,
            candidate_count,
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                candidate_count,
            )
            count = end - start

            expanded_trunk = (
                trunk_tokens[:, None]
                .expand(
                    -1,
                    count,
                    -1,
                    -1,
                )
                .reshape(
                    batch * count,
                    token_count,
                    width,
                )
            )

            selected_details = (
                detail_tokens[
                    :,
                    start:end,
                ]
                .reshape(
                    batch * count,
                    DETAIL_MODES,
                    width,
                )
            )

            sequence = torch.cat(
                (
                    expanded_trunk,
                    selected_details,
                ),
                dim=1,
            )

            features = self._detail_features(
                sequence
            )

            logits = self.head(features).reshape(
                batch,
                count,
                -1,
            )

            values.append(logits)

        return torch.cat(values, dim=1)

    def append_state_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "detail_tail": (
                self.detail_tail.state_dict()
            ),
            "detail_norm": (
                self.detail_norm.state_dict()
            ),
            "child": self.child.state_dict(),
            "positions": (
                self.positions.state_dict()
            ),
        }

    def load_append_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        self.detail_tail.load_state_dict(
            state["detail_tail"],
            strict=True,
        )
        self.detail_norm.load_state_dict(
            state["detail_norm"],
            strict=True,
        )
        self.child.load_state_dict(
            state["child"],
            strict=True,
        )
        self.positions.load_state_dict(
            state["positions"],
            strict=True,
        )

    def parameter_report(
        self,
    ) -> dict[str, int]:
        total = sum(
            parameter.numel()
            for parameter in self.parameters()
        )
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }


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


def predictive_information_gain(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """KL(candidate prediction || original prediction).

    The candidate distribution is the posterior after appending one
    patch's subtokens. The original distribution is the prior.
    """

    if candidate_logits.ndim != 3:
        raise ValueError(
            "candidate_logits must be [B, N, C]"
        )
    if baseline_logits.shape != (
        candidate_logits.shape[0],
        candidate_logits.shape[2],
    ):
        raise ValueError(
            "baseline and candidate logits do not match"
        )

    baseline_log_probability = F.log_softmax(
        baseline_logits.float(),
        dim=-1,
    )

    candidate_log_probability = F.log_softmax(
        candidate_logits.float(),
        dim=-1,
    )

    candidate_probability = (
        candidate_log_probability.exp()
    )

    information_gain = (
        candidate_probability
        * (
            candidate_log_probability
            - baseline_log_probability[:, None, :]
        )
    ).sum(dim=-1)

    baseline_probability = (
        baseline_log_probability.exp()
    )

    baseline_entropy = -(
        baseline_probability
        * baseline_log_probability
    ).sum(dim=-1)

    candidate_entropy = -(
        candidate_probability
        * candidate_log_probability
    ).sum(dim=-1)

    entropy_reduction = (
        baseline_entropy[:, None]
        - candidate_entropy
    )

    return information_gain, entropy_reduction


def deterministic_random_indices(
    sample_ids: list[str],
    *,
    candidate_count: int,
    selected_count: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[np.ndarray] = []

    for sample_id in sample_ids:
        digest = hashlib.sha256(
            (
                "deltasub-information-gain-control:"
                + str(sample_id)
            ).encode("utf-8")
        ).digest()

        seed = int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )

        generator = np.random.default_rng(seed)

        rows.append(
            generator.choice(
                candidate_count,
                size=selected_count,
                replace=False,
            ).astype(np.int64)
        )

    return torch.from_numpy(
        np.stack(rows)
    ).to(device)


def selection_overlap(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError(
            "selection shapes must match"
        )

    return (
        first[:, :, None]
        == second[:, None, :]
    ).any(dim=2).float().mean(dim=1)


def load_matched_baseline(
    device: torch.device,
) -> tuple[
    AppendSubtokenModel,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
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

    baseline_config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        baseline_config["method"]["variant"]
        != "selex"
    ):
        raise ValueError(
            "located baseline is not matched SelEx"
        )

    matched = construct_model(
        baseline_config,
        device=torch.device("cpu"),
    )

    state = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    matched.load_state_dict(
        state["model"],
        strict=True,
    )

    model = AppendSubtokenModel(
        matched,
        insertion_block=10,
    ).to(device)

    provenance = {
        "result": str(result_path),
        "resolved_config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
    }

    return (
        model,
        baseline_config,
        result,
        provenance,
    )


def optimizer_for(
    model: AppendSubtokenModel,
    config: ExperimentConfig,
) -> tuple[
    torch.optim.Optimizer,
    list[nn.Parameter],
    dict[str, int],
]:
    tail_parameters = [
        *model.detail_tail.parameters(),
        *model.detail_norm.parameters(),
    ]

    detail_parameters = [
        *model.child.parameters(),
        *model.positions.parameters(),
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": tail_parameters,
                "lr": config.tail_learning_rate,
                "name": "detail_tail",
            },
            {
                "params": detail_parameters,
                "lr": config.detail_learning_rate,
                "name": "subtoken_modules",
            },
        ],
        weight_decay=config.weight_decay,
    )

    parameters = [
        *tail_parameters,
        *detail_parameters,
    ]

    counts = {
        "detail_tail": sum(
            parameter.numel()
            for parameter in tail_parameters
        ),
        "subtoken_modules": sum(
            parameter.numel()
            for parameter in detail_parameters
        ),
    }

    return optimizer, parameters, counts


def checkpoint(
    path: Path,
    *,
    model: AppendSubtokenModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    generator: torch.Generator,
    configuration: ExperimentConfig,
) -> None:
    atomic_torch_save(
        {
            "schema_version": SCHEMA,
            "append_model": (
                model.append_state_dict()
            ),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "loader_generator": (
                generator.get_state()
            ),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda
                    .get_rng_state_all()
                ),
            },
            "configuration": (
                asdict(configuration)
            ),
        },
        path,
    )


def train_model(
    *,
    model: AppendSubtokenModel,
    dataset: ProductionManifestDataset,
    baseline_training: dict[str, Any],
    device: torch.device,
    configuration: ExperimentConfig,
    output: Path,
    resume: bool,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(0)

    loader = DataLoader(
        dataset,
        batch_size=(
            configuration.physical_batch_size
        ),
        shuffle=True,
        generator=generator,
        num_workers=configuration.num_workers,
        worker_init_fn=_worker_seed,
        persistent_workers=(
            configuration.num_workers > 0
        ),
    )

    optimizer, parameters, parameter_groups = (
        optimizer_for(
            model,
            configuration,
        )
    )

    optimizer_steps = (
        configuration.epochs
        * math.ceil(
            len(loader)
            / configuration.gradient_accumulation
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            max(1, optimizer_steps),
        )
    )

    checkpoint_path = (
        output / "checkpoint_last.pt"
    )
    metrics_path = output / "metrics.jsonl"

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

        if state.get(
            "configuration"
        ) != asdict(configuration):
            raise ValueError(
                "resume configuration mismatch"
            )

        model.load_append_state_dict(
            state["append_model"]
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

        generator.set_state(
            state["loader_generator"]
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

        resume_history.append(
            {
                "epoch": start_epoch,
                "checkpoint_sha256": (
                    sha256_file(
                        checkpoint_path
                    )
                ),
            }
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()

    schedule = K_VALUES

    for epoch in range(
        start_epoch,
        configuration.epochs,
    ):
        dataset.epoch = epoch
        model.train()

        sums: dict[str, float] = {}
        batches = 0

        for batch_index, batch in enumerate(
            loader
        ):
            views = batch["views"].to(device)
            targets = batch["target"].to(device)
            labelled = batch["labelled"].to(device)

            batch_size = views.shape[0]

            flat = views.reshape(
                batch_size * 2,
                3,
                224,
                224,
            )

            selected_count = schedule[
                (
                    epoch * len(loader)
                    + batch_index
                )
                % len(schedule)
            ]

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                (
                    trunk_tokens,
                    _,
                    detail_tokens,
                ) = model.trunk_and_details(flat)

                random_scores = torch.rand(
                    flat.shape[0],
                    PATCH_COUNT,
                    device=device,
                )

                indices = random_scores.topk(
                    k=selected_count,
                    dim=1,
                ).indices

                features, logits = (
                    model.encode_selected(
                        trunk_tokens,
                        detail_tokens,
                        indices,
                    )
                )

                paired_features = features.reshape(
                    batch_size,
                    2,
                    -1,
                )
                paired_logits = logits.reshape(
                    batch_size,
                    2,
                    -1,
                )

                supervised, contrastive = (
                    _branch_losses(
                        features=paired_features,
                        logits=paired_logits,
                        targets=targets,
                        labelled=labelled,
                        training=baseline_training,
                    )
                )

                with torch.no_grad():
                    teacher_logits = (
                        model.global_logits_from_trunk(
                            trunk_tokens
                        )
                        .reshape(
                            batch_size,
                            2,
                            -1,
                        )
                    )

                distillation = F.kl_div(
                    F.log_softmax(
                        paired_logits.float(),
                        dim=-1,
                    ),
                    F.softmax(
                        teacher_logits.float(),
                        dim=-1,
                    ),
                    reduction="batchmean",
                )

                task = (
                    float(
                        baseline_training[
                            "classification_weight"
                        ]
                    )
                    * supervised
                    + float(
                        baseline_training[
                            "selex_weight"
                        ]
                    )
                    * contrastive
                )

                loss = (
                    task
                    + configuration.distillation_weight
                    * distillation
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite append-subtoken loss"
                )

            (
                loss
                / configuration.gradient_accumulation
            ).backward()

            values = {
                "loss": float(loss.detach()),
                "task": float(task.detach()),
                "supervised": float(
                    supervised.detach()
                ),
                "contrastive": float(
                    contrastive.detach()
                ),
                "distillation": float(
                    distillation.detach()
                ),
                "training_k": float(
                    selected_count
                ),
            }

            for name, value in values.items():
                sums[name] = (
                    sums.get(name, 0.0)
                    + value
                )

            batches += 1

            should_step = (
                (
                    batch_index + 1
                )
                % configuration.gradient_accumulation
                == 0
                or batch_index + 1
                == len(loader)
            )

            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    configuration.gradient_clipping,
                )
                optimizer.step()
                optimizer.zero_grad(
                    set_to_none=True
                )
                scheduler.step()
                global_step += 1

        means = {
            name: value / max(batches, 1)
            for name, value in sums.items()
        }

        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "elapsed_seconds": (
                time.perf_counter()
                - started
            ),
            "learning_rates": {
                group.get("name", str(index)): (
                    group["lr"]
                )
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
            generator=generator,
            configuration=configuration,
        )

        print(
            f"epoch={epoch + 1:03d}/"
            f"{configuration.epochs} "
            f"loss={means['loss']:.6f} "
            f"task={means['task']:.6f} "
            f"distill="
            f"{means['distillation']:.6f} "
            f"k={means['training_k']:.2f}",
            flush=True,
        )

    state = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    model.load_append_state_dict(
        state["append_model"]
    )

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
        "checkpoint_epoch": int(
            state["epoch"]
        ),
        "global_step": int(
            state["global_step"]
        ),
        "parameter_groups": parameter_groups,
        "resume_history": resume_history,
        "runtime_seconds": (
            time.perf_counter() - started
        ),
    }


@torch.inference_mode()
def evaluate(
    *,
    model: AppendSubtokenModel,
    loader: DataLoader,
    device: torch.device,
    configuration: ExperimentConfig,
    candidate_count: int,
    bootstrap_draws: int,
    output: Path,
) -> dict[str, Any]:
    model.eval()

    valid_k_values = tuple(
        value
        for value in K_VALUES
        if value <= candidate_count
    )

    targets: list[int] = []
    old_values: list[bool] = []
    baseline_predictions: list[int] = []

    predictions: dict[
        str,
        dict[int, list[int]],
    ] = {
        "information_gain": {
            value: []
            for value in valid_k_values
        },
        "energy": {
            value: []
            for value in valid_k_values
        },
        "random": {
            value: []
            for value in valid_k_values
        },
    }

    diagnostics: dict[
        int,
        dict[str, list[float]],
    ] = {
        value: {
            "information_gain": [],
            "entropy_reduction": [],
            "positive_entropy_fraction": [],
            "energy_overlap": [],
            "random_overlap": [],
        }
        for value in valid_k_values
    }

    all_information_gain: list[float] = []
    all_entropy_reduction: list[float] = []

    started = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            (
                trunk_tokens,
                raw_details,
                detail_tokens,
            ) = model.trunk_and_details(images)

            baseline_logits = (
                model.global_logits_from_trunk(
                    trunk_tokens
                )
            )

            candidate_logits = (
                model.single_patch_logits(
                    trunk_tokens,
                    detail_tokens,
                    candidate_count=(
                        candidate_count
                    ),
                    chunk_size=(
                        configuration
                        .candidate_chunk_size
                    ),
                )
            )

            information_gain, entropy_reduction = (
                predictive_information_gain(
                    baseline_logits,
                    candidate_logits,
                )
            )

            energy = (
                raw_details[
                    :,
                    :candidate_count,
                ]
                .float()
                .square()
                .mean(dim=(2, 3))
                .sqrt()
            )

        baseline_predictions.extend(
            baseline_logits.float()
            .argmax(dim=1)
            .cpu()
            .tolist()
        )

        targets.extend(
            batch["target"].tolist()
        )
        old_values.extend(
            batch["old"].tolist()
        )

        all_information_gain.extend(
            information_gain.float()
            .mean(dim=1)
            .cpu()
            .tolist()
        )
        all_entropy_reduction.extend(
            entropy_reduction.float()
            .mean(dim=1)
            .cpu()
            .tolist()
        )

        sample_ids = [
            str(value)
            for value in batch["sample_id"]
        ]

        for selected_count in valid_k_values:
            information_indices = (
                information_gain.topk(
                    k=selected_count,
                    dim=1,
                ).indices
            )

            energy_indices = energy.topk(
                k=selected_count,
                dim=1,
            ).indices

            random_indices = (
                deterministic_random_indices(
                    sample_ids,
                    candidate_count=(
                        candidate_count
                    ),
                    selected_count=(
                        selected_count
                    ),
                    device=device,
                )
            )

            for name, indices in (
                (
                    "information_gain",
                    information_indices,
                ),
                ("energy", energy_indices),
                ("random", random_indices),
            ):
                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                ):
                    _, selected_logits = (
                        model.encode_selected(
                            trunk_tokens,
                            detail_tokens,
                            indices,
                        )
                    )

                predictions[name][
                    selected_count
                ].extend(
                    selected_logits.float()
                    .argmax(dim=1)
                    .cpu()
                    .tolist()
                )

            selected_information = (
                information_gain.gather(
                    1,
                    information_indices,
                )
            )

            selected_entropy = (
                entropy_reduction.gather(
                    1,
                    information_indices,
                )
            )

            diagnostics[selected_count][
                "information_gain"
            ].extend(
                selected_information.float()
                .mean(dim=1)
                .cpu()
                .tolist()
            )

            diagnostics[selected_count][
                "entropy_reduction"
            ].extend(
                selected_entropy.float()
                .mean(dim=1)
                .cpu()
                .tolist()
            )

            diagnostics[selected_count][
                "positive_entropy_fraction"
            ].extend(
                (
                    selected_entropy > 0
                )
                .float()
                .mean(dim=1)
                .cpu()
                .tolist()
            )

            diagnostics[selected_count][
                "energy_overlap"
            ].extend(
                selection_overlap(
                    information_indices,
                    energy_indices,
                )
                .cpu()
                .tolist()
            )

            diagnostics[selected_count][
                "random_overlap"
            ].extend(
                selection_overlap(
                    information_indices,
                    random_indices,
                )
                .cpu()
                .tolist()
            )

        print(
            f"evaluation batch "
            f"{batch_index + 1:04d}/"
            f"{len(loader):04d}",
            flush=True,
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

    baseline_metrics = metric_dict(
        target_array,
        baseline_array,
        old,
    )

    method_metrics: dict[
        str,
        dict[int, dict[str, float]],
    ] = {
        name: {}
        for name in predictions
    }

    prediction_arrays: dict[
        str,
        dict[int, np.ndarray],
    ] = {
        name: {}
        for name in predictions
    }

    for name in predictions:
        for selected_count in valid_k_values:
            array = np.asarray(
                predictions[name][selected_count],
                dtype=np.int64,
            )

            prediction_arrays[name][
                selected_count
            ] = array

            method_metrics[name][
                selected_count
            ] = metric_dict(
                target_array,
                array,
                old,
            )

    gates: dict[int, dict[str, Any]] = {}

    for selected_count in valid_k_values:
        candidate_metrics = (
            method_metrics[
                "information_gain"
            ][selected_count]
        )

        delta = {
            name: (
                candidate_metrics[name]
                - baseline_metrics[name]
            )
            for name in (
                "all",
                "old",
                "new",
                "hmean",
            )
        }

        interval = bootstrap_hmean_delta(
            targets=target_array,
            old=old,
            baseline=baseline_array,
            candidate=(
                prediction_arrays[
                    "information_gain"
                ][selected_count]
            ),
            draws=bootstrap_draws,
            seed=91_000 + selected_count,
        )

        control_hmean = max(
            method_metrics["energy"][
                selected_count
            ]["hmean"],
            method_metrics["random"][
                selected_count
            ]["hmean"],
        )

        passed = (
            delta["all"] >= 0.0
            and delta["new"] >= 0.005
            and delta["hmean"] >= 0.005
            and interval["lower_95"] > 0.0
            and candidate_metrics["hmean"]
            >= control_hmean
        )

        gates[selected_count] = {
            "delta": delta,
            "hmean_paired_bootstrap": interval,
            "best_control_hmean": (
                control_hmean
            ),
            "passed": passed,
        }

    passing = [
        selected_count
        for selected_count, value
        in gates.items()
        if value["passed"]
    ]

    best_k = max(
        valid_k_values,
        key=lambda selected_count: (
            method_metrics[
                "information_gain"
            ][selected_count]["hmean"]
        ),
    )

    verdict = (
        "follow"
        if passing
        else "abandon"
    )

    diagnostic_summary: dict[
        str,
        Any,
    ] = {
        "mean_information_gain_all_candidates": (
            float(
                np.mean(
                    all_information_gain
                )
            )
        ),
        "mean_entropy_reduction_all_candidates": (
            float(
                np.mean(
                    all_entropy_reduction
                )
            )
        ),
        "by_k": {},
        "evaluation_seconds": (
            time.perf_counter() - started
        ),
    }

    for selected_count in valid_k_values:
        diagnostic_summary["by_k"][
            str(selected_count)
        ] = {
            name: {
                "mean": float(
                    np.mean(values)
                ),
                "std": float(
                    np.std(values)
                ),
            }
            for name, values
            in diagnostics[
                selected_count
            ].items()
        }

    archive_values: dict[str, np.ndarray] = {
        "target": target_array,
        "old": old,
        "baseline_prediction": (
            baseline_array
        ),
    }

    for name in prediction_arrays:
        for selected_count in valid_k_values:
            archive_values[
                f"{name}_k{selected_count}"
            ] = prediction_arrays[name][
                selected_count
            ]

    np.savez_compressed(
        output / "predictions.npz",
        **archive_values,
    )

    return {
        "baseline": baseline_metrics,
        "metrics": {
            name: {
                str(selected_count): value
                for selected_count, value
                in methods.items()
            }
            for name, methods
            in method_metrics.items()
        },
        "gates": {
            str(selected_count): value
            for selected_count, value
            in gates.items()
        },
        "passing_k_values": passing,
        "best_information_gain_k": best_k,
        "diagnostics": diagnostic_summary,
        "verdict": verdict,
        "targets": target_array,
        "old": old,
        "baseline_predictions": (
            baseline_array
        ),
    }


def self_test() -> None:
    torch.manual_seed(5)

    baseline = torch.randn(3, 7)
    candidates = torch.randn(3, 11, 7)

    gain, entropy = predictive_information_gain(
        baseline,
        candidates,
    )

    assert gain.shape == (3, 11)
    assert entropy.shape == (3, 11)
    assert float(gain.min()) >= -1e-6

    random_indices = (
        deterministic_random_indices(
            ["a", "b", "c"],
            candidate_count=11,
            selected_count=4,
            device=torch.device("cpu"),
        )
    )

    repeated = deterministic_random_indices(
        ["a", "b", "c"],
        candidate_count=11,
        selected_count=4,
        device=torch.device("cpu"),
    )

    assert torch.equal(
        random_indices,
        repeated,
    )

    assert random_indices.shape == (
        3,
        4,
    )

    print("self-test: passed")


def balanced_smoke_records(
    records: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    old = [
        record
        for record in records
        if record["known_or_novel"]
        == "known"
    ]
    new = [
        record
        for record in records
        if record["known_or_novel"]
        == "novel"
    ]

    half = max(1, count // 2)

    return (
        old[:half]
        + new[:half]
    )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "exactly one visible GPU is required"
        )

    configuration = ExperimentConfig(
        epochs=args.epochs,
        candidate_chunk_size=(
            args.candidate_chunk_size
        ),
    )
    configuration.validate()

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
        return

    (
        model,
        baseline_config,
        baseline_result,
        baseline_provenance,
    ) = load_matched_baseline(device)

    records = read_manifest(
        baseline_config["dataset"][
            "manifest"
        ],
        dataset_root=(
            baseline_config["dataset"]["root"]
        ),
        check_images=True,
    )

    train_dataset = (
        ProductionManifestDataset(
            records,
            baseline_config[
                "dataset"
            ]["root"],
            split="train",
            seed=0,
            train=True,
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

    candidate_count = PATCH_COUNT
    bootstrap_draws = args.bootstrap_draws

    if args.smoke:
        train_dataset.records = (
            train_dataset.records[:64]
        )
        test_dataset.records = (
            balanced_smoke_records(
                test_dataset.records,
                16,
            )
        )
        candidate_count = 16
        bootstrap_draws = 20

    resolved = {
        "schema_version": SCHEMA,
        "experiment": asdict(
            configuration
        ),
        "k_values": list(K_VALUES),
        "candidate_count": candidate_count,
        "information_gain": (
            "KL(posterior after appending one "
            "patch's subtokens || original prediction)"
        ),
        "selection_uses_labels": False,
        "original_tokens_retained": True,
        "subtokens_appended": True,
        "detail_tokens_per_parent": 3,
        "baseline": baseline_provenance,
    }

    (
        output / "resolved_config.yaml"
    ).write_text(
        yaml.safe_dump(
            resolved,
            sort_keys=True,
        ),
        encoding="utf-8",
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
        "baseline": baseline_provenance,
        "dataset_manifest_sha256": (
            sha256_file(
                baseline_config[
                    "dataset"
                ]["manifest"]
            )
        ),
        "seed": 0,
        "oracle_used": False,
        "test_labels_used_for_training": False,
        "test_labels_used_for_selection": False,
        "original_tokens_retained": True,
        "subtokens_appended": True,
        "feature_injection": False,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
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

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    training_report = train_model(
        model=model,
        dataset=train_dataset,
        baseline_training=(
            baseline_config["training"]
        ),
        device=device,
        configuration=configuration,
        output=output,
        resume=args.resume,
    )

    evaluation_loader = DataLoader(
        test_dataset,
        batch_size=(
            configuration
            .evaluation_batch_size
        ),
        shuffle=False,
        num_workers=configuration.num_workers,
        worker_init_fn=_worker_seed,
        persistent_workers=(
            configuration.num_workers > 0
        ),
    )

    evaluation = evaluate(
        model=model,
        loader=evaluation_loader,
        device=device,
        configuration=configuration,
        candidate_count=candidate_count,
        bootstrap_draws=bootstrap_draws,
        output=output,
    )

    if not args.smoke:
        expected = baseline_result["metrics"]

        for name in (
            "all",
            "old",
            "new",
        ):
            if abs(
                evaluation["baseline"][name]
                - float(expected[name])
            ) > 1e-12:
                raise RuntimeError(
                    "recomputed matched baseline differs "
                    f"for {name}"
                )

    report = model.parameter_report()

    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "dataset": "cub",
        "seed": 0,
        "method": (
            "information_gain_append_subtokens"
        ),
        "oracle_used": False,
        "test_labels_used_for_training": False,
        "test_labels_used_for_selection": False,
        "original_tokens_retained": True,
        "subtokens_appended": True,
        "feature_injection": False,
        "information_gain_definition": (
            "KL(candidate predictive distribution "
            "after appending one parent's three "
            "subtokens || original predictive "
            "distribution)"
        ),
        "baseline": evaluation["baseline"],
        "metrics": evaluation["metrics"],
        "gates": evaluation["gates"],
        "passing_k_values": (
            evaluation["passing_k_values"]
        ),
        "best_information_gain_k": (
            evaluation[
                "best_information_gain_k"
            ]
        ),
        "diagnostics": (
            evaluation["diagnostics"]
        ),
        "training": training_report,
        "baseline_provenance": (
            baseline_provenance
        ),
        "configuration": asdict(
            configuration
        ),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "runtime_seconds": (
            time.perf_counter() - started
        ),
        "peak_cuda_memory_bytes": (
            torch.cuda
            .max_memory_allocated()
        ),
        "total_parameters": report["total"],
        "trainable_parameters": (
            report["trainable"]
        ),
        "screening_note": (
            "K values are jointly screened here only "
            "to decide whether the hypothesis deserves "
            "a separately locked confirmatory run. "
            "These results are not a final benchmark."
        ),
        "gate_rule": {
            "all_delta_minimum": 0.0,
            "new_delta_minimum": 0.005,
            "hmean_delta_minimum": 0.005,
            "hmean_lower_95_above_zero": True,
            "information_gain_hmean_not_below_controls": True,
        },
        "verdict": evaluation["verdict"],
    }

    _atomic_json(result_path, result)

    print()
    print(
        "===== INFORMATION-GAIN APPEND GATE ====="
    )

    print(
        "method                    "
        "K       all       old       new      hmean"
    )

    baseline = evaluation["baseline"]

    print(
        f"{'matched_selex':<25}"
        f"{'-':>3}"
        f"{baseline['all']:>10.6f}"
        f"{baseline['old']:>10.6f}"
        f"{baseline['new']:>10.6f}"
        f"{baseline['hmean']:>11.6f}"
    )

    for selected_count in K_VALUES:
        key = str(selected_count)

        if key not in evaluation[
            "metrics"
        ]["information_gain"]:
            continue

        for display_name, method_name in (
            (
                "information_gain",
                "information_gain",
            ),
            ("energy_control", "energy"),
            ("random_control", "random"),
        ):
            metrics = evaluation[
                "metrics"
            ][method_name][key]

            print(
                f"{display_name:<25}"
                f"{selected_count:>3}"
                f"{metrics['all']:>10.6f}"
                f"{metrics['old']:>10.6f}"
                f"{metrics['new']:>10.6f}"
                f"{metrics['hmean']:>11.6f}"
            )

        gate = evaluation["gates"][key]
        delta = gate["delta"]
        interval = gate[
            "hmean_paired_bootstrap"
        ]

        print(
            f"  K={selected_count}: "
            f"delta All={delta['all']:+.6f}, "
            f"New={delta['new']:+.6f}, "
            f"H={delta['hmean']:+.6f}, "
            f"H 95% CI="
            f"[{interval['lower_95']:+.6f}, "
            f"{interval['upper_95']:+.6f}], "
            f"gate="
            f"{'PASSED' if gate['passed'] else 'FAILED'}"
        )

    print()
    print(
        "best information-gain K: "
        f"{evaluation['best_information_gain_k']}"
    )
    print(
        "passing K values: "
        f"{evaluation['passing_k_values']}"
    )
    print()
    print(
        "DELTASUB VERDICT: "
        + evaluation["verdict"].upper()
    )
    print(f"result: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default=(
            "artifacts/deltasub_information_gain/"
            "cub/seed_0"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=2000,
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
