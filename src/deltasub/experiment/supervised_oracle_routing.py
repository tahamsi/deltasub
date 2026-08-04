"""Exact supervised oracle-routing upper bound for DeltaSub."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..data.manifests import read_manifest
from ..evaluation.gcd_v2 import (
    evaluate_gcd_v2,
    provenance as gcd_provenance,
)
from ..models.deltasub_v2 import DeltaSubV2
from ..utils.checkpointing import load_checkpoint
from ..utils.hashing import sha256_file
from ..utils.reproducibility import seed_everything
from .deltasub_v2_training import (
    ProductionManifestDataset,
    _atomic_json,
    _with_hmean,
    _worker_seed,
    construct_model,
    load_config,
)


SCHEMA = "deltasub-v2.supervised-oracle-routing.v1"
PATCH_COUNT = 256
K_VALUES = (1, 4, 8, 16)


def topk_mask(
    scores: torch.Tensor,
    k: int,
    *,
    positive_only: bool = False,
) -> torch.Tensor:
    """Create deterministic top-K masks."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [B, P]")
    if k < 0 or k > scores.shape[1]:
        raise ValueError("invalid K")

    mask = torch.zeros_like(scores, dtype=torch.bool)

    if k == 0:
        return mask

    indices = torch.argsort(
        scores,
        dim=1,
        descending=True,
        stable=True,
    )[:, :k]

    values = scores.gather(1, indices)
    keep = torch.isfinite(values)

    if positive_only:
        keep = keep & (values > 0)

    mask.scatter_(1, indices, keep)
    return mask


def direct_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    old: np.ndarray,
) -> dict[str, Any]:
    """Direct supervised accuracy without cluster remapping."""

    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )
    old = np.asarray(old, dtype=bool)

    if (
        targets.shape != predictions.shape
        or targets.shape != old.shape
    ):
        raise ValueError("metric arrays must have equal shape")

    correct = predictions == targets

    all_accuracy = float(correct.mean())
    old_accuracy = (
        float(correct[old].mean())
        if bool(old.any())
        else 0.0
    )
    new_accuracy = (
        float(correct[~old].mean())
        if bool((~old).any())
        else 0.0
    )

    hmean = (
        2.0
        * old_accuracy
        * new_accuracy
        / (old_accuracy + new_accuracy)
        if old_accuracy + new_accuracy > 0
        else 0.0
    )

    return {
        "all": all_accuracy,
        "old": old_accuracy,
        "new": new_accuracy,
        "hmean": hmean,
        "protocol": "direct_supervised",
        "unit": "fraction",
    }


def paired_bootstrap_delta(
    *,
    targets: np.ndarray,
    old: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap intervals for direct metric deltas."""

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")

    rng = np.random.default_rng(seed)
    sample_count = len(targets)

    values = {
        "all": [],
        "old": [],
        "new": [],
        "hmean": [],
    }

    for _ in range(draws):
        indices = rng.integers(
            0,
            sample_count,
            size=sample_count,
        )

        base_metrics = direct_metrics(
            targets[indices],
            baseline[indices],
            old[indices],
        )
        candidate_metrics = direct_metrics(
            targets[indices],
            candidate[indices],
            old[indices],
        )

        for metric in values:
            values[metric].append(
                float(candidate_metrics[metric])
                - float(base_metrics[metric])
            )

    result: dict[str, dict[str, float]] = {}

    for metric, samples in values.items():
        array = np.asarray(samples, dtype=np.float64)

        result[metric] = {
            "mean_delta": float(array.mean()),
            "lower_95": float(
                np.quantile(array, 0.025)
            ),
            "upper_95": float(
                np.quantile(array, 0.975)
            ),
        }

    return result


def deterministic_random_scores(
    sample_ids: list[str],
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-sample random controls independent of batching."""

    rows: list[torch.Tensor] = []

    for sample_id in sample_ids:
        digest = hashlib.sha256(
            f"{seed}:{sample_id}".encode("utf-8")
        ).digest()

        row_seed = int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        ) % (2**63 - 1)

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(row_seed)

        rows.append(
            torch.rand(
                PATCH_COUNT,
                generator=generator,
            )
        )

    return torch.stack(rows).to(device)


def base_parent_residuals(
    model: DeltaSubV2,
    trunk_tokens: torch.Tensor,
    adapted_details: torch.Tensor,
) -> torch.Tensor:
    """Compute unselected parent-aligned Haar residuals."""

    prefix_count = int(
        model.backbone.prefix_token_count
    )
    parent_tokens = trunk_tokens[
        :,
        prefix_count : prefix_count + PATCH_COUNT,
    ]

    detail_modes = (
        adapted_details
        + model.mode_embeddings[
            None,
            None,
            :,
            :,
        ].to(adapted_details)
    )

    queries = F.normalize(
        model.parent_query(
            model.parent_norm(parent_tokens)
        ).float(),
        dim=-1,
    )
    keys = F.normalize(
        model.detail_key(
            model.detail_norm(detail_modes)
        ).float(),
        dim=-1,
    )

    attention = F.softmax(
        (
            queries.unsqueeze(2)
            * keys
        ).sum(dim=-1)
        / 0.07,
        dim=2,
    ).to(adapted_details)

    residuals = (
        attention.unsqueeze(-1)
        * adapted_details
    ).sum(dim=2)

    return (
        torch.sigmoid(model.detail_scale_logit)
        .to(residuals)
        * residuals
    )


def encode_uniform_mask(
    model: DeltaSubV2,
    trunk_tokens: torch.Tensor,
    residuals: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Encode binary selections with uniform selected-patch strength."""

    if mask.shape != residuals.shape[:2]:
        raise ValueError(
            "mask must match residual batch and patch dimensions"
        )

    prefix_count = int(
        model.backbone.prefix_token_count
    )
    parent_start = prefix_count
    parent_end = prefix_count + PATCH_COUNT

    parent_tokens = trunk_tokens[
        :,
        parent_start:parent_end,
    ]

    weighted = (
        mask.to(residuals)
        .unsqueeze(-1)
        * residuals
    )

    selected_count = (
        mask.sum(dim=1)
        .clamp_min(1)
        .to(weighted)
    )

    pooled = (
        weighted.sum(dim=1)
        / selected_count.unsqueeze(1)
    )

    prefix_tokens = trunk_tokens[
        :,
        :prefix_count,
    ].clone()

    prefix_tokens = torch.cat(
        (
            (
                prefix_tokens[:, 0]
                + pooled.to(prefix_tokens)
            ).unsqueeze(1),
            prefix_tokens[:, 1:],
        ),
        dim=1,
    )

    injected_parents = (
        parent_tokens
        + weighted.to(parent_tokens)
    )

    branch_tokens = torch.cat(
        (
            prefix_tokens,
            injected_parents,
            trunk_tokens[:, parent_end:],
        ),
        dim=1,
    )

    return model._tail_encode(branch_tokens)


def exact_single_patch_utility(
    *,
    model: DeltaSubV2,
    trunk_tokens: torch.Tensor,
    residuals: torch.Tensor,
    targets: torch.Tensor,
    global_logits: torch.Tensor,
    candidate_chunk: int,
    candidate_count: int,
) -> torch.Tensor:
    """Exact true-label CE gain for each one-patch intervention."""

    if not 1 <= candidate_count <= PATCH_COUNT:
        raise ValueError("invalid candidate count")
    if candidate_chunk <= 0:
        raise ValueError("candidate chunk must be positive")

    batch_size = trunk_tokens.shape[0]

    global_ce = F.cross_entropy(
        global_logits.float(),
        targets,
        reduction="none",
    )

    utility = torch.full(
        (batch_size, PATCH_COUNT),
        float("-inf"),
        device=trunk_tokens.device,
        dtype=torch.float32,
    )

    for start in range(
        0,
        candidate_count,
        candidate_chunk,
    ):
        end = min(
            start + candidate_chunk,
            candidate_count,
        )
        count = end - start

        repeated_trunk = (
            trunk_tokens[:, None]
            .expand(-1, count, -1, -1)
            .reshape(
                batch_size * count,
                trunk_tokens.shape[1],
                trunk_tokens.shape[2],
            )
        )
        repeated_residuals = (
            residuals[:, None]
            .expand(-1, count, -1, -1)
            .reshape(
                batch_size * count,
                PATCH_COUNT,
                residuals.shape[2],
            )
        )

        masks = torch.zeros(
            batch_size,
            count,
            PATCH_COUNT,
            dtype=torch.bool,
            device=trunk_tokens.device,
        )

        patch_indices = torch.arange(
            start,
            end,
            device=trunk_tokens.device,
        )

        masks.scatter_(
            2,
            patch_indices[
                None,
                :,
                None,
            ].expand(batch_size, -1, -1),
            True,
        )

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            candidate_features = encode_uniform_mask(
                model,
                repeated_trunk,
                repeated_residuals,
                masks.reshape(
                    batch_size * count,
                    PATCH_COUNT,
                ),
            )
            candidate_logits = model.head(
                candidate_features
            )

        repeated_targets = (
            targets[:, None]
            .expand(-1, count)
            .reshape(-1)
        )

        candidate_ce = F.cross_entropy(
            candidate_logits.float(),
            repeated_targets,
            reduction="none",
        ).reshape(batch_size, count)

        utility[:, start:end] = (
            global_ce[:, None]
            - candidate_ce
        )

    return utility


def evaluate_oracle(
    *,
    model: DeltaSubV2,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    candidate_chunk: int,
    candidate_count: int,
    bootstrap_draws: int,
    output_directory: Path,
) -> dict[str, Any]:
    """Evaluate energy, random, and exact supervised oracle routing."""

    model.eval()
    model.requires_grad_(False)

    method_names = [
        "global",
        "trained_adaptive",
    ]

    for k in K_VALUES:
        method_names.extend(
            (
                f"energy_k{k}",
                f"random_k{k}",
                f"oracle_k{k}",
            )
        )

    predictions: dict[str, list[int]] = {
        name: []
        for name in method_names
    }

    targets_all: list[int] = []
    old_all: list[bool] = []
    sample_ids_all: list[str] = []
    utilities_all: list[np.ndarray] = []

    diagnostics: dict[str, list[float]] = {
        "positive_patch_fraction": [],
        "best_single_patch_gain": [],
        "trained_adaptive_k": [],
    }

    for k in K_VALUES:
        diagnostics[
            f"oracle_k{k}_selected"
        ] = []
        diagnostics[
            f"energy_k{k}_oracle_recall"
        ] = []
        diagnostics[
            f"random_k{k}_oracle_recall"
        ] = []

    started = time.perf_counter()
    processed = 0
    next_report = 128

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            sample_ids = [
                str(value)
                for value in batch["sample_id"]
            ]

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                trunk_tokens, parents = model._run_trunk(
                    images
                )
                global_features = model._tail_encode(
                    trunk_tokens
                )
                global_logits = model.head(
                    global_features
                )

                raw_details = model._haar_details(
                    images,
                    parents,
                )
                adapted_details = model.detail_adapter(
                    raw_details
                )
                routing_scores = model._routing_scores(
                    trunk_tokens,
                    adapted_details,
                )
                residuals = base_parent_residuals(
                    model,
                    trunk_tokens,
                    adapted_details,
                )

                trained_selection = model._selection(
                    trunk_tokens,
                    adapted_details,
                )
                trained_features = model._detail_branch(
                    trunk_tokens,
                    adapted_details,
                    trained_selection,
                    global_features,
                )
                trained_logits = model.head(
                    trained_features
                )

            utility = exact_single_patch_utility(
                model=model,
                trunk_tokens=trunk_tokens,
                residuals=residuals,
                targets=targets,
                global_logits=global_logits,
                candidate_chunk=candidate_chunk,
                candidate_count=candidate_count,
            )

            random_scores = deterministic_random_scores(
                sample_ids,
                seed=seed,
                device=device,
            )

            predictions["global"].extend(
                global_logits.float()
                .argmax(dim=1)
                .cpu()
                .tolist()
            )
            predictions["trained_adaptive"].extend(
                trained_logits.float()
                .argmax(dim=1)
                .cpu()
                .tolist()
            )

            diagnostics[
                "positive_patch_fraction"
            ].extend(
                (utility > 0)
                .float()
                .mean(dim=1)
                .cpu()
                .tolist()
            )
            diagnostics[
                "best_single_patch_gain"
            ].extend(
                utility.amax(dim=1)
                .cpu()
                .tolist()
            )
            diagnostics[
                "trained_adaptive_k"
            ].extend(
                trained_selection.adaptive_k
                .float()
                .cpu()
                .tolist()
            )

            for k in K_VALUES:
                energy_mask = topk_mask(
                    routing_scores.float(),
                    k,
                )
                random_mask = topk_mask(
                    random_scores,
                    k,
                )
                oracle_mask = topk_mask(
                    utility,
                    k,
                    positive_only=True,
                )

                stacked_trunk = torch.cat(
                    (
                        trunk_tokens,
                        trunk_tokens,
                        trunk_tokens,
                    ),
                    dim=0,
                )
                stacked_residuals = torch.cat(
                    (
                        residuals,
                        residuals,
                        residuals,
                    ),
                    dim=0,
                )
                stacked_masks = torch.cat(
                    (
                        energy_mask,
                        random_mask,
                        oracle_mask,
                    ),
                    dim=0,
                )

                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                ):
                    stacked_features = encode_uniform_mask(
                        model,
                        stacked_trunk,
                        stacked_residuals,
                        stacked_masks,
                    )
                    stacked_logits = model.head(
                        stacked_features
                    )

                (
                    energy_logits,
                    random_logits,
                    oracle_logits,
                ) = stacked_logits.chunk(3, dim=0)

                predictions[f"energy_k{k}"].extend(
                    energy_logits.float()
                    .argmax(dim=1)
                    .cpu()
                    .tolist()
                )
                predictions[f"random_k{k}"].extend(
                    random_logits.float()
                    .argmax(dim=1)
                    .cpu()
                    .tolist()
                )
                predictions[f"oracle_k{k}"].extend(
                    oracle_logits.float()
                    .argmax(dim=1)
                    .cpu()
                    .tolist()
                )

                oracle_count = (
                    oracle_mask.sum(dim=1)
                    .clamp_min(1)
                )

                energy_overlap = (
                    energy_mask
                    & oracle_mask
                ).sum(dim=1).float() / (
                    oracle_count.float()
                )

                random_overlap = (
                    random_mask
                    & oracle_mask
                ).sum(dim=1).float() / (
                    oracle_count.float()
                )

                diagnostics[
                    f"oracle_k{k}_selected"
                ].extend(
                    oracle_mask.sum(dim=1)
                    .float()
                    .cpu()
                    .tolist()
                )
                diagnostics[
                    f"energy_k{k}_oracle_recall"
                ].extend(
                    energy_overlap.cpu().tolist()
                )
                diagnostics[
                    f"random_k{k}_oracle_recall"
                ].extend(
                    random_overlap.cpu().tolist()
                )

            targets_all.extend(
                targets.cpu().tolist()
            )
            old_all.extend(batch["old"].tolist())
            sample_ids_all.extend(sample_ids)
            utilities_all.append(
                utility.cpu().numpy()
            )

            processed += images.shape[0]

            if processed >= next_report:
                elapsed = time.perf_counter() - started
                print(
                    f"processed={processed} "
                    f"elapsed_min={elapsed / 60:.1f} "
                    f"samples_per_second="
                    f"{processed / max(elapsed, 1e-12):.3f}",
                    flush=True,
                )
                next_report += 128

    target_array = np.asarray(
        targets_all,
        dtype=np.int64,
    )
    old_array = np.asarray(
        old_all,
        dtype=bool,
    )

    prediction_arrays = {
        name: np.asarray(
            values,
            dtype=np.int64,
        )
        for name, values in predictions.items()
    }

    metrics: dict[str, Any] = {}

    for name, prediction in prediction_arrays.items():
        metrics[name] = {
            "gcd_v2": _with_hmean(
                evaluate_gcd_v2(
                    target_array,
                    prediction,
                    old_array,
                ).as_dict()
            ),
            "direct": direct_metrics(
                target_array,
                prediction,
                old_array,
            ),
        }

    bootstrap: dict[str, Any] = {}

    for k in K_VALUES:
        name = f"oracle_k{k}"

        bootstrap[name] = paired_bootstrap_delta(
            targets=target_array,
            old=old_array,
            baseline=prediction_arrays["global"],
            candidate=prediction_arrays[name],
            draws=bootstrap_draws,
            seed=seed + 10_000 + k,
        )

    oracle_names = [
        f"oracle_k{k}"
        for k in K_VALUES
    ]

    best_oracle = max(
        oracle_names,
        key=lambda name: float(
            metrics[name]["direct"]["hmean"]
        ),
    )

    baseline_direct = metrics["global"]["direct"]
    best_direct = metrics[best_oracle]["direct"]

    point_delta = {
        metric: (
            float(best_direct[metric])
            - float(baseline_direct[metric])
        )
        for metric in (
            "all",
            "old",
            "new",
            "hmean",
        )
    }

    best_interval = bootstrap[best_oracle]

    gate_passed = (
        point_delta["all"] >= 0.0
        and point_delta["new"] >= 0.005
        and point_delta["hmean"] >= 0.005
        and best_interval["hmean"]["lower_95"] > 0.0
    )

    diagnostic_summary = {
        name: {
            "count": len(values),
            "mean": float(
                np.asarray(
                    values,
                    dtype=np.float64,
                ).mean()
            ),
            "std": float(
                np.asarray(
                    values,
                    dtype=np.float64,
                ).std()
            ),
        }
        for name, values in diagnostics.items()
    }

    utility_array = np.concatenate(
        utilities_all,
        axis=0,
    )

    np.savez_compressed(
        output_directory / "predictions.npz",
        target=target_array,
        old=old_array,
        sample_id=np.asarray(sample_ids_all),
        utility=utility_array,
        **{
            f"prediction_{name}": value
            for name, value in prediction_arrays.items()
        },
    )

    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "oracle_definition": (
            "exact one-patch true-label cross-entropy gain; "
            "top-K selects only positive-gain patches"
        ),
        "test_label_usage": (
            "test labels are used only for the non-deployable "
            "oracle upper bound and evaluation"
        ),
        "selection_controls": {
            "trained_adaptive": (
                "existing semantic-detail routing and "
                "retained-mass budget"
            ),
            "energy": (
                "existing routing-score ranking with uniform "
                "selected-patch strength"
            ),
            "random": (
                "deterministic per-sample random ranking with "
                "uniform selected-patch strength"
            ),
            "oracle": (
                "true-label exact single-patch utility ranking "
                "with uniform selected-patch strength"
            ),
        },
        "candidate_count": candidate_count,
        "candidate_chunk": candidate_chunk,
        "k_values": list(K_VALUES),
        "metrics": metrics,
        "bootstrap_delta_vs_global": bootstrap,
        "diagnostics": diagnostic_summary,
        "best_oracle": best_oracle,
        "best_oracle_direct_delta_vs_global": (
            point_delta
        ),
        "oracle_headroom_gate": (
            "passed" if gate_passed else "failed"
        ),
        "gate_rule": {
            "all_delta_minimum": 0.0,
            "new_delta_minimum": 0.005,
            "hmean_delta_minimum": 0.005,
            "hmean_bootstrap_lower_95_above_zero": True,
        },
        "runtime_seconds": (
            time.perf_counter() - started
        ),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated()
        ),
    }

    _atomic_json(
        output_directory / "result.json",
        result,
    )

    print()
    print("===== DIRECT SUPERVISED ACCURACY =====")
    print(
        "method                 all       old       new      hmean"
    )

    ordered = [
        "global",
        "trained_adaptive",
    ]

    for k in K_VALUES:
        ordered.extend(
            (
                f"energy_k{k}",
                f"random_k{k}",
                f"oracle_k{k}",
            )
        )

    for name in ordered:
        values = metrics[name]["direct"]

        print(
            f"{name:<21}"
            f"{float(values['all']):>9.6f}"
            f"{float(values['old']):>10.6f}"
            f"{float(values['new']):>10.6f}"
            f"{float(values['hmean']):>11.6f}"
        )

    print()
    print("===== BEST ORACLE GATE =====")
    print(f"best oracle: {best_oracle}")

    for metric, delta in point_delta.items():
        print(f"{metric:<5}: {delta:+.6f}")

    interval = best_interval["hmean"]
    print(
        "hmean paired-bootstrap 95% CI: "
        f"[{interval['lower_95']:+.6f}, "
        f"{interval['upper_95']:+.6f}]"
    )
    print(
        "oracle headroom gate: "
        + ("PASSED" if gate_passed else "FAILED")
    )

    print()
    print("===== ROUTING DIAGNOSTICS =====")

    for name in sorted(diagnostic_summary):
        values = diagnostic_summary[name]
        print(
            f"{name}: "
            f"{values['mean']:.6f} "
            f"± {values['std']:.6f}"
        )

    return result


def run(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_directory: str | Path,
    batch_size: int,
    candidate_chunk: int,
    bootstrap_draws: int,
    smoke: bool,
) -> dict[str, Any]:
    config = load_config(config_path)

    if (
        config["training"].get(
            "supervision_mode",
            "gcd",
        )
        != "fully_supervised"
    ):
        raise ValueError(
            "oracle routing requires the fully supervised checkpoint"
        )

    if config["method"]["variant"] != "deltasub_v2":
        raise ValueError(
            "oracle routing requires DeltaSub v2"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "exactly one visible GPU is required"
        )

    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    device = torch.device("cuda:0")
    seed = int(config["seed"])
    seed_everything(seed)
    torch.cuda.reset_peak_memory_stats()

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    result_path = output / "result.json"

    if result_path.is_file() and not smoke:
        result = json.loads(
            result_path.read_text(encoding="utf-8")
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    checkpoint_path = Path(checkpoint_path)
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
        dataset_root=config["dataset"]["root"],
        check_images=True,
    )

    dataset = ProductionManifestDataset(
        records,
        config["dataset"]["root"],
        split="test",
        seed=seed,
        train=False,
    )

    candidate_count = PATCH_COUNT

    if smoke:
        old_index = next(
            index
            for index, record in enumerate(dataset.records)
            if record["known_or_novel"] == "known"
        )
        new_index = next(
            index
            for index, record in enumerate(dataset.records)
            if record["known_or_novel"] == "novel"
        )

        dataset = Subset(
            dataset,
            [old_index, new_index],
        )
        candidate_count = 16
        bootstrap_draws = min(
            bootstrap_draws,
            20,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=(
            0
            if smoke
            else int(
                config["training"]["num_workers"]
            )
        ),
        worker_init_fn=_worker_seed,
        persistent_workers=(
            not smoke
            and int(
                config["training"]["num_workers"]
            )
            > 0
        ),
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
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
        "manifest_sha256": sha256_file(
            config["dataset"]["manifest"]
        ),
        "split_sha256": sha256_file(
            config["dataset"]["split_validation"]
        ),
        "gcd": gcd_provenance(
            implementation_path=(
                Path(__file__).parents[1]
                / "evaluation"
                / "gcd_v2.py"
            )
        ),
        "seed": seed,
        "smoke": smoke,
    }

    _atomic_json(
        output / "provenance.json",
        provenance,
    )

    return evaluate_oracle(
        model=model,
        loader=loader,
        device=device,
        seed=seed,
        candidate_chunk=candidate_chunk,
        candidate_count=candidate_count,
        bootstrap_draws=bootstrap_draws,
        output_directory=output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--candidate-chunk",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    args = parser.parse_args()

    run(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_directory=args.output,
        batch_size=args.batch_size,
        candidate_chunk=args.candidate_chunk,
        bootstrap_draws=args.bootstrap_draws,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
