"""Training and evaluation for Novelty-Preserving DeltaSub."""

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
import yaml
from torch.utils.data import DataLoader

from ..data.manifests import read_manifest
from ..evaluation.gcd_v2 import (
    evaluate_gcd_v2,
    provenance as gcd_provenance,
)
from ..losses.selex import selex_loss
from ..models.backbones.dinov2 import DINOv2Adapter
from ..models.novelty_deltasub import NoveltyPreservingDeltaSub
from ..models.novelty_preserving import (
    NoveltyPreservingConfig,
    novelty_preserving_objective,
)
from ..utils.checkpointing import atomic_torch_save, load_checkpoint
from ..utils.hashing import sha256_file, stable_hash
from ..utils.reproducibility import seed_everything
from .training import (
    ProductionManifestDataset,
    _atomic_json,
    _atomic_text,
    _worker_seed,
)


SCHEMA = "novelty-preserving-deltasub.diagnostic.v1"

TOP_KEYS = {
    "schema_version",
    "dataset",
    "backbone",
    "selex",
    "seed",
    "training",
    "method",
    "baseline_result",
    "output_directory",
}

TRAINING_KEYS = {
    "epochs",
    "physical_batch_size",
    "gradient_accumulation",
    "evaluation_batch_size",
    "num_workers",
    "learning_rate",
    "weight_decay",
    "temperature",
    "supervised_weight",
    "classification_weight",
    "selex_weight",
    "gradient_clipping",
}

METHOD_KEYS = {
    "maximum_detail_parents",
    "gate_hidden_dim",
    "novelty_temperature",
    "known_similarity_threshold",
    "preservation_weight",
    "view_consistency_weight",
    "novel_dispersion_weight",
    "sparsity_weight",
    "dispersion_margin",
    "prototype_momentum",
    "cold_start_novelty",
}


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    if not isinstance(value, dict) or set(value) != TOP_KEYS:
        raise ValueError(
            f"top-level config mismatch: "
            f"{sorted(set(value or {}) ^ TOP_KEYS)}"
        )
    if value["schema_version"] != SCHEMA:
        raise ValueError("unsupported novelty diagnostic schema")
    if set(value["training"]) != TRAINING_KEYS:
        raise ValueError("training configuration keys mismatch")
    if set(value["method"]) != METHOD_KEYS:
        raise ValueError("method configuration keys mismatch")
    if int(value["seed"]) != 0:
        raise ValueError("first diagnostic must use seed 0")

    training = value["training"]

    if (
        int(training["physical_batch_size"])
        * int(training["gradient_accumulation"])
        != 128
    ):
        raise ValueError("diagnostic effective batch must be exactly 128")
    if int(training["epochs"]) != 20:
        raise ValueError("diagnostic must match the 20-epoch baseline")
    if not Path(value["baseline_result"]).is_file():
        raise FileNotFoundError("matched baseline result is absent")

    return value


def _method_configuration(config: dict[str, Any]) -> NoveltyPreservingConfig:
    method = config["method"]

    return NoveltyPreservingConfig(
        feature_dim=768,
        maximum_detail_parents=int(
            method["maximum_detail_parents"]
        ),
        gate_hidden_dim=int(method["gate_hidden_dim"]),
        novelty_temperature=float(method["novelty_temperature"]),
        known_similarity_threshold=float(
            method["known_similarity_threshold"]
        ),
        preservation_weight=float(method["preservation_weight"]),
        view_consistency_weight=float(
            method["view_consistency_weight"]
        ),
        novel_dispersion_weight=float(
            method["novel_dispersion_weight"]
        ),
        sparsity_weight=float(method["sparsity_weight"]),
        dispersion_margin=float(method["dispersion_margin"]),
    )


def construct_model(
    config: dict[str, Any],
    *,
    device: torch.device,
) -> NoveltyPreservingDeltaSub:
    backbone_config = config["backbone"]

    backbone = DINOv2Adapter.from_official_checkpoint(
        backbone_config["checkpoint"],
        backbone_config["checkpoint_sha256"],
        source_root=backbone_config["source_root"],
        model_name=backbone_config["name"],
    )

    return NoveltyPreservingDeltaSub(
        backbone,
        class_count=int(config["dataset"]["classes"]),
        config=_method_configuration(config),
        seed=int(config["seed"]),
        prototype_momentum=float(
            config["method"]["prototype_momentum"]
        ),
        cold_start_novelty=float(
            config["method"]["cold_start_novelty"]
        ),
    ).to(device)


def _checkpoint(
    path: Path,
    *,
    model: NoveltyPreservingDeltaSub,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_loss: float,
    config: dict[str, Any],
    loader_generator: torch.Generator,
) -> None:
    atomic_torch_save(
        {
            "schema_version": SCHEMA,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_selection_loss": best_loss,
            "config_hash": stable_hash(config),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
            "loader_generator": loader_generator.get_state(),
        },
        path,
    )


def _training_batch(
    *,
    model: NoveltyPreservingDeltaSub,
    batch: dict[str, Any],
    training: dict[str, Any],
    method_config: NoveltyPreservingConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    views = batch["views"].to(device)
    labelled = batch["labelled"].to(device)
    targets = batch["target"].to(device)
    batch_size = views.shape[0]

    output = model.paired_features(views)

    features = output.fused_features.reshape(
        batch_size,
        2,
        -1,
    )
    global_features = output.global_features.reshape(
        batch_size,
        2,
        -1,
    )
    gates = output.gate.reshape(batch_size, 2)
    adaptive_k = output.selection.adaptive_k.reshape(
        batch_size,
        2,
    )

    logits = model.head(features)

    if labelled.any():
        supervised = torch.nn.functional.cross_entropy(
            logits[:, 0][labelled].float(),
            targets[labelled],
        )
    else:
        supervised = logits.float().sum() * 0.0

    pseudo = logits.detach().mean(dim=1).argmax(dim=1)
    confusion = torch.eye(
        2 * batch_size,
        device=device,
        dtype=features.dtype,
    )

    unsupervised = selex_loss(
        features,
        targets.clamp_min(0),
        labelled,
        (pseudo,),
        confusion,
        temperature=float(training["temperature"]),
        sup_con_weight=float(training["supervised_weight"]),
    )

    if labelled.any():
        model.update_known_prototypes(
            global_features[:, 0][labelled],
            targets[labelled],
        )

    active_prototypes = model.prototype_bank.active()
    prototypes = (
        active_prototypes
        if active_prototypes.shape[0]
        else None
    )

    auxiliary = novelty_preserving_objective(
        fused_view_one=features[:, 0],
        fused_view_two=features[:, 1],
        global_view_one=global_features[:, 0],
        global_view_two=global_features[:, 1],
        gate_view_one=gates[:, 0],
        gate_view_two=gates[:, 1],
        labelled_mask=labelled,
        known_prototypes=prototypes,
        config=method_config,
    )

    total = (
        float(training["classification_weight"]) * supervised
        + float(training["selex_weight"]) * unsupervised
        + auxiliary.total
    )

    stats = {
        "total": float(total.detach()),
        "supervised": float(supervised.detach()),
        "selex": float(unsupervised.detach()),
        "preservation": float(auxiliary.preservation.detach()),
        "view_consistency": float(
            auxiliary.view_consistency.detach()
        ),
        "novel_dispersion": float(
            auxiliary.novel_dispersion.detach()
        ),
        "sparsity": float(auxiliary.sparsity.detach()),
        "mean_gate": float(gates.detach().float().mean()),
        "mean_k": float(adaptive_k.detach().float().mean()),
        "active_prototypes": float(
            model.prototype_bank.active_class_count
        ),
    }

    return total, stats


def _summary(values: list[float], mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values, dtype=np.float64)[mask]

    return {
        "count": int(selected.size),
        "mean": float(selected.mean()),
        "std": float(selected.std()),
    }


@torch.inference_mode()
def evaluate(
    model: NoveltyPreservingDeltaSub,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()

    predictions: list[int] = []
    targets: list[int] = []
    old_mask: list[bool] = []
    gates: list[float] = []
    novelty: list[float] = []
    selected_k: list[float] = []

    started = time.perf_counter()

    for batch in loader:
        output = model.features(
            batch["image"].to(device),
            return_auxiliary=True,
        )
        logits = model.head(output.fused_features)

        predictions.extend(
            logits.float().argmax(dim=1).cpu().tolist()
        )
        targets.extend(batch["target"].tolist())
        old_mask.extend(batch["old"].tolist())
        gates.extend(output.gate.float().cpu().tolist())
        novelty.extend(output.novelty.float().cpu().tolist())
        selected_k.extend(
            output.selection.adaptive_k.float().cpu().tolist()
        )

    elapsed = time.perf_counter() - started
    old = np.asarray(old_mask, dtype=bool)
    all_mask = np.ones_like(old, dtype=bool)

    metrics = evaluate_gcd_v2(
        targets,
        predictions,
        old,
    ).as_dict()

    diagnostics = {
        "all": {
            "gate": _summary(gates, all_mask),
            "novelty": _summary(novelty, all_mask),
            "selected_k": _summary(selected_k, all_mask),
        },
        "old": {
            "gate": _summary(gates, old),
            "novelty": _summary(novelty, old),
            "selected_k": _summary(selected_k, old),
        },
        "new": {
            "gate": _summary(gates, ~old),
            "novelty": _summary(novelty, ~old),
            "selected_k": _summary(selected_k, ~old),
        },
        "evaluation_seconds": elapsed,
        "samples_per_second": len(targets) / max(elapsed, 1e-12),
    }

    return metrics, diagnostics


def _comparison(
    baseline: dict[str, Any],
    ours: dict[str, float],
) -> dict[str, Any]:
    baseline_metrics = baseline["metrics"]

    deltas = {
        metric: float(ours[metric] - baseline_metrics[metric])
        for metric in ("all", "old", "new")
    }

    return {
        "baseline_method": baseline.get("method", "baseline"),
        "baseline_checkpoint_sha256": baseline.get(
            "checkpoint_sha256"
        ),
        "baseline_metrics": {
            key: float(baseline_metrics[key])
            for key in ("all", "old", "new")
        },
        "ours_metrics": {
            key: float(ours[key])
            for key in ("all", "old", "new")
        },
        "absolute_delta": deltas,
        "verdict": (
            "positive"
            if deltas["all"] > 0
            else "negative"
        ),
    }


def run(
    config_path: str | Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    raw = load_config(config_path)
    config = json.loads(json.dumps(raw))

    split = json.loads(
        Path(config["dataset"]["split_validation"]).read_text(
            encoding="utf-8"
        )
    )
    config["dataset"]["classes"] = len(
        set(split["known_class_ids"])
        | set(split["novel_class_ids"])
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("run requires exactly one visible GPU")

    device = torch.device("cuda:0")
    seed = int(config["seed"])
    seed_everything(seed)

    output = Path(config["output_directory"])
    result_path = output / "result.json"

    if resume and result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))

    output.mkdir(parents=True, exist_ok=True)

    resolved_text = yaml.safe_dump(config, sort_keys=True)
    resolved_path = output / "resolved_config.yaml"

    if (
        resolved_path.is_file()
        and resolved_path.read_text(encoding="utf-8")
        != resolved_text
    ):
        raise ValueError("output contains a different resolved config")

    _atomic_text(resolved_path, resolved_text)

    records = read_manifest(
        config["dataset"]["manifest"],
        dataset_root=config["dataset"]["root"],
        check_images=True,
    )

    train = ProductionManifestDataset(
        records,
        config["dataset"]["root"],
        split="train",
        seed=seed,
        train=True,
    )
    test = ProductionManifestDataset(
        records,
        config["dataset"]["root"],
        split="test",
        seed=seed,
        train=False,
    )

    model = construct_model(config, device=device)
    training = config["training"]
    method_config = _method_configuration(config)

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        train,
        batch_size=int(training["physical_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
        worker_init_fn=_worker_seed,
        persistent_workers=int(training["num_workers"]) > 0,
    )

    evaluation_loader = DataLoader(
        test,
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        worker_init_fn=_worker_seed,
    )

    epochs = int(training["epochs"])
    accumulation = int(training["gradient_accumulation"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        max(1, epochs * math.ceil(len(loader) / accumulation)),
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
    ).stdout

    provenance = {
        "schema_version": SCHEMA,
        "repository_commit": commit,
        "repository_dirty": bool(dirty.strip()),
        "dataset_manifest_sha256": sha256_file(
            config["dataset"]["manifest"]
        ),
        "split_validation_sha256": sha256_file(
            config["dataset"]["split_validation"]
        ),
        "backbone_checkpoint_sha256": sha256_file(
            config["backbone"]["checkpoint"]
        ),
        "baseline_result_sha256": sha256_file(
            config["baseline_result"]
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
    best_loss = float("inf")
    history: list[dict[str, Any]] = []

    last_path = output / "checkpoint_last.pt"
    best_path = output / "checkpoint_best.pt"

    if resume and last_path.is_file():
        state = load_checkpoint(last_path, map_location=device)

        if state.get("config_hash") != stable_hash(config):
            raise ValueError("resume config mismatch")

        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        best_loss = float(state["best_selection_loss"])

        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        generator.set_state(state["loader_generator"])

        history.append(
            {
                "resumed_epoch": start_epoch,
                "checkpoint_sha256": sha256_file(last_path),
            }
        )

    metrics_path = output / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, epochs):
        train.epoch = epoch
        model.train()

        sums: dict[str, float] = {}
        batches = 0

        for batch_index, batch in enumerate(loader):
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                raw_loss, stats = _training_batch(
                    model=model,
                    batch=batch,
                    training=training,
                    method_config=method_config,
                    device=device,
                )
                loss = raw_loss / accumulation

            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")

            loss.backward()

            for key, value in stats.items():
                sums[key] = sums.get(key, 0.0) + value
            batches += 1

            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(loader)
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
        selection_loss = means["total"]

        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "selection_loss": selection_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
            **means,
        }

        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        _checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
            best_loss=min(best_loss, selection_loss),
            config=config,
            loader_generator=generator,
        )

        if selection_loss < best_loss:
            best_loss = selection_loss
            _checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                global_step=global_step,
                best_loss=best_loss,
                config=config,
                loader_generator=generator,
            )

        print(
            f"epoch={epoch + 1:02d}/{epochs} "
            f"loss={selection_loss:.6f} "
            f"gate={means['mean_gate']:.4f} "
            f"k={means['mean_k']:.2f} "
            f"prototypes={int(means['active_prototypes'])}",
            flush=True,
        )

    best_state = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best_state["model"], strict=True)

    final_metrics, diagnostics = evaluate(
        model,
        evaluation_loader,
        device,
    )

    baseline = json.loads(
        Path(config["baseline_result"]).read_text(encoding="utf-8")
    )
    comparison = _comparison(baseline, final_metrics)

    elapsed = time.perf_counter() - started

    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "method": "novelty_preserving_deltasub",
        "dataset": config["dataset"]["name"],
        "seed": seed,
        "metrics": final_metrics,
        "diagnostics": diagnostics,
        "comparison_to_matched_baseline": comparison,
        "best_checkpoint_epoch": int(best_state["epoch"]),
        "best_checkpoint_selection_rule": (
            "minimum training objective; test labels used only "
            "after checkpoint restoration"
        ),
        "train_samples": len(train),
        "eval_samples": len(test),
        "runtime_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(
            device
        ),
        "total_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "active_known_prototypes": (
            model.prototype_bank.active_class_count
        ),
        "checkpoint_sha256": sha256_file(best_path),
        "resume_history": history,
    }

    _atomic_json(result_path, result)
    _atomic_json(output / "comparison.json", comparison)

    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def smoke(config_path: str | Path) -> None:
    raw = load_config(config_path)
    config = json.loads(json.dumps(raw))

    split = json.loads(
        Path(config["dataset"]["split_validation"]).read_text(
            encoding="utf-8"
        )
    )
    config["dataset"]["classes"] = len(
        set(split["known_class_ids"])
        | set(split["novel_class_ids"])
    )

    device = torch.device("cuda:0")
    seed_everything(int(config["seed"]))

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
        "views": torch.stack([item["views"] for item in items]),
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

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, stats = _training_batch(
            model=model,
            batch=batch,
            training=config["training"],
            method_config=_method_configuration(config),
            device=device,
        )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    if not gradients:
        raise RuntimeError("smoke test produced no trainable gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("smoke test produced invalid gradients")

    print("status: passed")
    print(f"loss: {float(loss.detach()):.6f}")
    print(f"mean gate: {stats['mean_gate']:.6f}")
    print(f"mean K: {stats['mean_k']:.2f}")
    print(
        "active prototypes: "
        f"{model.prototype_bank.active_class_count}"
    )
    print(
        "peak CUDA MiB: "
        f"{torch.cuda.max_memory_allocated(device) / 2**20:.2f}"
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
