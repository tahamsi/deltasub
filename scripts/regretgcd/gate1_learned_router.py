#!/usr/bin/env python
"""RegretGCD Gate 1: class-held-out learned arbitration.

This is the first deployable RegretGCD experiment.  Test labels are loaded for final
GCD-v2 evaluation only.  Router fitting, feature ablations, threshold calibration, and
model selection use genuinely labelled training examples from known classes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from deltasub.data.manifests import read_manifest
from deltasub.evaluation.gcd_v2 import evaluate_gcd_v2
from deltasub.experiment.training import MEAN, STD, _atomic_json, _worker_seed
from deltasub.regretgcd import (
    apply_mapping,
    build_prototypes,
    build_router_features,
    feature_indices,
    fit_regret_router,
    fixed_alignment_paired_bootstrap,
    hmean,
    hungarian_mapping,
    known_centroid_scores,
    normalize_rows,
    random_matched_switch,
    regret_labels,
    route_predictions,
)
from deltasub.utils.hashing import sha256_file
from deltasub.utils.reproducibility import seed_everything


SCHEMA = "regretgcd.gate1-result.v1"
CONFIG_SCHEMA = "regretgcd.gate1.v1"
VIEWS = ("original", "low_saturation", "mild_blur", "dimmed")


def load_ig_module():
    path = Path(__file__).parents[1] / "deltasub_v2" / "information_gain_append_gate.py"
    spec = importlib.util.spec_from_file_location("regretgcd_ig_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


IG = load_ig_module()


class RegretViews(Dataset):
    """Deterministic aligned views for labelled train or held-out test samples."""

    def __init__(self, records: list[dict[str, Any]], root: str | Path, split: str):
        if split == "labelled_train":
            self.records = [
                record
                for record in records
                if record["train_or_test_split"] == "train"
                and record["labelled_or_unlabelled"] == "labelled"
            ]
        elif split == "test":
            self.records = [
                record for record in records if record["train_or_test_split"] == "test"
            ]
        else:
            raise ValueError("split must be labelled_train or test")
        if not self.records:
            raise ValueError(f"empty RegretGCD split: {split}")
        self.root = Path(root)
        self.resize = transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        self.crop = transforms.CenterCrop(224)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def tensor(image: Image.Image) -> torch.Tensor:
        return TF.normalize(TF.to_tensor(image), MEAN, STD)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        relative = Path(record["image_path"])
        if relative.is_absolute():
            raise ValueError("production manifests must contain relative image paths")
        with Image.open(self.root / relative) as source:
            image = source.convert("RGB")
        original = self.crop(self.resize(image))
        views = (
            original,
            ImageEnhance.Color(original).enhance(0.25),
            original.filter(ImageFilter.GaussianBlur(radius=1.5)),
            ImageEnhance.Brightness(original).enhance(0.80),
        )
        return {
            "views": torch.stack([self.tensor(view) for view in views]),
            "target": int(record["original_class_id"]),
            "old": record["known_or_novel"] == "known",
            "sample_id": str(record["sample_id"]),
        }


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported RegretGCD Gate-1 configuration")
    if tuple(value["extraction"]["views"]) != VIEWS:
        raise ValueError(f"views must be exactly {VIEWS}")
    if int(value["router"]["folds"]) < 3:
        raise ValueError("at least three class-held-out folds are required")
    return value


def metric_dict(target: np.ndarray, prediction: np.ndarray, old: np.ndarray) -> dict[str, float]:
    raw = evaluate_gcd_v2(target, prediction, old).as_dict()
    old_value = float(raw["old"])
    new_value = float(raw["new"])
    return {
        "all": float(raw["all"]),
        "old": old_value,
        "new": new_value,
        "hmean": hmean(old_value, new_value),
    }


@torch.inference_mode()
def extract_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cache_path: Path,
    label: str,
) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    targets: list[int] = []
    old: list[bool] = []
    sample_ids: list[str] = []
    for batch_number, batch in enumerate(loader, 1):
        views = batch["views"].to(device)
        batch_size, view_count = views.shape[:2]
        flat = views.reshape(batch_size * view_count, 3, 224, 224)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls = model.backbone(flat).cls_token
            score = model.head(cls)
        features.append(cls.reshape(batch_size, view_count, -1).float().cpu().numpy())
        logits.append(score.reshape(batch_size, view_count, -1).float().cpu().numpy())
        targets.extend(batch["target"].tolist())
        old.extend(batch["old"].tolist())
        sample_ids.extend([str(value) for value in batch["sample_id"]])
        if batch_number % 20 == 0 or batch_number == len(loader):
            print(f"{label} batch {batch_number:04d}/{len(loader):04d}", flush=True)
    result = {
        "features": np.concatenate(features).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
        "target": np.asarray(targets, dtype=np.int64),
        "old": np.asarray(old, dtype=bool),
        "sample_id": np.asarray(sample_ids, dtype=str),
    }
    atomic_npz(cache_path, **result)
    return result


@torch.inference_mode()
def logits_from_features(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    flat = np.asarray(features, dtype=np.float32).reshape(-1, features.shape[-1])
    values: list[np.ndarray] = []
    for start in range(0, len(flat), batch_size):
        tensor = torch.from_numpy(flat[start : start + batch_size]).to(device)
        values.append(model.head(tensor).float().cpu().numpy())
    return np.concatenate(values).reshape(*features.shape[:-1], -1).astype(np.float32)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    required = {"features", "logits", "target", "old", "sample_id"}
    missing = required - set(archive.files)
    if missing:
        raise KeyError(f"cache {path} lacks {sorted(missing)}")
    return {key: archive[key] for key in archive.files}


def prepare_test_cache(
    *,
    model: torch.nn.Module,
    dataset: RegretViews,
    loader: DataLoader,
    device: torch.device,
    reusable: Path,
    cache_path: Path,
    resume: bool,
) -> dict[str, np.ndarray]:
    if resume and cache_path.is_file():
        print("reusing RegretGCD test cache", flush=True)
        return load_cache(cache_path)
    if reusable.is_file():
        archive = np.load(reusable, allow_pickle=False)
        required = {"features", "target", "old", "sample_id"}
        if required <= set(archive.files) and archive["features"].shape[:2] == (len(dataset), len(VIEWS)):
            print("reusing prior multi-view test features; computing only head logits", flush=True)
            result = {
                "features": archive["features"].astype(np.float32),
                "logits": logits_from_features(model, archive["features"], device),
                "target": archive["target"].astype(np.int64),
                "old": archive["old"].astype(bool),
                "sample_id": archive["sample_id"].astype(str),
            }
            atomic_npz(cache_path, **result)
            return result
    return extract_split(model, loader, device, cache_path, "test")


def select_alpha(
    parametric_probability: np.ndarray,
    prototype_probability: np.ndarray,
    target: np.ndarray,
    mapping: dict[int, int],
) -> tuple[float, float]:
    best_key: tuple[float, float] | None = None
    best_alpha = 0.5
    best_accuracy = -1.0
    for alpha in np.linspace(0.0, 1.0, 101):
        prediction = (
            alpha * parametric_probability + (1.0 - alpha) * prototype_probability
        ).argmax(axis=1)
        accuracy = float(np.mean(apply_mapping(prediction, mapping) == target))
        key = (accuracy, -abs(float(alpha) - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_alpha = float(alpha)
            best_accuracy = accuracy
    return best_alpha, best_accuracy


def deterministic_oracle(
    target: np.ndarray,
    parametric: np.ndarray,
    prototype: np.ndarray,
) -> np.ndarray:
    p_map = hungarian_mapping(parametric, target)
    t_map = hungarian_mapping(prototype, target)
    p_correct = apply_mapping(parametric, p_map) == target
    t_correct = apply_mapping(prototype, t_map) == target
    result = prototype.copy()
    result[p_correct & ~t_correct] = parametric[p_correct & ~t_correct]
    return result


def write_router(path: Path, name: str, fit: Any) -> None:
    atomic_npz(
        path,
        mean=np.asarray(fit.mean, dtype=np.float64),
        scale=np.asarray(fit.scale, dtype=np.float64),
        coefficient=np.asarray(fit.coefficient, dtype=np.float64),
        intercept=np.asarray([fit.intercept], dtype=np.float64),
        threshold=np.asarray([fit.threshold], dtype=np.float64),
        feature_names=np.asarray(fit.feature_names, dtype=str),
    )
    _atomic_json(
        path.with_suffix(".json"),
        {
            "name": name,
            "feature_names": list(fit.feature_names),
            "threshold": fit.threshold,
            "oof_auc": fit.oof_auc,
            "oof_accuracy": fit.oof_accuracy,
            "oof_switch_rate": fit.oof_switch_rate,
            "unique_winners": fit.unique_winners,
            "positive_winners": fit.positive_winners,
            "negative_winners": fit.negative_winners,
            "fold_kind": fit.fold_kind,
            "fold_count": fit.fold_count,
        },
    )


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("RegretGCD Gate 1 requires exactly one visible CUDA GPU")

    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if args.resume and result_path.is_file():
        value = json.loads(result_path.read_text(encoding="utf-8"))
        print("REGRETGCD GATE-1 VERDICT:", value["verdict"].upper())
        print("result:", result_path)
        return

    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    model, baseline_config, _, provenance = IG.load_matched_baseline(device)
    model.requires_grad_(False).eval()
    records = read_manifest(
        baseline_config["dataset"]["manifest"],
        dataset_root=baseline_config["dataset"]["root"],
        check_images=True,
    )
    train_dataset = RegretViews(records, baseline_config["dataset"]["root"], "labelled_train")
    test_dataset = RegretViews(records, baseline_config["dataset"]["root"], "test")
    loader_kwargs = {
        "batch_size": int(config["extraction"]["batch_size"]),
        "shuffle": False,
        "num_workers": int(config["extraction"]["num_workers"]),
        "worker_init_fn": _worker_seed,
        "persistent_workers": int(config["extraction"]["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    train_cache_path = output / "labelled_train_cache.npz"
    test_cache_path = output / "test_cache.npz"
    train_data = (
        load_cache(train_cache_path)
        if args.resume and train_cache_path.is_file()
        else extract_split(model, train_loader, device, train_cache_path, "labelled-train")
    )
    test_data = prepare_test_cache(
        model=model,
        dataset=test_dataset,
        loader=test_loader,
        device=device,
        reusable=Path(config["source"]["reusable_test_features"]),
        cache_path=test_cache_path,
        resume=args.resume,
    )

    source_archive_path = Path(config["source"]["prototype_archive"])
    source_result_path = Path(config["source"]["prototype_result"])
    gate0_result_path = Path(config["source"]["gate0_result"])
    if not all(path.is_file() for path in (source_archive_path, source_result_path, gate0_result_path)):
        raise FileNotFoundError("Gate 0 source artifacts are missing")
    gate0_result = json.loads(gate0_result_path.read_text(encoding="utf-8"))
    if gate0_result.get("verdict") != "continue":
        raise RuntimeError("RegretGCD Gate 0 did not authorize learned-router training")
    source = np.load(source_archive_path, allow_pickle=False)
    required = {
        "target",
        "old",
        "matched_selex_prediction",
        "prototype_control_prediction",
    }
    missing = required - set(source.files)
    if missing:
        raise KeyError(f"prototype archive lacks {sorted(missing)}")

    test_target = test_data["target"].astype(np.int64)
    test_old = test_data["old"].astype(bool)
    if not np.array_equal(test_target, source["target"].astype(np.int64)):
        raise RuntimeError("test target order differs from Gate 0 archive")
    if not np.array_equal(test_old, source["old"].astype(bool)):
        raise RuntimeError("test Old/New mask differs from Gate 0 archive")
    test_parametric = source["matched_selex_prediction"].astype(np.int64)
    test_prototype = source["prototype_control_prediction"].astype(np.int64)

    train_features = normalize_rows(train_data["features"].astype(np.float64))
    test_features = normalize_rows(test_data["features"].astype(np.float64))
    train_mean = normalize_rows(train_features.mean(axis=1))
    test_mean = normalize_rows(test_features.mean(axis=1))
    prototype_state = build_prototypes(test_mean, test_parametric)

    train_known_scores = known_centroid_scores(
        train_mean,
        train_data["target"].astype(np.int64),
        train_mean,
        leave_one_out=True,
    )
    test_known_scores = known_centroid_scores(
        train_mean,
        train_data["target"].astype(np.int64),
        test_mean,
        leave_one_out=False,
    )

    train_parametric = train_data["logits"][:, 0].argmax(axis=1).astype(np.int64)
    train_router = build_router_features(
        parametric_logits=train_data["logits"],
        view_features=train_features,
        prototype_state=prototype_state,
        known_scores=train_known_scores,
        parametric_hard=train_parametric,
        test_leave_one_out=False,
        prototype_temperature=float(config["prototype"]["temperature"]),
    )
    test_router = build_router_features(
        parametric_logits=test_data["logits"],
        view_features=test_features,
        prototype_state=prototype_state,
        known_scores=test_known_scores,
        parametric_hard=test_parametric,
        test_leave_one_out=True,
        prototype_temperature=float(config["prototype"]["temperature"]),
    )

    prototype_reconstruction_mismatches = int(
        np.sum(test_router.prototype_prediction != test_prototype)
    )
    # Hard decisions remain the archived Gate-0 experts.  Reconstructed probabilities
    # are only router features, and the mismatch count is disclosed in the result.
    test_router_prototype_hard = test_prototype

    train_target = train_data["target"].astype(np.int64)
    mapping = hungarian_mapping(train_router.parametric_prediction, train_target)

    regret_target, unique_winner, _, _ = regret_labels(
        train_router.parametric_prediction,
        train_router.prototype_prediction,
        train_target,
        mapping,
    )
    prototype_wins = int(regret_target[unique_winner].sum())
    parametric_wins = int(unique_winner.sum() - prototype_wins)

    if prototype_wins == 0 or parametric_wins == 0:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        result = {
            "schema_version": SCHEMA,
            "status": "completed",
            "dataset": config["dataset"],
            "seed": seed,
            "method": "RegretGCD",
            "gate": "failed",
            "verdict": "abandon",
            "failure_reason": "single_class_unique_winner_supervision",
            "failure_explanation": (
                "The labelled known-class unique-winner set contains only one "
                "expert outcome, so the binary regret target is not identifiable "
                "without synthetic supervision or test-label leakage."
            ),
            "test_labels_used_for_router_training": False,
            "test_labels_used_for_threshold_selection": False,
            "transductive_unlabelled_test_features_used": True,
            "class_holdout_router": True,
            "router_diagnostics": {
                "unique_winners": int(unique_winner.sum()),
                "prototype_wins": prototype_wins,
                "parametric_wins": parametric_wins,
                "winner_classes_present": int(
                    len(np.unique(regret_target[unique_winner]))
                ),
                "oof_auc": None,
                "router_fitted": False,
            },
            "gate0_metrics": gate0_result.get("metrics", {}),
            "next_stage_if_continue": None,
            "cars": {
                "status": "not_run",
                "reason": "dataset unavailable by user choice",
            },
            "source": {
                "prototype_archive": str(source_archive_path),
                "prototype_archive_sha256": sha256_file(source_archive_path),
                "prototype_result": str(source_result_path),
                "prototype_result_sha256": sha256_file(source_result_path),
                "gate0_result": str(gate0_result_path),
                "gate0_result_sha256": sha256_file(gate0_result_path),
                "gate0_verdict": gate0_result.get("verdict"),
                "matched_baseline_result": provenance["result"],
                "matched_baseline_checkpoint": provenance["checkpoint"],
                "matched_baseline_checkpoint_sha256": provenance[
                    "checkpoint_sha256"
                ],
            },
            "provenance": {
                "repository_commit": commit,
                "config": str(args.config),
                "config_sha256": sha256_file(args.config),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda": str(torch.version.cuda),
                "device": torch.cuda.get_device_name(),
            },
            "runtime_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        _atomic_json(result_path, result)

        print()
        print("===== REGRETGCD GATE 1: IDENTIFIABILITY CHECK =====")
        print(f"unique winners  : {int(unique_winner.sum())}")
        print(f"parametric wins : {parametric_wins}")
        print(f"prototype wins  : {prototype_wins}")
        print("router fitted   : no")
        print()
        print("REGRETGCD GATE-1 VERDICT: ABANDON")
        print("reason: single-class unique-winner supervision")
        print("result:", result_path)
        return

    folds = int(config["router"]["folds"])
    feature_sets = {
        "regretgcd": feature_indices(
            train_router.names,
            groups=train_router.groups,
        ),
        "without_stability": feature_indices(
            train_router.names,
            groups=train_router.groups,
            exclude_groups=("stability",),
        ),
        "without_density": feature_indices(
            train_router.names,
            groups=train_router.groups,
            exclude_groups=("density",),
        ),
        "without_knownness": feature_indices(
            train_router.names,
            groups=train_router.groups,
            exclude_groups=("knownness",),
        ),
        "without_cross_expert": feature_indices(
            train_router.names,
            groups=train_router.groups,
            exclude_groups=("cross",),
        ),
        "knownness_only": feature_indices(
            train_router.names,
            groups=train_router.groups,
            include_groups=("knownness",),
        ),
    }

    fits: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {
        "matched_selex": test_parametric,
        "prototype_control": test_prototype,
    }
    router_probabilities: dict[str, np.ndarray] = {}
    for index, (name, selected_features) in enumerate(feature_sets.items()):
        fit, _ = fit_regret_router(
            features=train_router.matrix,
            feature_names=train_router.names,
            target=train_target,
            class_ids=train_target,
            sample_ids=train_data["sample_id"].astype(str).tolist(),
            parametric_prediction=train_router.parametric_prediction,
            prototype_prediction=train_router.prototype_prediction,
            mapping=mapping,
            feature_index=selected_features,
            folds=folds,
            seed=seed + 100 * index,
            fold_kind="class",
        )
        probability = fit.predict_proba(test_router.matrix[:, selected_features])
        predictions[name] = route_predictions(
            test_parametric,
            test_router_prototype_hard,
            probability,
            fit.threshold,
        )
        router_probabilities[name] = probability
        fits[name] = fit
        write_router(output / f"router_{name}.npz", name, fit)

    full_features = feature_sets["regretgcd"]
    instance_fit, _ = fit_regret_router(
        features=train_router.matrix,
        feature_names=train_router.names,
        target=train_target,
        class_ids=train_target,
        sample_ids=train_data["sample_id"].astype(str).tolist(),
        parametric_prediction=train_router.parametric_prediction,
        prototype_prediction=train_router.prototype_prediction,
        mapping=mapping,
        feature_index=full_features,
        folds=folds,
        seed=seed + 700,
        fold_kind="instance",
    )
    instance_probability = instance_fit.predict_proba(test_router.matrix[:, full_features])
    predictions["instance_holdout_router"] = route_predictions(
        test_parametric,
        test_router_prototype_hard,
        instance_probability,
        instance_fit.threshold,
    )
    router_probabilities["instance_holdout_router"] = instance_probability
    fits["instance_holdout_router"] = instance_fit
    write_router(output / "router_instance_holdout.npz", "instance_holdout_router", instance_fit)

    name_to_index = {name: index for index, name in enumerate(test_router.names)}
    use_prototype_confidence = (
        test_router.matrix[:, name_to_index["t_confidence"]]
        > test_router.matrix[:, name_to_index["p_confidence"]]
    )
    predictions["maximum_confidence"] = np.where(
        use_prototype_confidence,
        test_prototype,
        test_parametric,
    ).astype(np.int64)
    use_prototype_entropy = (
        test_router.matrix[:, name_to_index["t_entropy"]]
        < test_router.matrix[:, name_to_index["p_entropy"]]
    )
    predictions["minimum_entropy"] = np.where(
        use_prototype_entropy,
        test_prototype,
        test_parametric,
    ).astype(np.int64)

    alpha, train_blend_accuracy = select_alpha(
        train_router.parametric_probability,
        train_router.prototype_probability,
        train_target,
        mapping,
    )
    predictions["global_probability_blend"] = (
        alpha * test_router.parametric_probability
        + (1.0 - alpha) * test_router.prototype_probability
    ).argmax(axis=1).astype(np.int64)

    full_probability = router_probabilities["regretgcd"]
    full_switch = (
        (test_parametric != test_prototype)
        & (full_probability >= fits["regretgcd"].threshold)
    )
    predictions["random_matched_switch_rate"] = random_matched_switch(
        test_parametric,
        test_prototype,
        int(full_switch.sum()),
        test_data["sample_id"].astype(str).tolist(),
        seed + 900,
    )

    # Test labels enter only after every deployable prediction above is finalized.
    predictions["oracle_arbitration_diagnostic"] = deterministic_oracle(
        test_target,
        test_parametric,
        test_prototype,
    )
    metrics = {
        name: metric_dict(test_target, prediction, test_old)
        for name, prediction in predictions.items()
    }

    reference = metrics["prototype_control"]
    candidate = metrics["regretgcd"]
    delta = {
        key: candidate[key] - reference[key]
        for key in ("all", "old", "new", "hmean")
    }
    interval = fixed_alignment_paired_bootstrap(
        target=test_target,
        old=test_old,
        baseline_prediction=predictions["prototype_control"],
        candidate_prediction=predictions["regretgcd"],
        draws=int(config["router"]["bootstrap_draws"]),
        seed=seed + 1900,
    )
    non_oracle_controls = [
        name
        for name in predictions
        if name not in {"regretgcd", "oracle_arbitration_diagnostic"}
    ]
    best_control = max(non_oracle_controls, key=lambda name: metrics[name]["hmean"])
    gate = config["gate"]
    auc = fits["regretgcd"].oof_auc
    passed = (
        delta["all"] >= float(gate["all_delta_minimum"])
        and delta["old"] >= float(gate["old_delta_minimum"])
        and delta["new"] >= float(gate["new_delta_minimum"])
        and delta["hmean"] >= float(gate["hmean_delta_minimum"])
        and interval["lower_95"] > 0.0
        and candidate["hmean"] > metrics[best_control]["hmean"]
        and auc is not None
        and auc >= float(config["router"]["minimum_oof_auc"])
    )
    verdict = "continue" if passed else "abandon"

    atomic_npz(
        output / "predictions.npz",
        target=test_target,
        old=test_old,
        sample_id=test_data["sample_id"].astype(str),
        **{f"{name}_prediction": value for name, value in predictions.items()},
        **{f"{name}_probability": value for name, value in router_probabilities.items()},
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "all", "old", "new", "hmean"])
        for name, value in metrics.items():
            writer.writerow([name, value["all"], value["old"], value["new"], value["hmean"]])

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    for method in ("matched_selex", "prototype_control"):
        for key in ("all", "old", "new", "hmean"):
            expected = float(source_result["metrics"][method][key])
            actual = float(metrics[method][key])
            if abs(expected - actual) > 1.0e-12:
                raise RuntimeError(
                    f"Gate-0 source mismatch for {method}.{key}: {expected} != {actual}"
                )
    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "dataset": config["dataset"],
        "seed": seed,
        "method": "RegretGCD",
        "test_labels_used_for_router_training": False,
        "test_labels_used_for_threshold_selection": False,
        "test_labels_used_for_final_evaluation": True,
        "transductive_unlabelled_test_features_used": True,
        "class_holdout_router": True,
        "experts": {
            "parametric": "matched SelEx classifier",
            "prototype": "four-view leave-one-out transductive prototype classifier",
        },
        "metrics": metrics,
        "delta_vs_prototype_control": delta,
        "hmean_fixed_alignment_paired_bootstrap": interval,
        "best_non_oracle_control": {
            "name": best_control,
            "hmean": metrics[best_control]["hmean"],
        },
        "router_diagnostics": {
            name: {
                "features": list(fit.feature_names),
                "threshold": fit.threshold,
                "oof_auc": fit.oof_auc,
                "oof_accuracy": fit.oof_accuracy,
                "oof_switch_rate": fit.oof_switch_rate,
                "unique_winners": fit.unique_winners,
                "prototype_wins": fit.positive_winners,
                "parametric_wins": fit.negative_winners,
                "fold_kind": fit.fold_kind,
            }
            for name, fit in fits.items()
        },
        "controls": {
            "global_blend_alpha": alpha,
            "global_blend_labelled_train_accuracy": train_blend_accuracy,
            "regretgcd_test_switch_count": int(full_switch.sum()),
            "regretgcd_test_switch_rate": float(full_switch.mean()),
        },
        "prototype_diagnostics": {
            "cluster_count": int(len(prototype_state.labels)),
            "minimum_cluster_size": int(prototype_state.counts.min()),
            "maximum_cluster_size": int(prototype_state.counts.max()),
            "reconstructed_hard_prediction_mismatches": prototype_reconstruction_mismatches,
            "archived_gate0_hard_predictions_used": True,
        },
        "gate_rule": {
            **gate,
            "minimum_oof_auc": float(config["router"]["minimum_oof_auc"]),
            "reference": "prototype_control",
        },
        "gate": "passed" if passed else "failed",
        "verdict": verdict,
        "next_stage_if_continue": {
            "cub_seeds": [0, 1, 2],
            "aircraft_seeds": [0, 1, 2],
            "cars": {
                "status": "not_run",
                "reason": "dataset unavailable by user choice",
            },
            "required_comparison": "matched SOTA protocols only",
        },
        "source": {
            "prototype_archive": str(source_archive_path),
            "prototype_archive_sha256": sha256_file(source_archive_path),
            "prototype_result": str(source_result_path),
            "prototype_result_sha256": sha256_file(source_result_path),
            "gate0_result": str(gate0_result_path),
            "gate0_result_sha256": sha256_file(gate0_result_path),
            "gate0_verdict": gate0_result.get("verdict"),
            "matched_baseline_result": provenance["result"],
            "matched_baseline_checkpoint": provenance["checkpoint"],
            "matched_baseline_checkpoint_sha256": provenance["checkpoint_sha256"],
        },
        "provenance": {
            "repository_commit": commit,
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "device": torch.cuda.get_device_name(),
        },
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    _atomic_json(result_path, result)

    print()
    print("===== REGRETGCD GATE 1: CLASS-HELD-OUT LEARNED ROUTER =====")
    print("method                              all       old       new      hmean")
    order = [
        "matched_selex",
        "prototype_control",
        "regretgcd",
        "maximum_confidence",
        "minimum_entropy",
        "knownness_only",
        "global_probability_blend",
        "random_matched_switch_rate",
        "instance_holdout_router",
        "without_stability",
        "without_density",
        "without_knownness",
        "without_cross_expert",
        "oracle_arbitration_diagnostic",
    ]
    for name in order:
        value = metrics[name]
        print(
            f"{name:<36}{value['all']:>9.6f}{value['old']:>10.6f}"
            f"{value['new']:>10.6f}{value['hmean']:>11.6f}"
        )
    print()
    print("delta versus prototype control")
    for key in ("all", "old", "new", "hmean"):
        print(f"{key:<5}: {delta[key]:+.6f}")
    print(
        "H95 fixed-alignment paired bootstrap: "
        f"[{interval['lower_95']:+.6f}, {interval['upper_95']:+.6f}]"
    )
    print(f"class-held-out unique-winner AUC: {auc if auc is not None else 'undefined'}")
    print(f"best non-oracle control: {best_control} ({metrics[best_control]['hmean']:.6f})")
    print(f"prototype reconstruction mismatches: {prototype_reconstruction_mismatches}")
    print()
    print("REGRETGCD GATE-1 VERDICT:", verdict.upper())
    print("result:", result_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/regretgcd/cub_seed0.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
