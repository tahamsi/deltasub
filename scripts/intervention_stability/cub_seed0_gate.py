#!/usr/bin/env python
"""Frozen intervention-stability feasibility gate for GCD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from deltasub.data.manifests import read_manifest
from deltasub.evaluation.gcd_v2 import evaluate_gcd_v2
from deltasub.experiment.deltasub_residual_compare import (
    bootstrap_hmean_delta,
    locate_matched_baseline,
)
from deltasub.experiment.deltasub_v2_training import construct_model
from deltasub.experiment.training import MEAN, STD, _atomic_json, _worker_seed
from deltasub.utils.checkpointing import load_checkpoint
from deltasub.utils.hashing import sha256_file
from deltasub.utils.reproducibility import seed_everything


SCHEMA = "intervention-stability.gate.v1"
VIEW_NAMES = (
    "original",
    "horizontal_flip",
    "low_saturation",
    "mild_blur",
    "center_zoom",
)
ANCHOR_FRACTIONS = (0.50, 0.70, 0.90)


class InterventionDataset(Dataset):
    def __init__(self, records, root, limit=0):
        self.records = [
            record
            for record in records
            if record["train_or_test_split"] == "test"
        ]
        if limit:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError("empty test partition")

        self.root = Path(root)
        self.resize = transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        self.crop = transforms.CenterCrop(224)

    def __len__(self):
        return len(self.records)

    @staticmethod
    def tensor(image):
        return TF.normalize(
            TF.to_tensor(image),
            MEAN,
            STD,
        )

    def __getitem__(self, index):
        record = self.records[index]
        path = Path(record["image_path"])
        if path.is_absolute():
            raise ValueError("manifest paths must be relative")

        with Image.open(self.root / path) as source:
            image = source.convert("RGB")

        original = self.crop(self.resize(image))

        zoom = original.crop((14, 14, 210, 210)).resize(
            (224, 224),
            Image.Resampling.BICUBIC,
        )

        views = (
            original,
            ImageOps.mirror(original),
            ImageEnhance.Color(original).enhance(0.25),
            original.filter(ImageFilter.GaussianBlur(radius=1.5)),
            zoom,
        )

        return {
            "views": torch.stack([self.tensor(view) for view in views]),
            "target": int(record["original_class_id"]),
            "old": record["known_or_novel"] == "known",
            "sample_id": str(record["sample_id"]),
        }


def hmean(old, new):
    return (
        2.0 * old * new / (old + new)
        if old + new > 0
        else 0.0
    )


def metrics(target, prediction, old):
    value = evaluate_gcd_v2(
        target,
        prediction,
        old,
    ).as_dict()

    return {
        "all": float(value["all"]),
        "old": float(value["old"]),
        "new": float(value["new"]),
        "hmean": hmean(
            float(value["old"]),
            float(value["new"]),
        ),
    }


def assignment_correctness(target, prediction):
    dimension = int(max(target.max(), prediction.max())) + 1
    contingency = np.zeros(
        (dimension, dimension),
        dtype=np.int64,
    )
    np.add.at(contingency, (prediction, target), 1)

    rows, columns = linear_sum_assignment(
        contingency.max() - contingency
    )

    mapping = np.full(dimension, -1, dtype=np.int64)
    mapping[rows] = columns

    return mapping[prediction] == target


def auc_score(score, positive):
    positive = np.asarray(positive, dtype=bool)
    negative = ~positive
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())

    if positive_count == 0 or negative_count == 0:
        return 0.5

    ranks = rankdata(score, method="average")

    return float(
        (
            ranks[positive].sum()
            - positive_count * (positive_count + 1) / 2
        )
        / (positive_count * negative_count)
    )


def knn_indices(features, k, device):
    tensor = F.normalize(
        torch.from_numpy(features).to(device),
        dim=1,
    )
    count = tensor.shape[0]
    chunks = []

    for start in range(0, count, 512):
        end = min(start + 512, count)
        similarity = tensor[start:end] @ tensor.T

        local = torch.arange(end - start, device=device)
        global_indices = torch.arange(start, end, device=device)
        similarity[local, global_indices] = float("-inf")

        chunks.append(
            similarity.topk(k=k, dim=1).indices.cpu().numpy()
        )

    return np.concatenate(chunks, axis=0)


def neighbourhood_overlap(first, second):
    return (
        first[:, :, None] == second[:, None, :]
    ).any(axis=2).mean(axis=1)


def normalized_rows(array):
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def centroids(features, labels, anchors):
    unique = np.unique(labels)
    values = []

    for label in unique:
        selected = anchors & (labels == label)
        if not selected.any():
            selected = labels == label

        values.append(features[selected].mean(axis=0))

    return unique, normalized_rows(np.stack(values))


def assign_to_centroids(features, labels, anchors):
    centroid_labels, centroid_values = centroids(
        features,
        labels,
        anchors,
    )

    result = labels.copy()
    query = ~anchors

    if query.any():
        similarity = features[query] @ centroid_values.T
        result[query] = centroid_labels[
            similarity.argmax(axis=1)
        ]

    return result


def stable_anchors(labels, stability, fraction):
    anchors = np.zeros(labels.shape[0], dtype=bool)

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        count = max(1, int(math.ceil(len(indices) * fraction)))
        order = indices[np.argsort(stability[indices])]
        anchors[order[-count:]] = True

    return anchors


def random_anchors(labels, stable_mask, seed):
    generator = np.random.default_rng(seed)
    anchors = np.zeros(labels.shape[0], dtype=bool)

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        count = int((stable_mask & (labels == label)).sum())
        anchors[
            generator.choice(indices, size=count, replace=False)
        ] = True

    return anchors


def majority_vote(view_predictions, baseline):
    result = baseline.copy()

    for index, row in enumerate(view_predictions):
        values, counts = np.unique(row, return_counts=True)
        maximum = counts.max()
        tied = values[counts == maximum]

        if baseline[index] in tied:
            result[index] = baseline[index]
        else:
            result[index] = tied.min()

    return result


def load_baseline(device):
    result_path, result = locate_matched_baseline()
    config_path = result_path.parent / "resolved_config.yaml"
    checkpoint_path = result_path.parent / "checkpoint_last.pt"
    prediction_path = result_path.parent / "predictions.npz"

    for path in (config_path, checkpoint_path, prediction_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    model = construct_model(config, device=device)
    state = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False).eval()

    return model, config, result, {
        "result": str(result_path),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "predictions": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
    }


@torch.inference_mode()
def extract(model, loader, device):
    feature_batches = []
    logit_batches = []
    targets = []
    old_values = []
    sample_ids = []

    started = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        views = batch["views"].to(device)
        batch_size, view_count = views.shape[:2]

        output = model.branches(
            views.reshape(batch_size * view_count, 3, 224, 224)
        )

        feature_batches.append(
            output.features.reshape(
                batch_size,
                view_count,
                -1,
            ).float().cpu().numpy()
        )
        logit_batches.append(
            output.logits.reshape(
                batch_size,
                view_count,
                -1,
            ).float().cpu().numpy()
        )
        targets.extend(batch["target"].tolist())
        old_values.extend(batch["old"].tolist())
        sample_ids.extend(batch["sample_id"])

        print(
            f"feature batch {batch_index + 1:04d}/{len(loader):04d}",
            flush=True,
        )

    return {
        "features": np.concatenate(feature_batches),
        "logits": np.concatenate(logit_batches),
        "target": np.asarray(targets, dtype=np.int64),
        "old": np.asarray(old_values, dtype=bool),
        "sample_id": np.asarray(sample_ids),
        "seconds": time.perf_counter() - started,
    }


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible GPU is required")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"

    if args.resume and result_path.is_file():
        value = json.loads(result_path.read_text(encoding="utf-8"))
        print("INTERVENTION-STABILITY VERDICT:", value["verdict"].upper())
        print("result:", result_path)
        return

    seed_everything(0)
    device = torch.device("cuda:0")

    model, config, baseline_result, provenance = load_baseline(device)

    records = read_manifest(
        config["dataset"]["manifest"],
        dataset_root=config["dataset"]["root"],
        check_images=True,
    )

    dataset = InterventionDataset(
        records,
        config["dataset"]["root"],
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=_worker_seed,
        persistent_workers=True,
    )

    extracted = extract(model, loader, device)

    reference = np.load(
        provenance["predictions"],
        allow_pickle=False,
    )

    limit = len(dataset)
    target = extracted["target"]
    old = extracted["old"]
    baseline = reference["fused_prediction"][:limit].astype(np.int64)

    if not np.array_equal(
        target,
        reference["target"][:limit].astype(np.int64),
    ):
        raise RuntimeError("test target order differs from baseline archive")
    if not np.array_equal(
        old,
        reference["old"][:limit].astype(bool),
    ):
        raise RuntimeError("Old/New mask differs from baseline archive")

    features = normalized_rows(
        extracted["features"].reshape(-1, 768)
    ).reshape(limit, len(VIEW_NAMES), 768)

    logits = extracted["logits"]
    view_predictions = logits.argmax(axis=2).astype(np.int64)
    original_prediction = view_predictions[:, 0]

    mismatch = original_prediction != baseline
    if mismatch.any():
        raise RuntimeError(
            "exact original forward differs from archived baseline: "
            f"{int(mismatch.sum())} predictions"
        )

    mean_features = normalized_rows(features.mean(axis=1))
    original_features = features[:, 0]

    assignment_agreement = (
        view_predictions[:, 1:] == baseline[:, None]
    ).mean(axis=1)

    feature_consistency = (
        original_features[:, None, :]
        * features[:, 1:, :]
    ).sum(axis=2).mean(axis=1)

    original_knn = knn_indices(
        original_features,
        k=10,
        device=device,
    )
    mean_knn = knn_indices(
        mean_features,
        k=10,
        device=device,
    )
    neighbour_consistency = neighbourhood_overlap(
        original_knn,
        mean_knn,
    )

    stability = (
        0.40 * assignment_agreement
        + 0.30 * np.clip(
            (feature_consistency + 1.0) / 2.0,
            0.0,
            1.0,
        )
        + 0.30 * neighbour_consistency
    )

    predictions = {
        "baseline": baseline,
        "majority_vote": majority_vote(
            view_predictions,
            baseline,
        ),
    }

    all_centroid_labels, all_centroid_values = centroids(
        mean_features,
        baseline,
        np.ones(limit, dtype=bool),
    )
    predictions["mean_centroid"] = all_centroid_labels[
        (mean_features @ all_centroid_values.T).argmax(axis=1)
    ]

    anchor_reports = {}

    for fraction in ANCHOR_FRACTIONS:
        suffix = str(int(round(fraction * 100)))
        anchors = stable_anchors(
            baseline,
            stability,
            fraction,
        )
        random_mask = random_anchors(
            baseline,
            anchors,
            seed=41_000 + int(fraction * 100),
        )

        predictions[f"stability_anchor_{suffix}"] = (
            assign_to_centroids(
                mean_features,
                baseline,
                anchors,
            )
        )
        predictions[f"random_anchor_{suffix}"] = (
            assign_to_centroids(
                mean_features,
                baseline,
                random_mask,
            )
        )

        anchor_reports[suffix] = {
            "fraction": fraction,
            "anchor_count": int(anchors.sum()),
            "reassigned_count": int((~anchors).sum()),
        }

    metric_values = {
        name: metrics(target, prediction, old)
        for name, prediction in predictions.items()
    }
    baseline_metrics = metric_values["baseline"]

    correct = assignment_correctness(target, baseline)
    stability_auc = auc_score(stability, correct)
    stability_gap = float(
        stability[correct].mean() - stability[~correct].mean()
    )

    gates = {}

    for fraction in ANCHOR_FRACTIONS:
        suffix = str(int(round(fraction * 100)))
        candidate_name = f"stability_anchor_{suffix}"
        random_name = f"random_anchor_{suffix}"

        candidate_metrics = metric_values[candidate_name]
        delta = {
            name: candidate_metrics[name] - baseline_metrics[name]
            for name in ("all", "old", "new", "hmean")
        }

        interval = bootstrap_hmean_delta(
            targets=target,
            old=old,
            baseline=baseline,
            candidate=predictions[candidate_name],
            draws=args.bootstrap_draws,
            seed=52_000 + int(fraction * 100),
        )

        best_control_hmean = max(
            metric_values[random_name]["hmean"],
            metric_values["mean_centroid"]["hmean"],
            metric_values["majority_vote"]["hmean"],
        )

        passed = (
            delta["all"] >= 0.0
            and delta["new"] >= 0.005
            and delta["hmean"] >= 0.005
            and interval["lower_95"] > 0.0
            and candidate_metrics["hmean"] >= best_control_hmean
            and stability_auc >= 0.60
            and stability_gap >= 0.05
        )

        gates[suffix] = {
            "candidate": candidate_name,
            "delta": delta,
            "hmean_paired_bootstrap": interval,
            "best_control_hmean": best_control_hmean,
            "passed": passed,
        }

    passing = [
        suffix
        for suffix, gate in gates.items()
        if gate["passed"]
    ]
    verdict = "continue" if passing else "abandon"

    np.savez_compressed(
        output / "predictions.npz",
        target=target,
        old=old,
        stability=stability.astype(np.float32),
        assignment_agreement=assignment_agreement.astype(np.float32),
        feature_consistency=feature_consistency.astype(np.float32),
        neighbour_consistency=neighbour_consistency.astype(np.float32),
        **{
            f"{name}_prediction": prediction
            for name, prediction in predictions.items()
        },
    )

    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "dataset": "cub",
        "seed": 0,
        "method": "intervention_stable_category_refinement",
        "training_used": False,
        "test_labels_used_for_refinement": False,
        "test_labels_used_for_selection": False,
        "interventions": list(VIEW_NAMES),
        "baseline": baseline_metrics,
        "metrics": metric_values,
        "gates": gates,
        "passing_anchor_fractions": passing,
        "stability_diagnostics": {
            "error_prediction_auc": stability_auc,
            "correct_minus_incorrect_mean_gap": stability_gap,
            "correct_mean": float(stability[correct].mean()),
            "incorrect_mean": float(stability[~correct].mean()),
            "assignment_agreement_mean": float(
                assignment_agreement.mean()
            ),
            "feature_consistency_mean": float(
                feature_consistency.mean()
            ),
            "neighbour_consistency_mean": float(
                neighbour_consistency.mean()
            ),
            "baseline_correct_fraction_after_assignment": float(
                correct.mean()
            ),
        },
        "anchor_reports": anchor_reports,
        "baseline_provenance": provenance,
        "samples": limit,
        "feature_extraction_seconds": extracted["seconds"],
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "gate_rule": {
            "all_delta_minimum": 0.0,
            "new_delta_minimum": 0.005,
            "hmean_delta_minimum": 0.005,
            "hmean_lower_95_above_zero": True,
            "candidate_hmean_not_below_controls": True,
            "stability_error_auc_minimum": 0.60,
            "correct_incorrect_stability_gap_minimum": 0.05,
        },
        "screening_note": (
            "Three predefined within-cluster anchor fractions are screened. "
            "A passing setting would require a separately locked confirmation."
        ),
        "verdict": verdict,
    }

    _atomic_json(result_path, result)

    print()
    print("===== INTERVENTION-STABILITY GATE =====")
    print(
        "method                    "
        "all       old       new      hmean"
    )

    order = [
        "baseline",
        "majority_vote",
        "mean_centroid",
        "stability_anchor_50",
        "random_anchor_50",
        "stability_anchor_70",
        "random_anchor_70",
        "stability_anchor_90",
        "random_anchor_90",
    ]

    for name in order:
        value = metric_values[name]
        print(
            f"{name:<25}"
            f"{value['all']:>9.6f}"
            f"{value['old']:>10.6f}"
            f"{value['new']:>10.6f}"
            f"{value['hmean']:>11.6f}"
        )

    print()
    print(f"stability error-prediction AUC: {stability_auc:.6f}")
    print(f"correct-minus-incorrect gap:   {stability_gap:+.6f}")

    for suffix, gate in gates.items():
        delta = gate["delta"]
        interval = gate["hmean_paired_bootstrap"]
        print(
            f"anchor {suffix}%: "
            f"All={delta['all']:+.6f}, "
            f"New={delta['new']:+.6f}, "
            f"H={delta['hmean']:+.6f}, "
            f"H95=[{interval['lower_95']:+.6f}, "
            f"{interval['upper_95']:+.6f}], "
            f"gate={'PASSED' if gate['passed'] else 'FAILED'}"
        )

    print()
    print(
        "INTERVENTION-STABILITY VERDICT:",
        verdict.upper(),
    )
    print("result:", result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/intervention_stability/cub/seed_0",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
