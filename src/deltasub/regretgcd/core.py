"""Leak-resistant expert arbitration utilities for RegretGCD.

The module is deliberately NumPy/scikit-learn based.  The expensive vision model is
used only to produce cached features and logits; every router comparison thereafter is
cheap, deterministic, and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


EPS = 1.0e-12


@dataclass(frozen=True)
class PrototypeState:
    """Transductive prototypes indexed by parametric pseudo-labels."""

    labels: np.ndarray
    inverse: np.ndarray
    counts: np.ndarray
    full: np.ndarray
    leave_one_out: np.ndarray


@dataclass(frozen=True)
class RouterFit:
    """A small standardized logistic regret router."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    threshold: float
    oof_auc: float | None
    oof_accuracy: float
    oof_switch_rate: float
    unique_winners: int
    positive_winners: int
    negative_winners: int
    fold_kind: str
    fold_count: int

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        standardized = (matrix - self.mean) / self.scale
        logit = standardized @ self.coefficient + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0)))


@dataclass(frozen=True)
class RouterFeatures:
    """Router matrix and common-space expert probabilities."""

    matrix: np.ndarray
    names: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]
    parametric_prediction: np.ndarray
    prototype_prediction: np.ndarray
    parametric_probability: np.ndarray
    prototype_probability: np.ndarray


def normalize_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return (array / np.maximum(norm, EPS)).astype(np.float64, copy=False)


def softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exponential = np.exp(np.clip(shifted, -80.0, 80.0))
    return exponential / np.maximum(exponential.sum(axis=axis, keepdims=True), EPS)


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    classes = probability.shape[-1]
    denominator = max(math.log(max(classes, 2)), EPS)
    return -np.sum(
        probability * np.log(np.maximum(probability, EPS)),
        axis=-1,
    ) / denominator


def view_jsd(view_probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(view_probability, dtype=np.float64)
    mean = probability.mean(axis=1, keepdims=True)
    return np.mean(
        np.sum(
            probability
            * (
                np.log(np.maximum(probability, EPS))
                - np.log(np.maximum(mean, EPS))
            ),
            axis=-1,
        ),
        axis=1,
    )


def symmetric_jsd(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    midpoint = 0.5 * (first + second)
    first_kl = np.sum(
        first
        * (
            np.log(np.maximum(first, EPS))
            - np.log(np.maximum(midpoint, EPS))
        ),
        axis=-1,
    )
    second_kl = np.sum(
        second
        * (
            np.log(np.maximum(second, EPS))
            - np.log(np.maximum(midpoint, EPS))
        ),
        axis=-1,
    )
    return 0.5 * (first_kl + second_kl)


def top_two(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-1] < 2:
        first = array[..., 0]
        return first, np.zeros_like(first)
    partition = np.partition(array, -2, axis=-1)
    return partition[..., -1], partition[..., -2]


def view_agreement(view_prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(view_prediction, dtype=np.int64)
    if prediction.ndim != 2:
        raise ValueError("view predictions must have shape [N,V]")
    result = np.empty(prediction.shape[0], dtype=np.float64)
    for index, row in enumerate(prediction):
        _, counts = np.unique(row, return_counts=True)
        result[index] = float(counts.max()) / float(row.size)
    return result


def build_prototypes(mean_features: np.ndarray, pseudo_labels: np.ndarray) -> PrototypeState:
    features = normalize_rows(mean_features)
    labels = np.asarray(pseudo_labels, dtype=np.int64)
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("prototype inputs have incompatible shapes")

    unique, inverse = np.unique(labels, return_inverse=True)
    sums = np.zeros((len(unique), features.shape[1]), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.int64)
    np.add.at(sums, inverse, features)
    np.add.at(counts, inverse, 1)
    full = normalize_rows(sums / np.maximum(counts[:, None], 1))

    leave_one_out = np.empty_like(features)
    for index, cluster in enumerate(inverse):
        if counts[cluster] > 1:
            leave_one_out[index] = normalize_rows(
                ((sums[cluster] - features[index]) / (counts[cluster] - 1))[None]
            )[0]
        else:
            leave_one_out[index] = full[cluster]

    return PrototypeState(
        labels=unique.astype(np.int64),
        inverse=inverse.astype(np.int64),
        counts=counts,
        full=full,
        leave_one_out=leave_one_out,
    )


def prototype_view_scores(
    view_features: np.ndarray,
    state: PrototypeState,
    *,
    own_cluster: np.ndarray | None = None,
    temperature: float = 0.10,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("prototype temperature must be positive")
    features = normalize_rows(view_features)
    scores = np.einsum("nvd,kd->nvk", features, state.full)
    if own_cluster is not None:
        own = np.asarray(own_cluster, dtype=np.int64)
        if own.shape != (features.shape[0],):
            raise ValueError("own_cluster must have shape [N]")
        if len(state.leave_one_out) != len(features):
            raise ValueError("leave-one-out state does not match sample count")
        replacement = np.einsum("nvd,nd->nv", features, state.leave_one_out)
        scores[
            np.arange(features.shape[0])[:, None],
            np.arange(features.shape[1])[None, :],
            own[:, None],
        ] = replacement
    return scores / float(temperature)


def known_centroid_scores(
    train_mean_features: np.ndarray,
    train_targets: np.ndarray,
    query_mean_features: np.ndarray,
    *,
    leave_one_out: bool,
) -> np.ndarray:
    train = normalize_rows(train_mean_features)
    query = normalize_rows(query_mean_features)
    targets = np.asarray(train_targets, dtype=np.int64)
    labels, inverse = np.unique(targets, return_inverse=True)
    sums = np.zeros((len(labels), train.shape[1]), dtype=np.float64)
    counts = np.zeros(len(labels), dtype=np.int64)
    np.add.at(sums, inverse, train)
    np.add.at(counts, inverse, 1)
    centroids = normalize_rows(sums / counts[:, None])
    scores = query @ centroids.T

    if leave_one_out:
        if len(query) != len(train):
            raise ValueError("leave-one-out knownness requires the training samples")
        replacement = np.empty(len(train), dtype=np.float64)
        for index, cluster in enumerate(inverse):
            if counts[cluster] > 1:
                centroid = normalize_rows(
                    ((sums[cluster] - train[index]) / (counts[cluster] - 1))[None]
                )[0]
            else:
                centroid = centroids[cluster]
            replacement[index] = float(train[index] @ centroid)
        scores[np.arange(len(train)), inverse] = replacement

    return scores


def _scatter_prototype_probability(
    probability: np.ndarray,
    labels: np.ndarray,
    class_count: int,
) -> np.ndarray:
    output = np.zeros((*probability.shape[:-1], class_count), dtype=np.float64)
    valid = (labels >= 0) & (labels < class_count)
    output[..., labels[valid]] = probability[..., valid]
    normalizer = output.sum(axis=-1, keepdims=True)
    return output / np.maximum(normalizer, EPS)


def build_router_features(
    *,
    parametric_logits: np.ndarray,
    view_features: np.ndarray,
    prototype_state: PrototypeState,
    known_scores: np.ndarray,
    parametric_hard: np.ndarray | None = None,
    test_leave_one_out: bool,
    prototype_temperature: float = 0.10,
) -> RouterFeatures:
    logits = np.asarray(parametric_logits, dtype=np.float64)
    features = normalize_rows(view_features)
    if logits.ndim != 3 or features.ndim != 3:
        raise ValueError("logits and features must have shape [N,V,*]")
    if logits.shape[:2] != features.shape[:2]:
        raise ValueError("logit and feature view dimensions differ")
    sample_count, view_count, class_count = logits.shape

    p_view = softmax(logits, axis=-1)
    p_mean = p_view.mean(axis=1)
    p_mean /= np.maximum(p_mean.sum(axis=1, keepdims=True), EPS)
    p_view_prediction = p_view.argmax(axis=-1)
    p_prediction = (
        np.asarray(parametric_hard, dtype=np.int64)
        if parametric_hard is not None
        else p_mean.argmax(axis=-1).astype(np.int64)
    )
    if p_prediction.shape != (sample_count,):
        raise ValueError("parametric_hard must have shape [N]")

    own = prototype_state.inverse if test_leave_one_out else None
    t_scores = prototype_view_scores(
        features,
        prototype_state,
        own_cluster=own,
        temperature=prototype_temperature,
    )
    t_view = softmax(t_scores, axis=-1)
    t_mean = t_view.mean(axis=1)
    t_mean /= np.maximum(t_mean.sum(axis=1, keepdims=True), EPS)
    mean_features = normalize_rows(features.mean(axis=1))
    hard_scores = mean_features @ prototype_state.full.T
    if test_leave_one_out:
        hard_scores[np.arange(sample_count), prototype_state.inverse] = np.sum(
            mean_features * prototype_state.leave_one_out,
            axis=1,
        )
    t_cluster = hard_scores.argmax(axis=-1)
    t_prediction = prototype_state.labels[t_cluster]
    t_view_prediction = prototype_state.labels[t_view.argmax(axis=-1)]

    t_common_view = _scatter_prototype_probability(
        t_view,
        prototype_state.labels,
        class_count,
    )
    t_common = t_common_view.mean(axis=1)
    t_common /= np.maximum(t_common.sum(axis=1, keepdims=True), EPS)

    p_top1, p_top2 = top_two(p_mean)
    t_top1, t_top2 = top_two(t_mean)
    density_top1, density_top2 = top_two(hard_scores)
    known_top1, known_top2 = top_two(np.asarray(known_scores, dtype=np.float64))

    p_support_for_t = np.zeros(sample_count, dtype=np.float64)
    t_support_for_p = np.zeros(sample_count, dtype=np.float64)
    valid_t = (t_prediction >= 0) & (t_prediction < class_count)
    valid_p = (p_prediction >= 0) & (p_prediction < class_count)
    p_support_for_t[valid_t] = p_mean[np.arange(sample_count)[valid_t], t_prediction[valid_t]]
    t_support_for_p[valid_p] = t_common[np.arange(sample_count)[valid_p], p_prediction[valid_p]]

    names = (
        "p_confidence",
        "p_margin",
        "p_entropy",
        "p_view_agreement",
        "p_view_jsd",
        "t_confidence",
        "t_margin",
        "t_entropy",
        "t_view_agreement",
        "t_view_jsd",
        "density_top1",
        "density_top2",
        "density_gap",
        "density_log_cluster_size",
        "experts_agree",
        "cross_symmetric_jsd",
        "p_support_for_t",
        "t_support_for_p",
        "confidence_delta_t_minus_p",
        "entropy_delta_p_minus_t",
        "known_similarity_top1",
        "known_similarity_gap",
    )

    cluster_size = prototype_state.counts[t_cluster]
    columns = (
        p_top1,
        p_top1 - p_top2,
        normalized_entropy(p_mean),
        view_agreement(p_view_prediction),
        view_jsd(p_view),
        t_top1,
        t_top1 - t_top2,
        normalized_entropy(t_mean),
        view_agreement(t_view_prediction),
        view_jsd(t_view),
        density_top1,
        density_top2,
        density_top1 - density_top2,
        np.log1p(cluster_size.astype(np.float64)),
        (p_prediction == t_prediction).astype(np.float64),
        symmetric_jsd(p_mean, t_common),
        p_support_for_t,
        t_support_for_p,
        t_top1 - p_top1,
        normalized_entropy(p_mean) - normalized_entropy(t_mean),
        known_top1,
        known_top1 - known_top2,
    )
    matrix = np.column_stack(columns).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("router features contain non-finite values")

    groups = {
        "confidence": (
            "p_confidence",
            "p_margin",
            "p_entropy",
            "t_confidence",
            "t_margin",
            "t_entropy",
        ),
        "stability": (
            "p_view_agreement",
            "p_view_jsd",
            "t_view_agreement",
            "t_view_jsd",
        ),
        "density": (
            "density_top1",
            "density_top2",
            "density_gap",
            "density_log_cluster_size",
        ),
        "cross": (
            "experts_agree",
            "cross_symmetric_jsd",
            "p_support_for_t",
            "t_support_for_p",
            "confidence_delta_t_minus_p",
            "entropy_delta_p_minus_t",
        ),
        "knownness": (
            "known_similarity_top1",
            "known_similarity_gap",
        ),
    }

    return RouterFeatures(
        matrix=matrix,
        names=names,
        groups=groups,
        parametric_prediction=p_prediction.astype(np.int64),
        prototype_prediction=t_prediction.astype(np.int64),
        parametric_probability=p_mean,
        prototype_probability=t_common,
    )


def hungarian_mapping(prediction: np.ndarray, target: np.ndarray) -> dict[int, int]:
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("prediction and target must be one-dimensional and aligned")
    pred_labels, pred_inverse = np.unique(prediction, return_inverse=True)
    target_labels, target_inverse = np.unique(target, return_inverse=True)
    contingency = np.zeros((len(pred_labels), len(target_labels)), dtype=np.int64)
    np.add.at(contingency, (pred_inverse, target_inverse), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    return {
        int(pred_labels[row]): int(target_labels[column])
        for row, column in zip(rows, columns)
    }


def apply_mapping(prediction: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    return np.asarray(
        [mapping.get(int(value), -1) for value in np.asarray(prediction)],
        dtype=np.int64,
    )


def regret_labels(
    parametric_prediction: np.ndarray,
    prototype_prediction: np.ndarray,
    target: np.ndarray,
    mapping: Mapping[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.int64)
    p_correct = apply_mapping(parametric_prediction, mapping) == target
    t_correct = apply_mapping(prototype_prediction, mapping) == target
    unique = p_correct ^ t_correct
    label = t_correct.astype(np.int64)
    return label, unique, p_correct, t_correct


def feature_indices(
    names: Sequence[str],
    *,
    include_groups: Iterable[str] | None = None,
    exclude_groups: Iterable[str] = (),
    groups: Mapping[str, Sequence[str]],
) -> np.ndarray:
    name_to_index = {name: index for index, name in enumerate(names)}
    if include_groups is None:
        selected = set(names)
    else:
        selected: set[str] = set()
        for group in include_groups:
            selected.update(groups[group])
    for group in exclude_groups:
        selected.difference_update(groups[group])
    indices = [name_to_index[name] for name in names if name in selected]
    if not indices:
        raise ValueError("feature ablation selected no features")
    return np.asarray(indices, dtype=np.int64)


def _class_fold_assignment(class_ids: np.ndarray, folds: int, seed: int) -> np.ndarray:
    classes = np.unique(class_ids)
    generator = np.random.default_rng(seed)
    shuffled = classes.copy()
    generator.shuffle(shuffled)
    assignment = {int(value): index % folds for index, value in enumerate(shuffled)}
    return np.asarray([assignment[int(value)] for value in class_ids], dtype=np.int64)


def _instance_fold_assignment(sample_ids: Sequence[str], folds: int, seed: int) -> np.ndarray:
    values = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(f"regretgcd:{seed}:{sample_id}".encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "little") % folds)
    return np.asarray(values, dtype=np.int64)


def _fit_logistic(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[StandardScaler, LogisticRegression]:
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError("regret-router training requires both unique-winner classes")
    scaler = StandardScaler().fit(features)
    transformed = scaler.transform(features)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=4000,
        solver="lbfgs",
        random_state=seed,
    ).fit(transformed, labels)
    return scaler, model


def route_predictions(
    parametric_prediction: np.ndarray,
    prototype_prediction: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    parametric = np.asarray(parametric_prediction, dtype=np.int64)
    prototype = np.asarray(prototype_prediction, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if not (parametric.shape == prototype.shape == probability.shape):
        raise ValueError("routing arrays must have identical shape")
    switch = (parametric != prototype) & (probability >= float(threshold))
    result = parametric.copy()
    result[switch] = prototype[switch]
    return result


def _calibrate_threshold(
    probability: np.ndarray,
    parametric_prediction: np.ndarray,
    prototype_prediction: np.ndarray,
    target: np.ndarray,
    mapping: Mapping[int, int],
) -> tuple[float, float, float]:
    probability = np.asarray(probability, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0], probability, [1.0 + 1.0e-9])))
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        prediction = route_predictions(
            parametric_prediction,
            prototype_prediction,
            probability,
            float(threshold),
        )
        accuracy = float(np.mean(apply_mapping(prediction, mapping) == target))
        switch_rate = float(
            np.mean(
                (parametric_prediction != prototype_prediction)
                & (probability >= threshold)
            )
        )
        key = (accuracy, -switch_rate, float(threshold))
        if best is None or key > best:
            best = key
    assert best is not None
    return float(best[2]), float(best[0]), float(-best[1])


def fit_regret_router(
    *,
    features: np.ndarray,
    feature_names: Sequence[str],
    target: np.ndarray,
    class_ids: np.ndarray,
    sample_ids: Sequence[str],
    parametric_prediction: np.ndarray,
    prototype_prediction: np.ndarray,
    mapping: Mapping[int, int],
    feature_index: np.ndarray,
    folds: int,
    seed: int,
    fold_kind: str,
) -> tuple[RouterFit, np.ndarray]:
    matrix = np.asarray(features, dtype=np.float64)[:, feature_index]
    target = np.asarray(target, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    labels, unique, _, _ = regret_labels(
        parametric_prediction,
        prototype_prediction,
        target,
        mapping,
    )
    if int(unique.sum()) < 20:
        raise RuntimeError("too few unique-winner examples for a defensible router")
    if len(np.unique(labels[unique])) != 2:
        raise RuntimeError("unique-winner set contains only one expert outcome")

    if fold_kind == "class":
        fold = _class_fold_assignment(class_ids, folds, seed)
    elif fold_kind == "instance":
        fold = _instance_fold_assignment(sample_ids, folds, seed)
    else:
        raise ValueError("fold_kind must be class or instance")

    oof = np.full(len(matrix), np.nan, dtype=np.float64)
    for fold_index in range(folds):
        validation = fold == fold_index
        training = (~validation) & unique
        if not validation.any():
            continue
        if len(np.unique(labels[training])) != 2:
            prior = float(labels[unique].mean())
            oof[validation] = prior
            continue
        scaler, model = _fit_logistic(matrix[training], labels[training], seed + fold_index)
        oof[validation] = model.predict_proba(scaler.transform(matrix[validation]))[:, 1]

    if np.isnan(oof).any():
        raise RuntimeError("out-of-fold router probabilities are incomplete")

    threshold, oof_accuracy, oof_switch_rate = _calibrate_threshold(
        oof,
        parametric_prediction,
        prototype_prediction,
        target,
        mapping,
    )
    auc: float | None
    if len(np.unique(labels[unique])) == 2:
        auc = float(roc_auc_score(labels[unique], oof[unique]))
    else:
        auc = None

    scaler, model = _fit_logistic(matrix[unique], labels[unique], seed + 1000)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale == 0] = 1.0
    fit = RouterFit(
        feature_names=tuple(feature_names[index] for index in feature_index),
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=scale,
        coefficient=np.asarray(model.coef_[0], dtype=np.float64),
        intercept=float(model.intercept_[0]),
        threshold=threshold,
        oof_auc=auc,
        oof_accuracy=oof_accuracy,
        oof_switch_rate=oof_switch_rate,
        unique_winners=int(unique.sum()),
        positive_winners=int(labels[unique].sum()),
        negative_winners=int(unique.sum() - labels[unique].sum()),
        fold_kind=fold_kind,
        fold_count=folds,
    )
    return fit, oof


def random_matched_switch(
    parametric_prediction: np.ndarray,
    prototype_prediction: np.ndarray,
    switch_count: int,
    sample_ids: Sequence[str],
    seed: int,
) -> np.ndarray:
    parametric = np.asarray(parametric_prediction, dtype=np.int64)
    prototype = np.asarray(prototype_prediction, dtype=np.int64)
    disagreement = np.flatnonzero(parametric != prototype)
    switch_count = int(max(0, min(switch_count, len(disagreement))))
    order = sorted(
        disagreement.tolist(),
        key=lambda index: hashlib.sha256(
            f"regretgcd-random:{seed}:{sample_ids[index]}".encode("utf-8")
        ).digest(),
    )
    selected = np.asarray(order[:switch_count], dtype=np.int64)
    result = parametric.copy()
    result[selected] = prototype[selected]
    return result


def hmean(old: float, new: float) -> float:
    return 2.0 * old * new / (old + new) if old + new > 0 else 0.0


def fixed_alignment_paired_bootstrap(
    *,
    target: np.ndarray,
    old: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=np.int64)
    old = np.asarray(old, dtype=bool)
    baseline = np.asarray(baseline_prediction, dtype=np.int64)
    candidate = np.asarray(candidate_prediction, dtype=np.int64)
    baseline_correct = apply_mapping(baseline, hungarian_mapping(baseline, target)) == target
    candidate_correct = apply_mapping(candidate, hungarian_mapping(candidate, target)) == target
    old_indices = np.flatnonzero(old)
    new_indices = np.flatnonzero(~old)
    if len(old_indices) == 0 or len(new_indices) == 0:
        raise ValueError("paired bootstrap requires non-empty Old and New partitions")

    generator = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled_old = generator.choice(old_indices, size=len(old_indices), replace=True)
        sampled_new = generator.choice(new_indices, size=len(new_indices), replace=True)
        base_old = float(baseline_correct[sampled_old].mean())
        base_new = float(baseline_correct[sampled_new].mean())
        candidate_old = float(candidate_correct[sampled_old].mean())
        candidate_new = float(candidate_correct[sampled_new].mean())
        values[draw] = hmean(candidate_old, candidate_new) - hmean(base_old, base_new)

    point = hmean(
        float(candidate_correct[old].mean()),
        float(candidate_correct[~old].mean()),
    ) - hmean(
        float(baseline_correct[old].mean()),
        float(baseline_correct[~old].mean()),
    )
    return {
        "draws": int(draws),
        "seed": int(seed),
        "point": float(point),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }
