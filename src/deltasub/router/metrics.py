from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F


def _defined(value: float | None):
    return {"defined": value is not None, "value": value}


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2
        index = end
    return ranks


def _correlation(a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_metrics(
    scores: Sequence[float], targets: Sequence[float], groups: Sequence[str],
    *, labelled: Sequence[bool] | None = None, known_or_novel: Sequence[str | None] | None = None,
    parents: Sequence[int] | None = None, sign_epsilon: float = 0.,
    huber_delta: float = 1., diagnostic_ks: Sequence[int] = (1, 4, 16),
    quantiles: int = 4, invalid_anchor_count: int = 0,
) -> dict:
    score, target = np.asarray(scores, dtype=float), np.asarray(targets, dtype=float)
    if len(score) != len(target) or not np.isfinite(score).all() or not np.isfinite(target).all():
        raise ValueError("metrics require equal finite score/target vectors")
    error = score - target
    pearson, spearman = _correlation(score, target), _correlation(_rank(score), _rank(target))
    concordant = total = ties = 0
    by_group = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    recalls, dcgs = {k: [] for k in diagnostic_ks}, []
    for indices in by_group.values():
        for offset, a in enumerate(indices):
            for b in indices[offset + 1:]:
                truth, prediction = np.sign(target[a] - target[b]), np.sign(score[a] - score[b])
                if truth:
                    total += 1
                    concordant += truth == prediction
                    ties += prediction == 0
        for k in diagnostic_ks:
            count = min(k, len(indices))
            if count:
                truth_top = set(sorted(indices, key=lambda i: (-target[i], i))[:count])
                score_top = set(sorted(indices, key=lambda i: (-score[i], i))[:count])
                recalls[k].append(len(truth_top & score_top) / count)
        ordered = sorted(indices, key=lambda i: (-score[i], i))
        relevance = np.asarray([target[i] for i in indices])
        relevance = relevance - relevance.min()
        dcg = sum(relevance[indices.index(i)] / math.log2(rank + 2) for rank, i in enumerate(ordered))
        ideal = sum(x / math.log2(rank + 2) for rank, x in enumerate(sorted(relevance, reverse=True)))
        if ideal > 0:
            dcgs.append(dcg / ideal)
    positive_target, positive_score = target > sign_epsilon, score > sign_epsilon
    tp, fp, fn = ((positive_target & positive_score).sum(),
                  (~positive_target & positive_score).sum(),
                  (positive_target & ~positive_score).sum())
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    bins = []
    if len(score):
        for indices in np.array_split(np.argsort(score, kind="mergesort"), min(quantiles, len(score))):
            bins.append({"count": len(indices), "mean_score": float(score[indices].mean()),
                         "mean_target_gain": float(target[indices].mean())})
    result = {
        "count": len(score), "mae": float(np.abs(error).mean()) if len(score) else 0.,
        "rmse": float(np.sqrt(np.mean(error ** 2))) if len(score) else 0.,
        "huber": float(F.huber_loss(torch.tensor(score), torch.tensor(target), delta=huber_delta)) if len(score) else 0.,
        "pearson": _defined(pearson), "spearman": _defined(spearman),
        "pairwise_ranking_accuracy": _defined(concordant / total if total else None),
        "kendall_concordance": _defined((concordant - (total - concordant - ties)) / total if total else None),
        "top_k_target_gain_recall": {str(k): float(np.mean(v)) if v else None for k, v in recalls.items()},
        "ndcg": _defined(float(np.mean(dcgs)) if dcgs else None),
        "positive_gain_precision": _defined(precision), "positive_gain_recall": _defined(recall),
        "positive_gain_f1": _defined(f1), "quantile_calibration": bins,
        "invalid_anchor_count": invalid_anchor_count,
        "top_k_is_diagnostic_only": True,
    }
    def subgroup(mask):
        indices = np.flatnonzero(mask)
        return {
            "count": int(len(indices)),
            "mae": float(np.abs(error[indices]).mean()) if len(indices) else 0.,
            "rmse": float(np.sqrt(np.mean(error[indices] ** 2))) if len(indices) else 0.,
        }
    result["metrics_by_target_sign"] = {
        "positive": subgroup(target > sign_epsilon),
        "negative": subgroup(target < -sign_epsilon),
        "near_zero": subgroup(np.abs(target) <= sign_epsilon),
    }
    if labelled is not None:
        labelled_array = np.asarray(labelled, dtype=bool)
        result["strata_counts"] = {
            "labelled": sum(labelled), "unlabelled": len(labelled) - sum(labelled)
        }
        result["metrics_by_label_status"] = {
            "labelled": subgroup(labelled_array), "unlabelled": subgroup(~labelled_array)
        }
    if known_or_novel is not None:
        status = np.asarray(known_or_novel, dtype=object)
        result.setdefault("strata_counts", {}).update(
            known=sum(x == "known" for x in known_or_novel),
            novel=sum(x == "novel" for x in known_or_novel),
        )
        result["metrics_by_known_or_novel"] = {
            "known": subgroup(status == "known"), "novel": subgroup(status == "novel")
        }
    if parents is not None:
        region = np.asarray([f"{value // 64},{(value % 16) // 4}" for value in parents])
        result["metrics_by_parent_region"] = {
            key: subgroup(region == key) for key in sorted(set(region))
        }
    return result
