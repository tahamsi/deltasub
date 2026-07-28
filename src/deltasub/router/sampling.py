from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Sequence

from .schema import RouterExample
from ..utils.hashing import stable_hash


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    batches: tuple[tuple[RouterExample, ...], ...]
    coverage_count: int
    informative_count: int
    sha256: str


def _order(key: str, seed: int, epoch: int, stream: str) -> str:
    return hashlib.sha256(f"{seed}|{epoch}|{stream}|{key}".encode()).hexdigest()


def target_sign(value: float, epsilon: float = 1e-8) -> str:
    return "positive" if value > epsilon else "negative" if value < -epsilon else "near_zero"


def plan_epoch(
    examples: Sequence[RouterExample], *, batch_size: int, informative_fraction: float,
    seed: int, epoch: int, allow_duplicates: bool = False,
) -> EpochPlan:
    if batch_size <= 0 or not 0 <= informative_fraction <= 1:
        raise ValueError("invalid sampling configuration")
    if any(x.split_assignment != "train" for x in examples):
        raise ValueError("training sampler received validation/test records")
    unique = {x.gain_record_key: x for x in examples}
    if len(unique) != len(examples):
        raise ValueError("duplicate training record key")
    coverage = sorted(examples, key=lambda x: (
        x.candidate_parent_index, _order(x.gain_record_key, seed, epoch, "coverage")
    ))
    informative = sorted(examples, key=lambda x: (
        -abs(x.target_gain), target_sign(x.target_gain),
        _order(x.gain_record_key, seed, epoch, "informative")
    ))
    n_info = round(batch_size * informative_fraction)
    n_coverage = batch_size - n_info
    batches, coverage_used, informative_used = [], 0, 0
    cursor_a = cursor_b = 0
    total_batches = math.ceil(len(examples) / batch_size)
    for _ in range(total_batches):
        selected: list[RouterExample] = []
        while len([x for x in selected if x.sampling_stream == "coverage"]) < n_coverage and cursor_a < len(coverage):
            item = coverage[cursor_a]; cursor_a += 1
            selected.append(replace(item, sampling_stream="coverage"))
            coverage_used += 1
        while len(selected) < batch_size and cursor_b < len(informative):
            item = informative[cursor_b]; cursor_b += 1
            if allow_duplicates or all(x.gain_record_key != item.gain_record_key for x in selected):
                selected.append(replace(item, sampling_stream="informative"))
                informative_used += 1
        while len(selected) < batch_size and cursor_a < len(coverage):
            item = coverage[cursor_a]; cursor_a += 1
            if allow_duplicates or all(x.gain_record_key != item.gain_record_key for x in selected):
                selected.append(replace(item, sampling_stream="coverage"))
                coverage_used += 1
        if selected:
            batches.append(tuple(selected))
    keys = [[[x.gain_record_key, x.sampling_stream] for x in batch] for batch in batches]
    return EpochPlan(epoch, tuple(batches), coverage_used, informative_used, stable_hash(keys))


def sampling_summary(plan: EpochPlan) -> dict:
    examples = [x for batch in plan.batches for x in batch]
    return {
        "coverage_stream_examples": plan.coverage_count,
        "informative_stream_examples": plan.informative_count,
        "parent_index_coverage": sorted({x.candidate_parent_index for x in examples}),
        "target_sign_composition": dict(Counter(target_sign(x.target_gain) for x in examples)),
        "effective_sample_weight": sum(x.sample_weight for x in examples),
        "inverse_propensity_weighting": False,
        "training_plan_hash": plan.sha256,
    }
