from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
import math

from .budget import BudgetSpec, token_counts


@dataclass(frozen=True)
class SampleAccounting:
    selected_parents: int
    added_detail_tokens: int
    effective_spatial_tokens: int
    effective_total_tokens: int
    padded_total_tokens: int
    padding_tokens: int
    padding_fraction: float
    bucket_id: str
    router_multiply_add_estimate: int
    transformer_token_count: int
    approximate_attention_token_pairs: int | None


def account_samples(k_values, prefix_tokens: int, padded_lengths, bucket_ids,
                    router_madds: int = 0, include_attention_estimate: bool = False):
    records = []
    for k, padded, bucket in zip(k_values, padded_lengths, bucket_ids, strict=True):
        counts = token_counts(int(k), prefix_tokens)
        effective = counts["total_tokens"]
        records.append(SampleAccounting(
            int(k), counts["added_detail_tokens"], counts["spatial_tokens"], effective,
            int(padded), int(padded) - effective, (int(padded) - effective) / int(padded),
            bucket, int(router_madds), effective,
            effective * effective if include_attention_estimate else None,
        ))
    return records


def summarize_accounting(records: list[SampleAccounting], budget: BudgetSpec) -> dict:
    if not records:
        raise ValueError("accounting requires at least one sample")
    ks = [x.selected_parents for x in records]
    effective = [x.effective_total_tokens for x in records]
    padded = [x.padded_total_tokens for x in records]
    realized = sum(budget.usage(k) for k in ks) / len(ks)
    violation = realized - budget.target
    return {
        "units": {"K": "selected_parents", "details": "added_detail_tokens",
                  "effective": "total_tokens", "padded": "padded_total_tokens",
                  "router_overhead": "multiply_adds"},
        "mean_k": sum(ks) / len(ks), "minimum_k": min(ks), "maximum_k": max(ks),
        "k_histogram": {str(k): v for k, v in sorted(Counter(ks).items())},
        "mean_effective_tokens": sum(effective) / len(effective),
        "mean_padded_tokens": sum(padded) / len(padded),
        "padding_overhead": (sum(padded) - sum(effective)) / sum(padded),
        "target_usage": budget.target, "realized_usage": realized,
        "token_budget_violation": violation,
        "percentage_budget_violation": (100 * violation / budget.target
                                         if budget.target else (0.0 if violation == 0 else math.inf)),
        "bucket_counts": dict(sorted(Counter(x.bucket_id for x in records).items())),
        "router_overhead_estimate": sum(x.router_multiply_add_estimate for x in records),
        "samples": [asdict(x) for x in records],
        "token_counts_are_not_flops": True, "latency_claimed": False,
    }
