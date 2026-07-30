from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

PARENT_COUNT = 256
DETAILS_PER_PARENT = 3


class BudgetUnit(str, Enum):
    SELECTED_PARENTS = "selected_parents"
    ADDED_DETAIL_TOKENS = "added_detail_tokens"
    SPATIAL_TOKENS = "spatial_tokens"
    TOTAL_TOKENS = "total_tokens"


def token_counts(k: int, prefix_tokens: int) -> dict[str, int]:
    if not isinstance(k, int) or not 0 <= k <= PARENT_COUNT:
        raise ValueError("K must be an integer in [0, 256]")
    if not isinstance(prefix_tokens, int) or prefix_tokens < 1:
        raise ValueError("prefix_tokens must be a positive integer")
    details = DETAILS_PER_PARENT * k
    return {
        BudgetUnit.SELECTED_PARENTS.value: k,
        BudgetUnit.ADDED_DETAIL_TOKENS.value: details,
        BudgetUnit.SPATIAL_TOKENS.value: PARENT_COUNT + details,
        BudgetUnit.TOTAL_TOKENS.value: prefix_tokens + PARENT_COUNT + details,
    }


def usage_to_k(value: float, unit: BudgetUnit | str, prefix_tokens: int) -> float:
    unit = BudgetUnit(unit)
    if not math.isfinite(float(value)):
        raise ValueError("budget must be finite")
    offsets = {
        BudgetUnit.SELECTED_PARENTS: (0, 1),
        BudgetUnit.ADDED_DETAIL_TOKENS: (0, 3),
        BudgetUnit.SPATIAL_TOKENS: (256, 3),
        BudgetUnit.TOTAL_TOKENS: (prefix_tokens + 256, 3),
    }
    offset, scale = offsets[unit]
    return (float(value) - offset) / scale


@dataclass(frozen=True)
class BudgetSpec:
    mode: str
    unit: BudgetUnit | str
    target: float
    min_k: int = 0
    max_k: int = 256
    prefix_tokens: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", BudgetUnit(self.unit))
        if self.mode not in {"fixed_k", "per_sample_k", "mean_detail_tokens", "mean_total_tokens"}:
            raise ValueError("unsupported budget mode")
        if not 0 <= self.min_k <= self.max_k <= PARENT_COUNT:
            raise ValueError("K bounds must satisfy 0 <= min_k <= max_k <= 256")
        k = usage_to_k(self.target, self.unit, self.prefix_tokens)
        if k < self.min_k or k > self.max_k:
            raise ValueError("budget is impossible under configured K bounds")
        if self.mode == "fixed_k" and (
            self.unit is not BudgetUnit.SELECTED_PARENTS or not k.is_integer()
        ):
            raise ValueError("fixed_k requires an integer selected_parents target")
        expected = {
            "mean_detail_tokens": BudgetUnit.ADDED_DETAIL_TOKENS,
            "mean_total_tokens": BudgetUnit.TOTAL_TOKENS,
        }
        if self.mode in expected and self.unit is not expected[self.mode]:
            raise ValueError("budget mode and explicit unit disagree")

    @property
    def target_selected_parents(self) -> float:
        return usage_to_k(self.target, self.unit, self.prefix_tokens)

    def usage(self, k: int) -> float:
        return float(token_counts(k, self.prefix_tokens)[self.unit.value])
