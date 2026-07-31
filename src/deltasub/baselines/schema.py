from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import math
from ..utils.hashing import stable_hash

SCHEMA_VERSION = "m8.comparison.v1"

def optional(value: Any, reason: str | None = None) -> dict[str, Any]:
    if isinstance(value, float) and not math.isfinite(value): raise ValueError("undefined values must not be NaN/Inf")
    if value is None and not reason: raise ValueError("null values require a reason")
    return {"value": value, "reason": reason}

@dataclass(frozen=True)
class ComparisonRecord:
    method_id: str; display_name: str; adapter_status: str; implementation_label: str
    source_revision: str | None; license_status: str; model_checkpoint_hash: str | None
    adapter_checkpoint_hash: str | None; dataset_hash: str | None; split_hash: str | None
    seed: int; device: str; precision: str; effective_tokens: list[int]; padded_tokens: list[int]
    approximate_cost: list[int]; measured_latency: dict[str, Any]; peak_memory: dict[str, Any]
    feature_hash: str; logits_hash: str | None; plan_hash: str; frozen_state_hashes: dict[str,str]
    fixture: bool; reportable: bool; limitations: tuple[str,...]; schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.fixture and self.reportable: raise ValueError("synthetic fixture results cannot be reportable")
        if not self.limitations: raise ValueError("explicit limitations are required")
    @property
    def record_hash(self): return stable_hash(asdict(self))
