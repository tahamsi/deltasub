"""Strict M8 common baseline adapter contract."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from ..utils.hashing import stable_hash


class AdapterStatus(str, Enum):
    PRODUCTION_READY = "production_ready"
    FIXTURE_ONLY = "fixture_only"
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class StatusEvidence:
    source_available: bool
    source_revision_valid: bool
    license_compatible: bool
    checkpoint_available: bool
    configuration_valid: bool
    implementation_complete: bool
    fixture_validated: bool
    production_validated: bool
    reason: str

    @property
    def status(self) -> AdapterStatus:
        if not self.license_compatible:
            return AdapterStatus.BLOCKED
        if not self.implementation_complete or not self.fixture_validated:
            return AdapterStatus.UNAVAILABLE
        if all((self.source_available, self.source_revision_valid, self.checkpoint_available,
                self.configuration_valid, self.production_validated)):
            return AdapterStatus.PRODUCTION_READY
        return AdapterStatus.FIXTURE_ONLY


@dataclass
class AdapterInput:
    sample_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    parent_tokens: torch.Tensor
    prefix_tokens: torch.Tensor
    parent_positions: torch.Tensor
    prefix_positions: torch.Tensor
    valid_parent_mask: torch.Tensor
    token_budget: int
    device: str = "cpu"
    precision: str = "fp32"
    seed: int = 0
    images: torch.Tensor | None = None
    child_tokens: torch.Tensor | None = None
    checkpoint_provenance: dict[str, Any] = field(default_factory=dict)
    source_provenance: dict[str, Any] = field(default_factory=dict)
    mode: str = "fixture"

    def validate(self) -> None:
        b = len(self.sample_ids)
        if len(set(self.sample_ids)) != b or len(self.view_ids) != b:
            raise ValueError("sample IDs must be unique and view IDs aligned")
        if self.parent_tokens.ndim != 3 or self.parent_tokens.shape[:2] != (b, 256):
            raise ValueError("parent_tokens must be [B,256,D]")
        p, d = self.prefix_tokens.shape[1], self.parent_tokens.shape[2]
        if self.prefix_tokens.shape != (b, p, d):
            raise ValueError("prefix token shape mismatch")
        if self.parent_positions.shape != self.parent_tokens.shape or self.prefix_positions.shape != self.prefix_tokens.shape:
            raise ValueError("position shape mismatch")
        if self.valid_parent_mask.shape != (b, 256) or not bool(self.valid_parent_mask.all()):
            raise ValueError("M8 requires all 256 parent regions valid")
        if self.precision not in {"fp32", "bf16"} or self.mode not in {"fixture", "production"}:
            raise ValueError("unsupported precision or mode")
        for value in (self.parent_tokens, self.prefix_tokens, self.parent_positions, self.prefix_positions):
            if not torch.isfinite(value).all():
                raise ValueError("adapter tensors must be finite")


@dataclass
class TokenAccounting:
    original_parent_count: int
    retained_parent_count: int
    added_token_count: int
    removed_token_count: int
    effective_total_tokens: int
    padded_total_tokens: int
    token_semantics: str
    transformer_passes: int
    router_tokenizer_overhead_estimate: int | None
    approximate_attention_cost: int
    peak_memory_bytes: int | None = None


@dataclass
class AdapterOutput:
    cls_features: torch.Tensor
    register_features: torch.Tensor
    parent_spatial_features: torch.Tensor | None
    logits: torch.Tensor | None
    effective_lengths: torch.Tensor
    padded_lengths: torch.Tensor
    token_type_metadata: tuple[dict[str, Any], ...]
    plan_records: tuple[dict[str, Any], ...]
    accounting: tuple[TokenAccounting, ...]
    selection_plan_hash: str
    frozen_state_hashes: dict[str, str]
    diagnostic_metadata: dict[str, Any]
    canonical_result_hash: str = ""

    def finalize(self) -> "AdapterOutput":
        if not torch.isfinite(self.cls_features).all() or (self.logits is not None and not torch.isfinite(self.logits).all()):
            raise ValueError("non-finite adapter output")
        self.canonical_result_hash = stable_hash({
            "cls": self.cls_features.detach().float().cpu().tolist(),
            "logits": None if self.logits is None else self.logits.detach().float().cpu().tolist(),
            "lengths": self.effective_lengths.tolist(), "plan": self.selection_plan_hash,
        })
        return self


class BaselineAdapter(ABC):
    method_id: str
    display_name: str
    implementation_label: str
    source_revision: str | None
    license_status: str
    required_checkpoint_provenance: tuple[str, ...] = ()
    required_source_provenance: tuple[str, ...] = ()

    def __init__(self, evidence: StatusEvidence):
        self.evidence = evidence

    @property
    def status(self) -> AdapterStatus:
        return self.evidence.status

    @property
    def production_available(self) -> bool:
        return self.status is AdapterStatus.PRODUCTION_READY

    @property
    def fixture_available(self) -> bool:
        return self.evidence.fixture_validated and self.evidence.implementation_complete

    def require_available(self, mode: str) -> None:
        if mode == "fixture" and self.fixture_available:
            return
        if mode == "production" and self.production_available:
            return
        raise RuntimeError(f"{self.method_id} is {self.status.value} for {mode}: {self.evidence.reason}")

    @abstractmethod
    def prepare_tokens(self, inputs: AdapterInput) -> Any: ...

    @abstractmethod
    def execute(self, prepared: Any, *, execution_mode: str) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def run(self, inputs: AdapterInput, head: torch.nn.Module | None = None, *, execution_mode: str = "padded") -> AdapterOutput: ...

    def measured_latency_hook(self, callable_, *, synchronize: bool = False) -> float:
        import time
        if synchronize and torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter(); callable_()
        if synchronize and torch.cuda.is_available(): torch.cuda.synchronize()
        return time.perf_counter() - start

    def diagnostic_metadata(self) -> dict[str, Any]:
        return {"method_id": self.method_id, "status": self.status.value,
                "evidence": self.evidence.__dict__, "no_silent_fallback": True}
