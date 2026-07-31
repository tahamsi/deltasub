from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from ..utils.hashing import stable_hash

SELECTION_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SelectionPlan:
    schema_version: int
    sample_id: str
    view_id: str
    batch_context_hash: str
    router_checkpoint_hash: str
    router_score_hash: str
    controller_configuration_hash: str
    controller_state_hash: str
    budget_unit: str
    target_budget: float
    minimum_k: int
    maximum_k: int
    realized_k: int
    selected_parent_indices: tuple[int, ...]
    selected_mask_hash: str
    canonical_rank_order: tuple[int, ...]
    prefix_token_count: int
    parent_token_count: int
    added_detail_token_count: int
    effective_total_token_count: int
    bucket_id: str | None
    model_checkpoint_hash: str
    dinov2_source_revision: str
    child_projector_provenance: str
    deterministic_plan_hash: str = ""

    def payload(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "deterministic_plan_hash"}

    def with_hash(self) -> "SelectionPlan":
        return replace(self, deterministic_plan_hash=stable_hash(self.payload()))

    def validate(self, scores: list[float] | None = None) -> None:
        if self.schema_version != SELECTION_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported selection-plan schema")
        indices = self.selected_parent_indices
        if len(indices) != len(set(indices)) or any(not 0 <= x < 256 for x in indices):
            raise ValueError("duplicate or out-of-range selected parent")
        if self.parent_token_count != 256 or self.realized_k != len(indices):
            raise ValueError("selection count mismatch")
        if self.added_detail_token_count != 3 * self.realized_k:
            raise ValueError("detail-token count mismatch")
        if self.effective_total_token_count != self.prefix_token_count + 256 + 3 * self.realized_k:
            raise ValueError("effective token count mismatch")
        if not self.minimum_k <= self.realized_k <= self.maximum_k <= 256:
            raise ValueError("impossible K bounds")
        if len(self.canonical_rank_order) != 256 or set(self.canonical_rank_order) != set(range(256)):
            raise ValueError("canonical rank order must be a permutation of parents")
        if tuple(self.canonical_rank_order[:self.realized_k]) != indices:
            raise ValueError("selected indices do not follow canonical rank order")
        if scores is not None:
            expected = tuple(sorted(range(256), key=lambda j: (-float(scores[j]), j)))
            if expected != self.canonical_rank_order:
                raise ValueError("non-canonical score/tie order")
        provenance = (
            self.sample_id, self.view_id, self.router_checkpoint_hash,
            self.router_score_hash, self.controller_configuration_hash,
            self.controller_state_hash, self.selected_mask_hash,
            self.model_checkpoint_hash, self.dinov2_source_revision,
            self.child_projector_provenance,
        )
        if any(not value for value in provenance):
            raise ValueError("missing selection provenance")
        if self.deterministic_plan_hash != stable_hash(self.payload()):
            raise ValueError("selection-plan hash mismatch")
