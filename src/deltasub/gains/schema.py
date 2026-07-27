from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import pyarrow as pa

GAIN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GainRecord:
    schema_version: int
    dataset_name: str
    dataset_manifest_sha256: str
    split_report_sha256: str
    sample_id: str
    anchor_batch_position: int
    view_layout_identifier: str
    candidate_parent_index: int
    parent_row: int
    parent_column: int
    labelled: bool
    known_or_novel: str | None
    class_id: int | None
    base_per_anchor_loss: float
    counterfactual_per_anchor_loss: float
    gain: float
    base_scalar_batch_loss: float
    counterfactual_scalar_batch_loss: float
    scalar_batch_loss_change: float
    non_anchor_spillover_sum: float
    non_anchor_spillover_max_abs: float
    anchor_valid: bool
    base_effective_token_count: int
    counterfactual_effective_token_count: int
    base_padded_token_count: int
    counterfactual_padded_token_count: int
    model_checkpoint_sha256: str
    dinov2_source_revision: str
    child_projector_state_sha256: str
    selex_implementation_sha256: str
    selex_reference_sha256: str
    selex_equivalence_gate_sha256: str
    batch_context_sha256: str
    configuration_sha256: str
    precision: str
    resolved_device: str
    seed: int
    collection_timestamp: str
    git_commit: str

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (
            self.batch_context_sha256, self.sample_id,
            self.anchor_batch_position, self.candidate_parent_index,
        )

    def validate(self) -> None:
        if self.schema_version != GAIN_SCHEMA_VERSION:
            raise ValueError("unsupported gain schema version")
        if not 0 <= self.candidate_parent_index <= 255:
            raise ValueError("candidate parent index must be in [0, 255]")
        if (self.parent_row, self.parent_column) != divmod(self.candidate_parent_index, 16):
            raise ValueError("parent row/column do not match candidate index")
        numeric = (
            self.base_per_anchor_loss, self.counterfactual_per_anchor_loss, self.gain,
            self.base_scalar_batch_loss, self.counterfactual_scalar_batch_loss,
            self.scalar_batch_loss_change, self.non_anchor_spillover_sum,
            self.non_anchor_spillover_max_abs,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("gain records require finite numeric values")
        if self.anchor_valid and self.gain != self.base_per_anchor_loss - self.counterfactual_per_anchor_loss:
            raise ValueError("gain sign/value invariant failed")
        if self.counterfactual_effective_token_count != self.base_effective_token_count + 3:
            raise ValueError("counterfactual must add exactly three effective tokens")
        for name in (
            "dataset_manifest_sha256", "split_report_sha256", "model_checkpoint_sha256",
            "child_projector_state_sha256", "selex_implementation_sha256",
            "selex_reference_sha256", "selex_equivalence_gate_sha256",
            "batch_context_sha256", "configuration_sha256",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be SHA256")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


_INTS = {
    "schema_version", "anchor_batch_position", "candidate_parent_index", "parent_row",
    "parent_column", "class_id", "base_effective_token_count",
    "counterfactual_effective_token_count", "base_padded_token_count",
    "counterfactual_padded_token_count", "seed",
}
_FLOATS = {
    "base_per_anchor_loss", "counterfactual_per_anchor_loss", "gain",
    "base_scalar_batch_loss", "counterfactual_scalar_batch_loss",
    "scalar_batch_loss_change", "non_anchor_spillover_sum",
    "non_anchor_spillover_max_abs",
}
_BOOLS = {"labelled", "anchor_valid"}
_NULLABLE = {"known_or_novel", "class_id"}
GAIN_ARROW_SCHEMA = pa.schema([
    pa.field(
        field.name,
        pa.int64() if field.name in _INTS else
        pa.float64() if field.name in _FLOATS else
        pa.bool_() if field.name in _BOOLS else pa.string(),
        nullable=field.name in _NULLABLE,
    )
    for field in fields(GainRecord)
], metadata={b"deltasub_gain_schema_version": b"1"})
