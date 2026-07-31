from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from ..gains.schema import GainRecord

ROUTER_EXAMPLE_SCHEMA_VERSION = 1


def gain_key_string(key: tuple[str, str, int, int]) -> str:
    return "|".join(map(str, key))


@dataclass(frozen=True)
class RouterExample:
    schema_version: int
    gain_cache_id: str
    gain_record_key: str
    batch_context_sha256: str
    sample_id: str
    anchor_batch_position: int
    candidate_parent_index: int
    parent_row: int
    parent_column: int
    labelled: bool
    known_or_novel: str | None
    target_gain: float
    base_loss: float
    counterfactual_loss: float
    anchor_valid: bool
    feature_source_hash: str
    router_feature_hash: str
    split_assignment: str
    sampling_stream: str = "unassigned"
    sample_weight: float = 1.0
    dataset_manifest_sha256: str = ""
    model_checkpoint_sha256: str = ""
    gain_configuration_sha256: str = ""

    def validate(self) -> None:
        if self.schema_version != ROUTER_EXAMPLE_SCHEMA_VERSION:
            raise ValueError("unsupported router-example schema")
        if not 0 <= self.candidate_parent_index < 256:
            raise ValueError("candidate index outside [0, 255]")
        if (self.parent_row, self.parent_column) != divmod(self.candidate_parent_index, 16):
            raise ValueError("parent geometry mismatch")
        if self.split_assignment not in {"train", "validation", "test"}:
            raise ValueError("invalid split assignment")
        if not all(math.isfinite(x) for x in (
            self.target_gain, self.base_loss, self.counterfactual_loss, self.sample_weight
        )):
            raise ValueError("router labels and weights must be finite")
        if self.sample_weight < 0:
            raise ValueError("sample weight must be nonnegative")
        for value in (
            self.gain_cache_id, self.batch_context_sha256, self.feature_source_hash,
            self.router_feature_hash, self.dataset_manifest_sha256,
            self.model_checkpoint_sha256, self.gain_configuration_sha256,
        ):
            if len(value) != 64:
                raise ValueError("router provenance values must be SHA256")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_gain(
        cls, record: GainRecord, *, cache_id: str, feature_source_hash: str,
        router_feature_hash: str, split: str,
    ) -> "RouterExample":
        value = cls(
            ROUTER_EXAMPLE_SCHEMA_VERSION, cache_id, gain_key_string(record.key),
            record.batch_context_sha256, record.sample_id, record.anchor_batch_position,
            record.candidate_parent_index, record.parent_row, record.parent_column,
            record.labelled, record.known_or_novel, record.gain,
            record.base_per_anchor_loss, record.counterfactual_per_anchor_loss,
            record.anchor_valid, feature_source_hash, router_feature_hash, split,
            dataset_manifest_sha256=record.dataset_manifest_sha256,
            model_checkpoint_sha256=record.model_checkpoint_sha256,
            gain_configuration_sha256=record.configuration_sha256,
        )
        value.validate()
        return value
