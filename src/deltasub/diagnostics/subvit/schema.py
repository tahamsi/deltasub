from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...utils.hashing import stable_hash

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Provenance:
    model: str
    checkpoint_sha256: str
    source_revision: str
    layer: int
    head: int | None
    sample: str
    view: str
    k: int
    context_sha256: str
    configuration_sha256: str

    def validate(self) -> None:
        if not self.model or not self.sample or not self.view or self.layer < 0 or not 0 <= self.k <= 256:
            raise ValueError("invalid SubViT provenance")
        for name in ("checkpoint_sha256", "context_sha256", "configuration_sha256"):
            value = getattr(self, name)
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA256")
        if not self.source_revision:
            raise ValueError("source revision is required")


@dataclass(frozen=True)
class VersionedRecord:
    schema_version: int
    kind: str
    provenance: Provenance
    payload: dict[str, Any]
    record_sha256: str = ""

    def with_hash(self) -> "VersionedRecord":
        value = asdict(self); value.pop("record_sha256")
        return VersionedRecord(self.schema_version, self.kind, self.provenance,
                               self.payload, stable_hash(value))

    def validate(self) -> None:
        self.provenance.validate()
        if self.schema_version != SCHEMA_VERSION or self.kind not in {
            "attention_extraction", "degradation", "selected_head",
            "distillation_example", "diagnostic_comparison",
        }:
            raise ValueError("unsupported M7 record")
        value = asdict(self); digest = value.pop("record_sha256")
        if not digest or stable_hash(value) != digest:
            raise ValueError("record corruption detected")


def make_record(kind: str, provenance: Provenance, payload: dict[str, Any]) -> VersionedRecord:
    record = VersionedRecord(SCHEMA_VERSION, kind, provenance, payload).with_hash()
    record.validate()
    return record
