from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Iterable

from ..gains.schema import GainRecord
from ..utils.hashing import stable_hash


@dataclass(frozen=True)
class SplitManifest:
    schema_version: int
    seed: int
    fractions: dict[str, float]
    assignments: dict[str, str]
    summary: dict
    sha256: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "seed": self.seed,
            "fractions": self.fractions, "assignments": self.assignments,
            "summary": self.summary, "sha256": self.sha256,
        }


def _unit_hash(sample_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def create_split_manifest(
    records: Iterable[GainRecord], *, train_fraction: float = .8,
    validation_fraction: float = .2, test_fraction: float = 0., seed: int = 0,
) -> SplitManifest:
    records = list(records)
    fractions = {"train": train_fraction, "validation": validation_fraction, "test": test_fraction}
    if any(x < 0 for x in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-12:
        raise ValueError("split fractions must be nonnegative and sum to one")
    samples = sorted({record.sample_id for record in records})
    assignments = {}
    for sample in samples:
        value = _unit_hash(sample, seed)
        assignments[sample] = (
            "train" if value < train_fraction else
            "validation" if value < train_fraction + validation_fraction else "test"
        )
    # Tiny deterministic fixtures can hash pathologically; deterministically rebalance
    # required empty splits without ever separating a sample's records.
    for required, fraction in fractions.items():
        if fraction and required not in assignments.values():
            donors = sorted((s for s in samples if list(assignments.values()).count(assignments[s]) > 1),
                            key=lambda s: (_unit_hash(s, seed), s))
            if not donors:
                raise ValueError(f"empty required split: {required}")
            assignments[donors[-1]] = required
    record_counts = Counter(assignments[r.sample_id] for r in records)
    sample_counts = Counter(assignments.values())
    composition = {
        split: {
            "labelled": sum(r.labelled for r in records if assignments[r.sample_id] == split),
            "unlabelled": sum(not r.labelled for r in records if assignments[r.sample_id] == split),
            "known": sum(r.known_or_novel == "known" for r in records if assignments[r.sample_id] == split),
            "novel": sum(r.known_or_novel == "novel" for r in records if assignments[r.sample_id] == split),
        } for split in fractions
    }
    summary = {"sample_counts": dict(sample_counts), "record_counts": dict(record_counts),
               "composition": composition}
    core = {"schema_version": 1, "seed": seed, "fractions": fractions,
            "assignments": assignments, "summary": summary}
    return SplitManifest(**core, sha256=stable_hash(core))


def require_matching_manifest(current: SplitManifest, prior: dict) -> None:
    if current.sha256 != prior.get("sha256"):
        raise ValueError("split manifest mismatch on resume")
