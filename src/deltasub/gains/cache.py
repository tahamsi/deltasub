from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import GAIN_ARROW_SCHEMA, GainRecord
from ..utils.hashing import sha256_file, stable_hash


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class GainCache:
    """Content-addressed, sharded M4 cache with fail-closed resume."""
    def __init__(self, output_root: str | Path, metadata: dict, *, shard_size: int = 1024,
                 resume: bool = False):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.metadata = dict(metadata)
        self.metadata.pop("cache_id", None)
        self.metadata_sha256 = stable_hash(self.metadata)
        self.cache_id = self.metadata_sha256
        self.root = Path(output_root) / self.cache_id
        self.shards = self.root / "shards"
        self.index_path = self.root / "index.jsonl"
        self.shard_size = shard_size
        self.pending: list[GainRecord] = []
        if self.root.exists() and not resume:
            raise FileExistsError(f"gain cache exists; use resume: {self.root}")
        self.shards.mkdir(parents=True, exist_ok=True)
        for temporary in self.shards.glob("*.tmp"):
            temporary.unlink()
        metadata_path = self.root / "metadata.json"
        envelope = {
            "cache_id": self.cache_id, "metadata_sha256": self.metadata_sha256,
            "metadata": self.metadata,
        }
        if metadata_path.exists():
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
            if actual != envelope:
                raise ValueError("incompatible or manually altered cache metadata")
        else:
            self._atomic_text(metadata_path, canonical_json(envelope) + "\n")
        self._verify_index()
        self.existing_keys = {record.key for record in self.iter_records()}

    @staticmethod
    def _atomic_text(path: Path, value: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)

    def _index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return [json.loads(line) for line in self.index_path.read_text(encoding="utf-8").splitlines() if line]

    def _verify_index(self) -> None:
        expected_number = 0
        for entry in self._index():
            if entry["shard_number"] != expected_number:
                raise ValueError("non-contiguous shard index")
            path = self.root / entry["path"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError("gain shard missing or checksum mismatch")
            table = pq.read_table(path)
            if table.schema != GAIN_ARROW_SCHEMA or table.num_rows != entry["record_count"]:
                raise ValueError("gain shard schema/count mismatch")
            expected_number += 1

    def append(self, records: Iterable[GainRecord]) -> int:
        added = 0
        staged = set()
        for record in records:
            record.validate()
            if record.key in self.existing_keys or record.key in staged:
                continue
            self.pending.append(record)
            staged.add(record.key)
            added += 1
            if len(self.pending) >= self.shard_size:
                self.flush()
        return added

    def flush(self) -> None:
        if not self.pending:
            return
        ordered = sorted(self.pending, key=lambda r: (
            r.batch_context_sha256, r.anchor_batch_position, r.candidate_parent_index
        ))
        number = len(self._index())
        name = f"shard_{number:06d}.parquet"
        path, temporary = self.shards / name, self.shards / f"{name}.tmp"
        table = pa.Table.from_pylist([record.to_dict() for record in ordered], schema=GAIN_ARROW_SCHEMA)
        pq.write_table(table, temporary, compression="zstd", use_dictionary=False,
                       write_statistics=True, data_page_version="1.0")
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        entry = {
            "shard_number": number, "path": f"shards/{name}",
            "record_count": len(ordered), "sha256": digest,
            "first_key": list(ordered[0].key), "last_key": list(ordered[-1].key),
        }
        entries = self._index() + [entry]
        self._atomic_text(
            self.index_path,
            "".join(canonical_json(item) + "\n" for item in entries),
        )
        self.existing_keys.update(record.key for record in ordered)
        self.pending.clear()

    def iter_records(self) -> Iterator[GainRecord]:
        for entry in self._index():
            batches = pq.ParquetFile(self.root / entry["path"]).iter_batches(batch_size=256)
            for batch in batches:
                for value in batch.to_pylist():
                    yield GainRecord(**value)

    def validate(self, *, expected_keys: set[tuple] | None = None) -> dict:
        self.flush()
        self._verify_index()
        records = list(self.iter_records())
        keys = [record.key for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate gain cache keys")
        for record in records:
            record.validate()
            if record.batch_context_sha256 not in self.metadata["batch_context_hashes"]:
                raise ValueError("record belongs to an unknown batch context")
        if expected_keys is None and "planned_key_sha256" in self.metadata:
            complete = (
                len(keys) == self.metadata["planned_record_count"]
                and stable_hash(sorted([list(key) for key in keys]))
                == self.metadata["planned_key_sha256"]
            )
        else:
            complete = expected_keys is not None and set(keys) == expected_keys
        if expected_keys is not None and not set(keys).issubset(expected_keys):
            raise ValueError("cache contains records outside the collection plan")
        core = {
            "cache_id": self.cache_id, "metadata_sha256": self.metadata_sha256,
            "record_count": len(records), "unique_key_count": len(set(keys)),
            "complete": complete,
            "shards": [{"path": x["path"], "sha256": x["sha256"],
                        "record_count": x["record_count"]} for x in self._index()],
            "record_content_sha256": stable_hash([record.to_dict() for record in records]),
        }
        core["deterministic_validation_sha256"] = stable_hash(core)
        self._atomic_text(self.root / "validation.json", json.dumps(core, indent=2, sort_keys=True) + "\n")
        return core


def open_cache(path: str | Path) -> GainCache:
    root = Path(path)
    envelope = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    cache = object.__new__(GainCache)
    cache.metadata = envelope["metadata"]
    cache.metadata_sha256 = envelope["metadata_sha256"]
    cache.cache_id = envelope["cache_id"]
    if stable_hash(cache.metadata) != cache.metadata_sha256 or cache.cache_id != cache.metadata_sha256:
        raise ValueError("wrong metadata hash")
    cache.root, cache.shards, cache.index_path = root, root / "shards", root / "index.jsonl"
    cache.shard_size, cache.pending = 1, []
    cache._verify_index()
    cache.existing_keys = {record.key for record in cache.iter_records()}
    return cache


def inspect_cache(path: str | Path) -> dict:
    cache = open_cache(path)
    records = list(cache.iter_records())
    gains = [r.gain for r in records if r.anchor_valid]
    candidates = sorted({r.candidate_parent_index for r in records})
    validation = json.loads((cache.root / "validation.json").read_text()) if (
        cache.root / "validation.json").is_file() else {"complete": False}
    import statistics
    return {
        "metadata": cache.metadata, "cache_id": cache.cache_id,
        "completion_state": "complete" if validation.get("complete") else "partial",
        "record_count": len(records), "unique_sample_count": len({r.sample_id for r in records}),
        "candidate_coverage": candidates, "invalid_anchor_count": sum(not r.anchor_valid for r in records),
        "gain_summary": ({
            "minimum": min(gains), "maximum": max(gains), "mean": statistics.fmean(gains),
            "standard_deviation": statistics.pstdev(gains),
        } if gains else None),
        "shard_hashes": [x["sha256"] for x in cache._index()],
        "provenance_hashes": {
            "metadata": cache.metadata_sha256,
            "configuration": cache.metadata.get("configuration_sha256"),
            "batch_contexts": cache.metadata.get("batch_context_hashes"),
        },
    }
