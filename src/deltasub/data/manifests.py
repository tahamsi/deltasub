"""Versioned, deterministic real-dataset manifests.

Synthetic fixtures may exercise this module, but manifests emitted here are never
presented as benchmark data unless they were parsed from an installed dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable


MANIFEST_SCHEMA_VERSION = 1
REQUIRED_FIELDS = {
    "sample_id",
    "dataset",
    "image_path",
    "original_class_id",
    "original_class_name",
    "known_or_novel",
    "labelled_or_unlabelled",
    "train_or_test_split",
    "bounding_box",
    "source_archive_checksum",
    "split_source",
    "split_revision",
    "manifest_schema_version",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _legacy_record(record: dict) -> dict:
    """Accept the pre-M1 synthetic smoke record shape without emitting it."""
    if REQUIRED_FIELDS <= record.keys():
        return dict(record)
    legacy = {
        "original_class",
        "class_status",
        "label_status",
        "split",
        "dataset_version",
        "source_checksum",
    }
    if not legacy <= record.keys():
        return dict(record)
    value = dict(record)
    original = value["original_class"]
    try:
        class_id = int(original)
    except (TypeError, ValueError):
        class_id = 0
    return {
        "sample_id": value["sample_id"],
        "dataset": value["dataset_version"],
        "image_path": value["image_path"],
        "original_class_id": class_id,
        "original_class_name": str(original),
        "known_or_novel": value["class_status"],
        "labelled_or_unlabelled": value["label_status"],
        "train_or_test_split": value["split"],
        "bounding_box": None,
        "source_archive_checksum": value["source_checksum"],
        "split_source": "legacy-synthetic-fixture",
        "split_revision": "legacy",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
    }


def canonical_records(records: Iterable[dict]) -> list[dict]:
    normalized = [_legacy_record(record) for record in records]
    return sorted(
        normalized,
        key=lambda item: (
            str(item.get("sample_id", "")),
            str(item.get("image_path", "")),
        ),
    )


def serialize_manifest(records: Iterable[dict]) -> bytes:
    ordered = canonical_records(records)
    validate_manifest(ordered)
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in ordered
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def manifest_sha256(records: Iterable[dict]) -> str:
    return hashlib.sha256(serialize_manifest(records)).hexdigest()


def validate_manifest(
    records: Iterable[dict],
    *,
    dataset_root: str | Path | None = None,
    known_class_ids: Iterable[int] | None = None,
    novel_class_ids: Iterable[int] | None = None,
    check_images: bool = False,
) -> str:
    ordered = canonical_records(records)
    sample_ids: set[str] = set()
    known = set(known_class_ids or ())
    novel = set(novel_class_ids or ())
    if known & novel:
        raise ValueError("known and novel class IDs overlap")
    allowed_ids = known | novel
    root = Path(dataset_root) if dataset_root is not None else None
    for index, record in enumerate(ordered):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"manifest record {index} missing fields: {sorted(missing)}")
        extra_version = record["manifest_schema_version"]
        if extra_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest_schema_version {extra_version}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        class_id = record["original_class_id"]
        if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
            raise ValueError(f"invalid original_class_id for {sample_id}: {class_id!r}")
        if allowed_ids and class_id not in allowed_ids:
            raise ValueError(f"class ID {class_id} is absent from the validated split")
        status = record["known_or_novel"]
        if status not in {"known", "novel"}:
            raise ValueError("known_or_novel must be known or novel")
        if known and status == "known" and class_id not in known:
            raise ValueError(f"class ID {class_id} is incorrectly assigned known")
        if novel and status == "novel" and class_id not in novel:
            raise ValueError(f"class ID {class_id} is incorrectly assigned novel")
        label = record["labelled_or_unlabelled"]
        if label not in {"labelled", "unlabelled"}:
            raise ValueError("labelled_or_unlabelled must be labelled or unlabelled")
        if status == "novel" and label == "labelled":
            raise ValueError("novel samples cannot be labelled")
        if record["train_or_test_split"] not in {"train", "test"}:
            raise ValueError("train_or_test_split must be train or test")
        if record["train_or_test_split"] == "test" and label == "labelled":
            raise ValueError("test samples cannot be labelled")
        box = record["bounding_box"]
        if box is not None:
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box)
                or any(not math.isfinite(float(v)) for v in box)
                or box[2] <= 0
                or box[3] <= 0
            ):
                raise ValueError(f"malformed bounding_box for {sample_id}")
        checksum = record["source_archive_checksum"]
        if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
            raise ValueError(f"source_archive_checksum must be a lowercase SHA256 for {sample_id}")
        for field in ("dataset", "image_path", "split_source", "split_revision"):
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError(f"{field} must be a non-empty string")
        if check_images:
            image = Path(record["image_path"])
            if not image.is_absolute():
                if root is None:
                    raise ValueError("dataset_root is required to validate relative image paths")
                image = root / image
            if not image.is_file():
                raise FileNotFoundError(f"manifest image is missing: {image}")
    payload = (
        ("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for r in ordered) + "\n")
        if ordered
        else ""
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_manifest(records: Iterable[dict], path: str | Path) -> str:
    payload = serialize_manifest(records)
    checksum = hashlib.sha256(payload).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    target.with_suffix(target.suffix + ".sha256").write_text(checksum + "\n", encoding="utf-8")
    return checksum


def read_manifest(path: str | Path, **validation_kwargs) -> list[dict]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"manifest does not exist: {target}")
    records: list[dict] = []
    with target.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed manifest JSON on line {line_number}: {error}") from error
    validate_manifest(records, **validation_kwargs)
    return canonical_records(records)
