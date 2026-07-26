from __future__ import annotations

import json
from pathlib import Path

from ..utils.hashing import stable_hash


REQUIRED_FIELDS = {
    "sample_id",
    "image_path",
    "original_class",
    "class_status",
    "label_status",
    "split",
    "dataset_version",
    "source_checksum",
}


def validate_manifest(records: list[dict]) -> str:
    sample_ids = set()
    for record in records:
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"manifest record missing fields: {sorted(missing)}")
        if record["sample_id"] in sample_ids:
            raise ValueError(f"duplicate sample_id: {record['sample_id']}")
        sample_ids.add(record["sample_id"])
        if record["class_status"] not in {"known", "novel"}:
            raise ValueError("class_status must be known or novel")
        if record["label_status"] not in {"labelled", "unlabelled"}:
            raise ValueError("label_status must be labelled or unlabelled")
    return stable_hash(records)


def write_manifest(records: list[dict], path: str | Path) -> str:
    checksum = validate_manifest(records)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    target.with_suffix(target.suffix + ".sha256").write_text(checksum + "\n", encoding="utf-8")
    return checksum


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    validate_manifest(records)
    return records
