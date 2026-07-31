"""Exact, provenance-carrying SSB/SelEx class split loading."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import yaml


def deterministic_class_split(class_names: list[str], known_fraction: float = 0.5) -> dict:
    """Legacy synthetic-fixture helper; never an SSB split substitute."""
    if not 0 < known_fraction < 1:
        raise ValueError("known_fraction must lie strictly between zero and one")
    ordered = sorted(class_names)
    cut = round(len(ordered) * known_fraction)
    return {"known": ordered[:cut], "novel": ordered[cut:]}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_lists(value: object) -> tuple[list[int], list[int]]:
    preserve_novel_order = False
    if isinstance(value, dict):
        known = value.get(
            "known_class_ids",
            value.get("known", value.get("train_classes", value.get("known_classes"))),
        )
        if "unknown_classes" in value:
            unknown = value["unknown_classes"]
            if not isinstance(unknown, dict):
                raise ValueError("unknown_classes must be a difficulty-partition dictionary")
            expected_partitions = {"Easy", "Medium", "Hard"}
            if set(unknown) != expected_partitions:
                raise ValueError(
                    "unknown_classes must contain exactly Easy, Medium, and Hard partitions"
                )
            if any(not isinstance(unknown[name], list) for name in expected_partitions):
                raise ValueError("unknown_classes difficulty partitions must be lists")
            # Match the class ordering used by the pinned upstream SSB loader.
            novel = unknown["Hard"] + unknown["Medium"] + unknown["Easy"]
            preserve_novel_order = True
        else:
            novel = value.get(
                "novel_class_ids", value.get("novel", value.get("unlabeled_classes"))
            )
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        known, novel = value
    else:
        raise ValueError("split file must contain known and novel class ID lists")
    if known is None or novel is None:
        raise ValueError("split file is missing known or novel class IDs")
    try:
        known_ids = [int(item) for item in known]
        novel_ids = [int(item) for item in novel]
    except (TypeError, ValueError) as error:
        raise ValueError("split class IDs must be integers") from error
    if len(known_ids) != len(set(known_ids)) or len(novel_ids) != len(set(novel_ids)):
        raise ValueError("split class IDs contain duplicates")
    if set(known_ids) & set(novel_ids):
        raise ValueError("known and novel class IDs overlap")
    if any(item < 0 for item in known_ids + novel_ids):
        raise ValueError("split class IDs must be non-negative")
    return sorted(known_ids), novel_ids if preserve_novel_order else sorted(novel_ids)


def load_split_definition(
    path: str | Path,
    *,
    upstream_repository: str,
    pinned_commit: str,
    expected_revision: str,
    labelled_proportion: float,
    expected_sha256: str | None = None,
) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required split file does not exist: {source}")
    if expected_revision != pinned_commit:
        raise ValueError(
            f"split revision mismatch: requested {expected_revision}, pinned {pinned_commit}"
        )
    observed_hash = file_sha256(source)
    if expected_sha256 is not None and observed_hash != expected_sha256:
        raise ValueError(
            f"split file SHA256 mismatch: expected {expected_sha256}, observed {observed_hash}"
        )
    if not 0 <= labelled_proportion <= 1:
        raise ValueError("labelled_proportion must lie between zero and one")
    suffix = source.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    elif suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
    elif suffix in {".pkl", ".pickle"}:
        # Pinned upstream SSB files are Python pickles. Only load trusted, locally
        # supplied files whose provenance/hash the caller has verified.
        with source.open("rb") as stream:
            value = pickle.load(stream)
    else:
        raise ValueError(f"unsupported split file format: {suffix}")
    known, novel = _split_lists(value)
    return {
        "upstream_repository": upstream_repository,
        "pinned_commit": pinned_commit,
        "source_file_path": str(source),
        "source_file_sha256": observed_hash,
        "known_class_ids": known,
        "novel_class_ids": novel,
        "labelled_proportion": float(labelled_proportion),
        "validation_outcome": "passed",
    }
