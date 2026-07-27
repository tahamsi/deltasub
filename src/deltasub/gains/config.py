from __future__ import annotations

from pathlib import Path
import yaml

from .planning import CandidatePlan

TOP_KEYS = {
    "schema_version", "test_only", "dataset", "backbone", "selex", "subtokens",
    "collection", "output_root",
}
DATASET_KEYS = {"name", "manifest", "manifest_sha256", "split_report", "split_report_sha256"}
BACKBONE_KEYS = {"name", "source_root", "checkpoint", "checkpoint_sha256"}
SELEX_KEYS = {"equivalence_gate", "equivalence_gate_sha256"}
SUBTOKEN_KEYS = {"config", "config_sha256"}
COLLECTION_KEYS = {
    "batch_size", "candidate", "seed", "precision", "device", "shard_size",
    "deterministic_algorithms", "counterfactual_chunk_size", "resume_policy",
}
CANDIDATE_KEYS = {
    "mode", "parent_indices", "start", "stop", "sample_limit", "batch_limit",
}


def _unknown(value, allowed, label):
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unknown {label} keys: {sorted(extra)}")


def load_gain_config(path: str | Path) -> tuple[dict, CandidatePlan]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("gain configuration must be a mapping")
    _unknown(value, TOP_KEYS, "top-level")
    for key in TOP_KEYS:
        if key not in value:
            raise ValueError(f"missing critical configuration key: {key}")
    if value["schema_version"] != 1:
        raise ValueError("unsupported M4 configuration schema")
    for key, allowed in (
        ("dataset", DATASET_KEYS), ("backbone", BACKBONE_KEYS),
        ("selex", SELEX_KEYS), ("subtokens", SUBTOKEN_KEYS),
        ("collection", COLLECTION_KEYS),
    ):
        if not isinstance(value[key], dict):
            raise ValueError(f"{key} must be a mapping")
        _unknown(value[key], allowed, key)
        missing = allowed - set(value[key])
        if missing:
            raise ValueError(f"missing {key} provenance: {sorted(missing)}")
    if not isinstance(value["collection"]["candidate"], dict):
        raise ValueError("candidate must be a mapping")
    _unknown(value["collection"]["candidate"], CANDIDATE_KEYS, "candidate")
    plan = CandidatePlan.from_config(value["collection"]["candidate"])
    collection = value["collection"]
    if collection["precision"] not in {"fp32", "bf16"}:
        raise ValueError("unsupported precision")
    if collection["device"] not in {"cpu", "cuda:0"}:
        raise ValueError("device must be explicit cpu or cuda:0")
    if not value["test_only"] and collection["device"] == "cpu":
        raise ValueError("production collection requires explicit cuda:0")
    if not value["test_only"] and value["backbone"]["name"].startswith("test_only"):
        raise ValueError("test-only model forbidden in production")
    if value["test_only"] and value["backbone"]["name"] != "test_only_m4_transformer":
        raise ValueError("fixture must use the explicit test-only model")
    for section, names in (
        ("dataset", ("manifest_sha256", "split_report_sha256")),
        ("backbone", ("checkpoint_sha256",)),
        ("selex", ("equivalence_gate_sha256",)),
        ("subtokens", ("config_sha256",)),
    ):
        for name in names:
            digest = value[section][name]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"missing exact SHA256: {section}.{name}")
    if collection["batch_size"] < 2 or collection["shard_size"] <= 0:
        raise ValueError("invalid batch/shard size")
    if collection["counterfactual_chunk_size"] <= 0:
        raise ValueError("counterfactual chunk size must be positive")
    if collection["deterministic_algorithms"] != "error":
        raise ValueError("M4 supports deterministic_algorithms: error only")
    if collection["resume_policy"] != "verify":
        raise ValueError("M4 supports resume_policy: verify only")
    return value, plan
