from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch

from ..router.model import GainRouter, RouterConfig
from ..utils.checkpointing import load_checkpoint
from ..utils.hashing import sha256_file


@dataclass(frozen=True)
class RouterProvenance:
    checkpoint_sha256: str
    m4_cache_id: str
    feature_source_hash: str
    dinov2_checkpoint_sha256: str
    dinov2_source_revision: str
    configuration_hash: str


def load_router_checkpoint(
    path: str | Path, config: RouterConfig, *, expected_checkpoint_sha256: str,
    expected_m4_cache_id: str, expected_feature_source_hash: str,
    expected_dinov2_checkpoint_sha256: str, expected_dinov2_source_revision: str,
    precision: str, map_location: str | torch.device, production: bool,
) -> tuple[GainRouter, RouterProvenance]:
    if map_location is None:
        raise ValueError("explicit map_location is required")
    actual_hash = sha256_file(path)
    if actual_hash != expected_checkpoint_sha256:
        raise ValueError("router checkpoint SHA256 mismatch")
    value = load_checkpoint(path, map_location=map_location)
    required = {
        "schema_version", "router_state", "router_configuration_hash", "m4_cache_id",
        "feature_source_hash", "dinov2_checkpoint_hash", "dinov2_source_revision",
        "precision",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"router checkpoint missing provenance: {sorted(missing)}")
    if value["schema_version"] != 1:
        raise ValueError("unsupported router checkpoint schema")
    if production and config.test_only:
        raise ValueError("fixture router is forbidden in production")
    if config.input_dim != (768 if production else config.input_dim):
        raise ValueError("router input dimension mismatch")
    expected = {
        "router_configuration_hash": GainRouter(config).configuration_hash,
        "m4_cache_id": expected_m4_cache_id,
        "feature_source_hash": expected_feature_source_hash,
        "dinov2_checkpoint_hash": expected_dinov2_checkpoint_sha256,
        "dinov2_source_revision": expected_dinov2_source_revision,
        "precision": precision,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f"router checkpoint mismatch: {key}")
    model = GainRouter(config)
    model.load_state_dict(value["router_state"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("router checkpoint contains non-finite parameters")
    return model, RouterProvenance(
        actual_hash, value["m4_cache_id"], value["feature_source_hash"],
        value["dinov2_checkpoint_hash"], value["dinov2_source_revision"],
        value["router_configuration_hash"],
    )
