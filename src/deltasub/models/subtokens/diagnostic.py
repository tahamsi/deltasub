"""Non-reportable M3 validation fixture and strict configuration loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml
from torch import nn

from .assembly import assemble_tokens
from .geometry import (
    extract_parent_patches, reconstruct_images, reconstruct_parent_patches,
    subdivide_parent_patches,
)
from .haar import HaarDetails
from .projection import ChildProjector, enforce_parent_consistency

REQUIRED = {
    "schema_version", "test_only_fixture", "parent_patch_size", "child_patch_size",
    "child_order", "haar_mode_order", "embedding_dimension",
    "child_projector_trainable", "mode_embedding_initialization",
    "consistency_loss_weight", "selected_parent_fixture_count", "dtype", "device",
}


def load_subtoken_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("subtoken configuration must be a mapping")
    unknown, missing = set(value) - REQUIRED, REQUIRED - set(value)
    if unknown or missing:
        raise ValueError(f"invalid subtoken fields: unknown={sorted(unknown)} missing={sorted(missing)}")
    fixed = {
        "schema_version": 1, "parent_patch_size": 14, "child_patch_size": 7,
        "child_order": ["top_left", "top_right", "bottom_left", "bottom_right"],
        "haar_mode_order": ["horizontal", "vertical", "diagonal"],
        "mode_embedding_initialization": "zeros",
    }
    for key, expected in fixed.items():
        if value[key] != expected:
            raise ValueError(f"{key} must equal {expected!r}")
    if value["embedding_dimension"] != 768 and not value["test_only_fixture"]:
        raise ValueError("production embedding_dimension must be 768")
    if not 0 <= value["selected_parent_fixture_count"] <= 256:
        raise ValueError("selected_parent_fixture_count must be between 0 and 256")
    if value["dtype"] not in {"float32", "bfloat16"} or value["device"] not in {"cpu", "cuda:0"}:
        raise ValueError("dtype/device must be float32|bfloat16 and cpu|cuda:0")
    return value


def run_fixture_diagnostic(config_path: str | Path) -> dict:
    config = load_subtoken_config(config_path)
    if not config["test_only_fixture"]:
        raise ValueError("this command currently requires test_only_fixture: true")
    torch.manual_seed(0)
    dtype = getattr(torch, config["dtype"])
    device = torch.device(config["device"])
    images = torch.arange(3 * 224 * 224, dtype=torch.float32, device=device).reshape(1, 3, 224, 224)
    images = (images.remainder(251) / 251).to(dtype)
    parents = extract_parent_patches(images)
    children = subdivide_parent_patches(parents)
    dim = config["embedding_dimension"]
    projection = nn.Conv2d(3, dim, 14, 14, device=device, dtype=dtype)
    projector = ChildProjector(
        projection, trainable=config["child_projector_trainable"],
        provenance="TEST-ONLY deterministic M3 fixture",
    )
    parent_tokens = projection(images).flatten(2).transpose(1, 2)
    raw = projector(children)
    consistent = enforce_parent_consistency(raw, parent_tokens)
    haar = HaarDetails().to(device=device, dtype=dtype)
    details = haar(consistent.consistent)
    rebuilt = haar.reconstruct(parent_tokens, details)
    selected = torch.zeros((1, 256), dtype=torch.bool, device=device)
    selected[:, :config["selected_parent_fixture_count"]] = True
    assembled = assemble_tokens(parent_tokens[:, :0], parent_tokens, details, selected)
    metadata = torch.stack((
        assembled.token_kind.to(torch.int64), assembled.parent_index,
        assembled.detail_mode.to(torch.int64),
    ), dim=-1).cpu().numpy().tobytes()
    return {
        "label": "M3 TEST-ONLY FIXTURE; DIAGNOSTIC AND NON-REPORTABLE",
        "parent_extraction_shape": list(parents.shape),
        "child_subdivision_shape": list(children.shape),
        "parent_reconstruction_error": float((reconstruct_images(parents) - images).abs().max()),
        "child_reconstruction_error": float((reconstruct_parent_patches(children) - parents).abs().max()),
        "initial_projection_consistency_error": float(
            (raw.mean(2) - parent_tokens).abs().max().detach()
        ),
        "hard_consistency_error": float(consistent.max_absolute_error.detach()),
        "haar_orthogonality_error": float(haar.orthogonality_error()),
        "haar_reconstruction_error": float(
            (rebuilt - consistent.consistent).abs().max().detach()
        ),
        "selected_parent_count": int(assembled.selected_parent_count[0]),
        "effective_token_count": int(assembled.effective_token_count[0]),
        "padded_token_count": assembled.padded_token_count,
        "deterministic_metadata_hash": hashlib.sha256(metadata).hexdigest(),
    }
