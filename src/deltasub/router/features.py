from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import torch

from ..gains.schema import GainRecord
from ..utils.hashing import sha256_file, stable_hash
from ..utils.checkpointing import atomic_torch_save, load_checkpoint


@dataclass(frozen=True)
class RouterFeatureBatch:
    sample_ids: tuple[str, ...]
    view_layout_identifier: str
    batch_context_sha256: str
    parent_embeddings: torch.Tensor
    model_checkpoint_sha256: str
    dinov2_source_revision: str
    dataset_manifest_sha256: str
    gain_configuration_sha256: str
    child_projector_state_sha256: str

    def validate(self) -> None:
        if self.parent_embeddings.ndim != 3 or self.parent_embeddings.shape[1:] != (256, self.parent_embeddings.shape[-1]):
            raise ValueError("feature batch must be [B, 256, D]")
        if len(self.sample_ids) != self.parent_embeddings.shape[0] or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("ordered sample IDs must be unique and match the batch")
        if not torch.isfinite(self.parent_embeddings).all():
            raise ValueError("router features must be finite")

    @property
    def source_hash(self) -> str:
        return stable_hash({
            "sample_ids": self.sample_ids, "view_layout_identifier": self.view_layout_identifier,
            "batch_context_sha256": self.batch_context_sha256,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "dinov2_source_revision": self.dinov2_source_revision,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "gain_configuration_sha256": self.gain_configuration_sha256,
            "child_projector_state_sha256": self.child_projector_state_sha256,
        })


@torch.no_grad()
def extract_pre_transformer_features(adapter, images: torch.Tensor) -> torch.Tensor:
    if getattr(adapter, "training", False):
        raise ValueError("frozen feature extraction requires eval mode")
    return adapter.pre_transformer_parent_embeddings(images).detach()


def join_features(
    records: Iterable[GainRecord], batches: Iterable[RouterFeatureBatch]
) -> dict[tuple[str, str, int, int], torch.Tensor]:
    by_identity: dict[tuple[str, str], tuple[RouterFeatureBatch, int]] = {}
    for batch in batches:
        batch.validate()
        for index, sample_id in enumerate(batch.sample_ids):
            identity = (batch.batch_context_sha256, sample_id)
            if identity in by_identity:
                raise ValueError("duplicate feature for exact sample/context identity")
            by_identity[identity] = (batch, index)
    result = {}
    for record in records:
        match = by_identity.get((record.batch_context_sha256, record.sample_id))
        if match is None:
            raise ValueError("missing exact feature for gain record")
        batch, index = match
        checks = (
            (record.view_layout_identifier, batch.view_layout_identifier, "view layout"),
            (record.dataset_manifest_sha256, batch.dataset_manifest_sha256, "manifest"),
            (record.model_checkpoint_sha256, batch.model_checkpoint_sha256, "checkpoint"),
            (record.configuration_sha256, batch.gain_configuration_sha256, "configuration"),
        )
        for expected, actual, label in checks:
            if expected != actual:
                raise ValueError(f"feature/gain {label} mismatch")
        result[record.key] = batch.parent_embeddings[index, record.candidate_parent_index].detach()
    return result


def save_feature_cache(batches: Iterable[RouterFeatureBatch], path: str | Path) -> str:
    values = list(batches)
    for batch in values:
        batch.validate()
    payload = {
        "schema_version": 1,
        "batches": [
            {
                **{name: getattr(batch, name) for name in (
                    "sample_ids", "view_layout_identifier", "batch_context_sha256",
                    "model_checkpoint_sha256", "dinov2_source_revision",
                    "dataset_manifest_sha256", "gain_configuration_sha256",
                    "child_projector_state_sha256",
                )},
                "parent_embeddings": batch.parent_embeddings.detach().cpu(),
                "source_hash": batch.source_hash,
            } for batch in values
        ],
    }
    payload["cache_hash"] = stable_hash([
        {key: value for key, value in item.items() if key != "parent_embeddings"}
        | {"tensor": stable_hash(item["parent_embeddings"].tolist())}
        for item in payload["batches"]
    ])
    atomic_torch_save(payload, path)
    return payload["cache_hash"]


def load_feature_cache(path: str | Path) -> list[RouterFeatureBatch]:
    payload = load_checkpoint(path, map_location="cpu")
    if payload.get("schema_version") != 1 or not isinstance(payload.get("batches"), list):
        raise ValueError("unsupported router feature-cache schema")
    batches = []
    for item in payload["batches"]:
        expected = item.pop("source_hash")
        batch = RouterFeatureBatch(**item)
        batch.validate()
        if batch.source_hash != expected:
            raise ValueError("router feature-cache source hash mismatch")
        batches.append(batch)
    actual = stable_hash([
        {
            **{name: getattr(batch, name) for name in (
                "sample_ids", "view_layout_identifier", "batch_context_sha256",
                "model_checkpoint_sha256", "dinov2_source_revision",
                "dataset_manifest_sha256", "gain_configuration_sha256",
                "child_projector_state_sha256",
            )},
            "source_hash": batch.source_hash,
            "tensor": stable_hash(batch.parent_embeddings.tolist()),
        } for batch in batches
    ])
    if actual != payload.get("cache_hash"):
        raise ValueError("router feature-cache content hash mismatch")
    return batches
