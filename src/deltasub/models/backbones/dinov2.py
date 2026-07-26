"""Pinned, local-only integration of the official DINOv2 ViT-B/14 source."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import torch
from torch import nn

from ...utils.hashing import sha256_file

DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_SOURCE_HASHES = {
    "dinov2/models/vision_transformer.py": "7799a260f2d7d0fe197331d08502fb8c542f9b7424723650f6a39b64fa2639ea",
    "dinov2/hub/backbones.py": "871fca671b12a9ff02e810654baf509e97ccf461bf8196ce5ddeefff2fd87d3e",
    "hubconf.py": "c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64",
}


@dataclass(frozen=True)
class DINOv2Geometry:
    model_name: str = "dinov2_vitb14"
    input_size: int = 224
    patch_size: int = 14
    grid_size: tuple[int, int] = (16, 16)
    patch_tokens: int = 256
    embed_dim: int = 768
    depth: int = 12


@dataclass
class BackboneOutput:
    pre_transformer_patch_embeddings: torch.Tensor
    patch_tokens: torch.Tensor
    cls_token: torch.Tensor
    register_tokens: torch.Tensor
    intermediate_patch_tokens: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class CheckpointInspection:
    compatible: bool
    checkpoint_sha256: str
    container: str
    normalized_prefix: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    accepted_noncritical_differences: tuple[str, ...]
    architecture: dict
    source_revision: str
    source_hashes: dict


def _verify_source(source_root: str | Path) -> tuple[Path, dict[str, str]]:
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"pinned DINOv2 source checkout is absent: {root}")
    hashes = {}
    for relative, expected in DINOV2_SOURCE_HASHES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned DINOv2 source file is absent: {path}")
        hashes[relative] = sha256_file(path)
        if hashes[relative] != expected:
            raise ValueError(f"DINOv2 source hash mismatch: {relative}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("DINOv2 source_root must be a git checkout at the pinned revision") from error
    if revision != DINOV2_REVISION:
        raise ValueError(f"DINOv2 source revision mismatch: {revision}")
    return root, hashes


def construct_official_vitb14(source_root: str | Path, *, register_tokens: int = 0) -> tuple[nn.Module, dict[str, str]]:
    """Construct the pinned official architecture without downloading weights."""
    root, hashes = _verify_source(source_root)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        for name in tuple(sys.modules):
            if name == "dinov2" or name.startswith("dinov2."):
                del sys.modules[name]
        module = importlib.import_module("dinov2.models.vision_transformer")
        model = module.vit_base(
            patch_size=14, img_size=224, init_values=1.0, block_chunks=0,
            num_register_tokens=register_tokens,
            interpolate_antialias=bool(register_tokens),
            interpolate_offset=0.0 if register_tokens else 0.1,
        )
    finally:
        sys.path[:] = old_path
    return model, hashes


def _extract_state_dict(value) -> tuple[dict[str, torch.Tensor], str]:
    container = "root"
    for key in ("state_dict", "model", "teacher", "student"):
        if isinstance(value, Mapping) and key in value and isinstance(value[key], Mapping):
            value, container = value[key], key
            break
    if not isinstance(value, Mapping) or not value:
        raise ValueError("checkpoint has no non-empty state dictionary")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in value.items()):
        raise ValueError("checkpoint state dictionary contains non-tensor values")
    return dict(value), container


def _normalize_prefix(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str]:
    documented = ("module.backbone.", "module.", "backbone.", "teacher.backbone.", "")
    for prefix in documented:
        normalized = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if normalized and ("patch_embed.proj.weight" in normalized or "cls_token" in normalized):
            return normalized, prefix
    return state, ""


def _register_count(state: Mapping[str, torch.Tensor]) -> int:
    value = state.get("register_tokens")
    return int(value.shape[1]) if value is not None and value.ndim == 3 else 0


def inspect_official_checkpoint(
    checkpoint_path: str | Path, expected_sha256: str, *, source_root: str | Path,
    model_name: str = "dinov2_vitb14",
) -> tuple[nn.Module, CheckpointInspection]:
    if model_name != "dinov2_vitb14":
        raise ValueError(f"unsupported production architecture: {model_name}")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"official DINOv2 checkpoint is absent: {path}")
    if len(expected_sha256) != 64:
        raise ValueError("an exact 64-character checkpoint SHA256 is required")
    digest = sha256_file(path)
    if digest != expected_sha256.lower():
        raise ValueError("DINOv2 checkpoint SHA256 mismatch")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"malformed DINOv2 checkpoint: {error}") from error
    state, container = _extract_state_dict(raw)
    state, prefix = _normalize_prefix(state)
    model, source_hashes = construct_official_vitb14(source_root, register_tokens=_register_count(state))
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatches = sorted(
        f"{key}: checkpoint={tuple(state[key].shape)} model={tuple(expected[key].shape)}"
        for key in set(expected) & set(state) if state[key].shape != expected[key].shape
    )
    noncritical = tuple(sorted(k for k in unexpected if k.startswith("head.")))
    critical_unexpected = [k for k in unexpected if k not in noncritical]
    compatible = not missing and not critical_unexpected and not mismatches
    inspection = CheckpointInspection(
        compatible, digest, container, prefix, tuple(missing), tuple(critical_unexpected),
        tuple(mismatches), noncritical, {
            "model_name": model_name, "patch_size": model.patch_size,
            "embed_dim": model.embed_dim, "depth": model.n_blocks,
            "patch_tokens_at_224": model.patch_embed.num_patches,
            "register_tokens": model.num_register_tokens,
        }, DINOV2_REVISION, source_hashes,
    )
    return model, inspection


class DINOv2Adapter(nn.Module):
    geometry = DINOv2Geometry()

    def __init__(self, model: nn.Module, checkpoint: str | Path, inspection: CheckpointInspection):
        super().__init__()
        if not inspection.compatible:
            raise ValueError("DINOv2 checkpoint is incompatible: " + json.dumps(asdict(inspection), default=str))
        self.model, self.checkpoint = model, Path(checkpoint)
        self.inspection = inspection
        self.checkpoint_sha256 = inspection.checkpoint_sha256
        self._validate_geometry()
        state, _ = _extract_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=True))
        state, _ = _normalize_prefix(state)
        for key in inspection.accepted_noncritical_differences:
            state.pop(key, None)
        self.model.load_state_dict(state, strict=True)

    @classmethod
    def from_official_checkpoint(cls, checkpoint_path, expected_sha256, *, source_root, model_name="dinov2_vitb14"):
        model, inspection = inspect_official_checkpoint(
            checkpoint_path, expected_sha256, source_root=source_root, model_name=model_name
        )
        return cls(model, checkpoint_path, inspection)

    def _validate_geometry(self) -> None:
        patch = self.model.patch_size[0] if isinstance(self.model.patch_size, (tuple, list)) else self.model.patch_size
        actual = (patch, self.model.embed_dim, self.model.n_blocks, self.model.patch_embed.num_patches)
        if actual != (14, 768, 12, 256):
            raise ValueError(f"incompatible DINOv2 ViT-B/14 geometry: {actual}")

    def set_trainable_blocks(self, policy: str | int) -> dict[str, int]:
        blocks = list(self.model.blocks)
        count = {"frozen": 0, "final": 1, "full": len(blocks)}.get(policy, policy if isinstance(policy, int) else -1)
        if not isinstance(count, int) or not 0 <= count <= len(blocks):
            raise ValueError("trainability must be frozen, final, full, or an integer final-N")
        for p in self.model.parameters():
            p.requires_grad = policy == "full"
        if policy != "full":
            for block in blocks[-count:] if count else []:
                for p in block.parameters():
                    p.requires_grad = True
            if count:
                for p in self.model.norm.parameters():
                    p.requires_grad = True
        return self.parameter_report()

    def parameter_report(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def forward(self, images: torch.Tensor, intermediate_blocks: Sequence[int] = ()) -> BackboneOutput:
        if tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError("DINOv2 ViT-B/14 requires input [B, 3, 224, 224]")
        pre = self.model.patch_embed(images)
        if pre.ndim != 3 or pre.shape[1:] != (256, 768):
            raise ValueError(f"official patch_embed contract mismatch: {tuple(pre.shape)}")
        result = self.model.forward_features(images)
        intermediate = tuple(self.model.get_intermediate_layers(images, n=list(intermediate_blocks))) if intermediate_blocks else ()
        return BackboneOutput(pre, result["x_norm_patchtokens"], result["x_norm_clstoken"],
                              result["x_norm_regtokens"], intermediate)

    def metadata(self):
        return {**asdict(self.geometry), **asdict(self.inspection), **self.parameter_report(),
                "checkpoint": str(self.checkpoint)}


class TestOnlyTinyBackbone(nn.Module):
    """Explicit CPU-test fixture; production configuration rejects this class."""
    test_only = True

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.projection = nn.Sequential(nn.Conv2d(3, embed_dim, 16, 16), nn.GELU())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images).flatten(2).mean(2)
