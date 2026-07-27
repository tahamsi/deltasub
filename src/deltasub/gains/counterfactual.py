from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Callable

import numpy as np
import torch
from torch import nn

from ..losses.per_anchor_selex import selex_per_anchor
from ..models.subtokens.assembly import AssembledTokens, assemble_tokens
from ..models.subtokens.geometry import extract_parent_patches, subdivide_parent_patches
from ..models.subtokens.haar import HaarDetails
from ..models.subtokens.positions import ParentAwareDetailPositions
from ..models.subtokens.projection import ChildProjector, enforce_parent_consistency
from ..utils.hashing import stable_hash
from ..utils.reproducibility import capture_rng_state, restore_rng_state


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    payload = str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode() + raw
    return hashlib.sha256(payload).hexdigest()


def module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class BatchContext:
    sample_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    image_hashes: tuple[str, ...]
    augmentation_seeds: tuple[int, ...]
    augmentation_parameters_hash: str
    label_hash: str
    labelled_flag_hash: str
    hierarchy_label_hash: str
    confusion_factor_hash: str
    pseudo_label_confidence_hash: str
    model_state_hash: str
    child_projector_state_hash: str
    detail_position_state_hash: str
    head_state_hash: str
    configuration_hash: str
    precision: str
    device: str
    model_mode: str
    source_git_commit: str

    @property
    def sha256(self) -> str:
        return stable_hash(asdict(self))

    def validate_tensors(self, images, labels, labelled, hierarchy_labels,
                         confusion_factor, pseudo_label_confidence) -> None:
        actual_images = tuple(
            tensor_sha256(images[i, v]) for i in range(images.shape[0])
            for v in range(images.shape[1])
        )
        actual = (
            actual_images, tensor_sha256(labels), tensor_sha256(labelled),
            stable_hash([tensor_sha256(x) for x in hierarchy_labels]),
            tensor_sha256(confusion_factor), tensor_sha256(pseudo_label_confidence),
        )
        expected = (
            self.image_hashes, self.label_hash, self.labelled_flag_hash,
            self.hierarchy_label_hash, self.confusion_factor_hash,
            self.pseudo_label_confidence_hash,
        )
        if actual != expected:
            raise ValueError("immutable batch tensor context mismatch")

    @classmethod
    def build(cls, *, images, sample_ids, view_ids, augmentation_seeds,
              augmentation_parameters, labels, labelled, hierarchy_labels, confusion_factor,
              pseudo_label_confidence, model, child_projector, position_module, head,
              configuration_hash, precision, device, source_git_commit):
        if images.ndim != 5 or images.shape[:2] != (len(sample_ids), len(view_ids)):
            raise ValueError("images/view/sample layout mismatch")
        if len(augmentation_seeds) != len(view_ids):
            raise ValueError("one augmentation seed is required per ordered view")
        return cls(
            tuple(sample_ids), tuple(view_ids),
            tuple(tensor_sha256(images[i, v]) for i in range(images.shape[0])
                  for v in range(images.shape[1])),
            tuple(augmentation_seeds), stable_hash(augmentation_parameters),
            tensor_sha256(labels), tensor_sha256(labelled),
            stable_hash([tensor_sha256(x) for x in hierarchy_labels]),
            tensor_sha256(confusion_factor), tensor_sha256(pseudo_label_confidence),
            module_sha256(model), module_sha256(child_projector),
            module_sha256(position_module), module_sha256(head),
            configuration_hash, precision, str(device),
            "eval" if not model.training else "train", source_git_commit,
        )


@dataclass
class GainEvaluation:
    anchor: int
    parent: int
    gain: float
    base_anchor_loss: float
    counterfactual_anchor_loss: float
    base_batch_loss: float
    counterfactual_batch_loss: float
    batch_loss_change: float
    spillover_sum: float
    spillover_max_abs: float
    anchor_valid: bool
    base_effective_token_count: int
    counterfactual_effective_token_count: int
    base_padded_token_count: int
    counterfactual_padded_token_count: int
    batch_context_sha256: str


class CounterfactualGainEvaluator:
    """One-at-a-time M4 reference evaluator.

    ``token_encoder`` consumes one unpadded [L,D] sequence and returns one feature.
    Running samples independently makes padding observationally irrelevant.
    """
    def __init__(self, model: nn.Module, child_projector: ChildProjector,
                 positions: ParentAwareDetailPositions, head: nn.Module,
                 token_encoder: Callable[[torch.Tensor], torch.Tensor]):
        self.model, self.child_projector, self.positions, self.head = (
            model, child_projector, positions, head
        )
        self.haar = HaarDetails()
        self.token_encoder = token_encoder

    def _tokens(self, images: torch.Tensor, selected: torch.Tensor) -> AssembledTokens:
        flat = images.reshape(-1, *images.shape[2:])
        parents = self.model.pre_transformer_parent_embeddings(flat)
        children = subdivide_parent_patches(extract_parent_patches(flat))
        raw = self.child_projector(children)
        consistent = enforce_parent_consistency(raw, parents)
        if not torch.isfinite(consistent.consistent).all():
            raise FloatingPointError("non-finite consistent child tokens")
        details = self.haar.to(parents)(consistent.consistent)
        parent_pos = self.model.parent_patch_positions().to(parents).expand(flat.shape[0], -1, -1)
        prefix = self.model.prefix_tokens_with_positions(flat.shape[0]).to(parents)
        return assemble_tokens(
            prefix, parents + parent_pos, details + self.positions(parent_pos),
            selected.reshape(-1, 256),
        )

    def _features(self, assembled: AssembledTokens, batch: int, views: int) -> torch.Tensor:
        values = []
        for row in range(batch * views):
            sequence = assembled.tokens[row, assembled.valid_mask[row]]
            feature = self.token_encoder(sequence)
            if feature.ndim != 1:
                raise ValueError("token encoder must return one feature vector")
            values.append(self.head(feature))
        return torch.stack(values).reshape(batch, views, -1)

    @contextlib.contextmanager
    def _paired_mode(self):
        rng = capture_rng_state()
        modes = {module: module.training for module in
                 (self.model, self.child_projector, self.positions, self.head)}
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            for module in modes:
                module.eval()
            for module in self.model.modules():
                if isinstance(module, (nn.Dropout, nn.modules.dropout._DropoutNd)) and module.training:
                    raise RuntimeError("active stochastic module in evaluation")
            torch.use_deterministic_algorithms(True)
            with torch.inference_mode():
                yield
        finally:
            torch.use_deterministic_algorithms(deterministic)
            for module, training in modes.items():
                module.train(training)
            restore_rng_state(rng)

    def evaluate(self, *, images, labels, labelled, hierarchy_labels,
                 confusion_factor, pseudo_label_confidence, context: BatchContext,
                 anchor: int, parent: int, base_cache: dict | None = None) -> GainEvaluation:
        if context.sha256 != context.sha256:  # pragma: no cover - defensive
            raise ValueError("unstable context")
        if not 0 <= anchor < images.shape[0] or not 0 <= parent <= 255:
            raise ValueError("anchor/parent out of range")
        context.validate_tensors(
            images, labels, labelled, hierarchy_labels, confusion_factor,
            pseudo_label_confidence,
        )
        current = (
            module_sha256(self.model), module_sha256(self.child_projector),
            module_sha256(self.positions), module_sha256(self.head),
        )
        expected = (
            context.model_state_hash, context.child_projector_state_hash,
            context.detail_position_state_hash, context.head_state_hash,
        )
        if current != expected or context.model_mode != ("eval" if not self.model.training else "train"):
            raise ValueError("immutable batch context mismatch")
        batch, views = images.shape[:2]
        base_selected = torch.zeros(batch, views, 256, dtype=torch.bool, device=images.device)
        cf_selected = base_selected.clone()
        cf_selected[anchor, :, parent] = True
        with self._paired_mode():
            key = context.sha256
            if base_cache is not None and key in base_cache:
                base_loss, base_assembled = base_cache[key]
            else:
                base_assembled = self._tokens(images, base_selected)
                base_features = self._features(base_assembled, batch, views)
                base_loss = selex_per_anchor(
                    base_features, labels, labelled, hierarchy_labels, confusion_factor,
                    pseudo_label_confidence=pseudo_label_confidence,
                )
                if base_cache is not None:
                    base_cache[key] = (base_loss, base_assembled)
            cf_assembled = self._tokens(images, cf_selected)
            cf_features = self._features(cf_assembled, batch, views)
            cf_loss = selex_per_anchor(
                cf_features, labels, labelled, hierarchy_labels, confusion_factor,
                pseudo_label_confidence=pseudo_label_confidence,
            )
        base = base_loss.total.detach().float()
        counter = cf_loss.total.detach().float()
        valid = bool(base_loss.valid[anchor] and cf_loss.valid[anchor])
        # Invalid labels remain explicit; their finite arithmetic value is diagnostic only.
        gain = float(base[anchor] - counter[anchor])
        delta = counter - base
        non_anchor = torch.cat((delta[:anchor], delta[anchor + 1:]))
        base_scalar = float(base[base_loss.valid].mean())
        cf_scalar = float(counter[cf_loss.valid].mean())
        return GainEvaluation(
            anchor, parent, gain, float(base[anchor]), float(counter[anchor]),
            base_scalar, cf_scalar, cf_scalar - base_scalar,
            float(non_anchor.sum()), float(non_anchor.abs().max()) if non_anchor.numel() else 0.0,
            valid, int(base_assembled.effective_token_count[anchor * views]),
            int(cf_assembled.effective_token_count[anchor * views]),
            base_assembled.padded_token_count, cf_assembled.padded_token_count,
            context.sha256,
        )

    def evaluate_chunk(self, candidates: list[tuple[int, int]], **kwargs) -> list[GainEvaluation]:
        # Deliberately isolated reference semantics; chunking amortizes base reuse only.
        cache: dict = kwargs.pop("base_cache", {})
        return [self.evaluate(anchor=a, parent=p, base_cache=cache, **kwargs) for a, p in candidates]
