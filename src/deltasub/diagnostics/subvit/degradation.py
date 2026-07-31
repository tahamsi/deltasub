from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch

from .attention import deterministic_topk
from ...utils.hashing import stable_hash


@dataclass(frozen=True)
class DegradationResult:
    distances: tuple[tuple[float, ...], ...]
    selected_heads: tuple[int, ...]
    selected_masks: torch.Tensor
    original_feature_hash: str
    degraded_feature_hashes: tuple[tuple[str, ...], ...]


def evaluate_head_degradation(
    parent_tokens: torch.Tensor, attention_maps: torch.Tensor, k: int | torch.Tensor,
    teacher: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], *, chunk_size: int | None = None,
) -> DegradationResult:
    """Mask only selected parents and select maximum FP32 CLS-feature degradation.

    ``teacher(tokens, valid_parent_mask)`` must preserve its own prefix/register
    ordering. The original call receives an all-true mask. Chunking only batches
    otherwise identical degraded examples.
    """
    if parent_tokens.ndim != 3 or parent_tokens.shape[1] != 256:
        raise ValueError("parent_tokens must be [B,256,D]")
    if attention_maps.ndim != 3 or attention_maps.shape[0] != parent_tokens.shape[0] or attention_maps.shape[2] != 256:
        raise ValueError("attention_maps must be [B,H,256]")
    if not torch.isfinite(parent_tokens).all() or not torch.isfinite(attention_maps).all():
        raise ValueError("inputs must be finite")
    b, h, _ = attention_maps.shape
    ks = ([int(k)] * b if isinstance(k, int) else [int(x) for x in k.cpu().tolist()])
    masks = torch.ones((b, h, 256), dtype=torch.bool, device=parent_tokens.device)
    for head in range(h):
        for row, chosen in enumerate(deterministic_topk(attention_maps[:, head], torch.tensor(ks))):
            masks[row, head, list(chosen)] = False
    original = teacher(parent_tokens, torch.ones((b, 256), dtype=torch.bool, device=parent_tokens.device))
    if original.ndim != 2 or original.shape[0] != b or not torch.isfinite(original).all():
        raise ValueError("teacher must return finite [B,D] CLS features")
    items = [(row, head) for row in range(b) for head in range(h)]
    size = len(items) if chunk_size is None else int(chunk_size)
    if size < 1:
        raise ValueError("chunk_size must be positive")
    values: list[list[torch.Tensor | None]] = [[None] * h for _ in range(b)]
    for start in range(0, len(items), size):
        chunk = items[start:start + size]
        tokens = torch.stack([parent_tokens[r] for r, _ in chunk])
        valid = torch.stack([masks[r, q] for r, q in chunk])
        degraded = teacher(tokens, valid)
        if degraded.ndim != 2 or degraded.shape[0] != len(chunk) or not torch.isfinite(degraded).all():
            raise ValueError("degraded teacher output invalid")
        for offset, (row, head) in enumerate(chunk):
            values[row][head] = degraded[offset].detach()
    distances = []; hashes = []; selected = []
    for row in range(b):
        ds = [float(torch.linalg.vector_norm(original[row].float() - x.float()).item()) for x in values[row]]
        selected.append(min(range(h), key=lambda q: (-ds[q], q)))
        distances.append(tuple(ds))
        hashes.append(tuple(stable_hash(x.float().cpu().tolist()) for x in values[row]))
    return DegradationResult(tuple(distances), tuple(selected), masks,
                             stable_hash(original.float().cpu().tolist()), tuple(hashes))
