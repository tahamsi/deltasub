from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class SelectionResult:
    selected_indices: tuple[tuple[int, ...], ...]
    selected_mask: torch.Tensor
    rank_positions: torch.Tensor
    canonical_rank_order: tuple[tuple[int, ...], ...]
    realized_k: torch.Tensor


@torch.no_grad()
def deterministic_select(
    scores: torch.Tensor, k: torch.Tensor | list[int] | tuple[int, ...],
    valid_parent_mask: torch.Tensor | None = None,
) -> SelectionResult:
    if scores.ndim != 2 or scores.shape[1] != 256 or not scores.is_floating_point():
        raise ValueError("scores must be floating point [B, 256]")
    if not torch.isfinite(scores).all():
        raise ValueError("router scores must be finite")
    batch = scores.shape[0]
    kvals = torch.as_tensor(k, dtype=torch.int64, device=scores.device)
    if kvals.shape != (batch,):
        raise ValueError("K must have shape [B]")
    if valid_parent_mask is None:
        valid_parent_mask = torch.ones_like(scores, dtype=torch.bool)
    if valid_parent_mask.shape != scores.shape or valid_parent_mask.dtype != torch.bool:
        raise ValueError("valid_parent_mask must be bool [B, 256]")
    available = valid_parent_mask.sum(1)
    if bool(((kvals < 0) | (kvals > 256) | (kvals > available)).any()):
        raise ValueError("K is outside valid-parent bounds")
    mask = torch.zeros_like(valid_parent_mask)
    ranks = torch.full_like(kvals[:, None].expand(-1, 256), -1)
    selected: list[tuple[int, ...]] = []
    orders: list[tuple[int, ...]] = []
    # Python's tuple ordering is an explicit backend-independent lexicographic policy.
    values = scores.detach().float().cpu()
    valid = valid_parent_mask.cpu()
    for row in range(batch):
        order = tuple(sorted(
            (j for j in range(256) if bool(valid[row, j])),
            key=lambda j: (-float(values[row, j]), j),
        ))
        chosen = order[:int(kvals[row].item())]
        selected.append(chosen); orders.append(order)
        if chosen:
            idx = torch.tensor(chosen, device=scores.device)
            mask[row, idx] = True
        for position, index in enumerate(order):
            ranks[row, index] = position
    return SelectionResult(tuple(selected), mask, ranks, tuple(orders), kvals.clone())
