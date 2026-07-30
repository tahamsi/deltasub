from __future__ import annotations

from dataclasses import dataclass
import torch

from ..models.subtokens.assembly import TokenKind
from ..utils.hashing import stable_hash
from .selection import SelectionResult


@dataclass
class AdaptiveSequence:
    tokens: torch.Tensor
    positions: torch.Tensor
    token_kind: torch.Tensor
    parent_index: torch.Tensor
    detail_mode: torch.Tensor
    valid_mask: torch.Tensor
    effective_lengths: torch.Tensor
    selection_plan_hash: str

    def unpadded(self, row: int) -> dict[str, torch.Tensor]:
        valid = self.valid_mask[row]
        return {key: getattr(self, key)[row, valid] for key in
                ("tokens", "positions", "token_kind", "parent_index", "detail_mode")}


def assemble_adaptive(
    prefix_tokens: torch.Tensor, parent_tokens: torch.Tensor, detail_tokens: torch.Tensor,
    prefix_positions: torch.Tensor, parent_positions: torch.Tensor,
    detail_positions: torch.Tensor, selection: SelectionResult,
    validity_mask: torch.Tensor | None = None,
) -> AdaptiveSequence:
    b, n, d = parent_tokens.shape
    p = prefix_tokens.shape[1]
    expected = (b, 256, 3, d)
    if n != 256 or prefix_tokens.shape != (b, p, d) or detail_tokens.shape != expected:
        raise ValueError("token shapes must be [B,P,D], [B,256,D], [B,256,3,D]")
    if prefix_positions.shape != prefix_tokens.shape or parent_positions.shape != parent_tokens.shape:
        raise ValueError("prefix/parent position shapes must match token shapes")
    if detail_positions.shape != expected:
        raise ValueError("detail position shape mismatch")
    if selection.selected_mask.shape != (b, 256):
        raise ValueError("selection batch mismatch")
    if validity_mask is not None and (
        validity_mask.shape != (b, 256) or
        bool((selection.selected_mask & ~validity_mask).any())
    ):
        raise ValueError("selection contains an invalid parent")
    lengths = p + 256 + 3 * selection.realized_k
    padded = int(lengths.max())
    tokens = parent_tokens.new_zeros((b, padded, d))
    positions = parent_positions.new_zeros((b, padded, d))
    valid = torch.zeros((b, padded), dtype=torch.bool, device=parent_tokens.device)
    kind = torch.full((b, padded), TokenKind.PADDING, dtype=torch.int8, device=parent_tokens.device)
    parents = torch.full((b, padded), -1, dtype=torch.int64, device=parent_tokens.device)
    modes = torch.full((b, padded), -1, dtype=torch.int8, device=parent_tokens.device)
    for row, indices in enumerate(selection.selected_indices):
        index = torch.tensor(indices, dtype=torch.int64, device=parent_tokens.device)
        selected_tokens = detail_tokens[row, index].reshape(-1, d)
        selected_positions = detail_positions[row, index].reshape(-1, d)
        length = int(lengths[row])
        tokens[row, :length] = torch.cat((prefix_tokens[row], parent_tokens[row], selected_tokens))
        positions[row, :length] = torch.cat((prefix_positions[row], parent_positions[row],
                                             selected_positions))
        valid[row, :length] = True
        kind[row, :p] = TokenKind.PREFIX
        kind[row, p:p + 256] = TokenKind.PARENT
        kind[row, p + 256:length] = TokenKind.DETAIL
        parents[row, p:p + 256] = torch.arange(256, device=parent_tokens.device)
        if indices:
            parents[row, p + 256:length] = index.repeat_interleave(3)
            modes[row, p + 256:length] = torch.arange(
                3, dtype=torch.int8, device=parent_tokens.device).repeat(len(indices))
    plan_hash = stable_hash({"indices": selection.selected_indices, "lengths": lengths.tolist(),
                             "prefix_tokens": p, "parent_tokens": 256})
    return AdaptiveSequence(tokens, positions, kind, parents, modes, valid, lengths, plan_hash)
