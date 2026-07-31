"""Stable padded assembly of prefix, parent, and externally selected detail tokens."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import torch


class TokenKind(IntEnum):
    PADDING = -1
    PREFIX = 0
    PARENT = 1
    DETAIL = 2


@dataclass
class AssembledTokens:
    tokens: torch.Tensor
    valid_mask: torch.Tensor
    token_kind: torch.Tensor
    parent_index: torch.Tensor
    detail_mode: torch.Tensor
    effective_token_count: torch.Tensor
    padded_token_count: int
    prefix_token_count: int
    parent_token_count: int
    selected_parent_count: torch.Tensor

    def unpadded(self, sample: int) -> dict[str, torch.Tensor]:
        valid = self.valid_mask[sample]
        return {name: getattr(self, name)[sample, valid] for name in
                ("tokens", "token_kind", "parent_index", "detail_mode")}


def assemble_tokens(prefix: torch.Tensor, parents: torch.Tensor, details: torch.Tensor,
                    selected: torch.Tensor, *, padding_value: float = 0.0) -> AssembledTokens:
    if prefix.ndim != 3 or parents.ndim != 3 or details.ndim != 4:
        raise ValueError("tokens must be [B,P,D], [B,256,D], and [B,256,3,D]")
    b, n, d = parents.shape
    if n != 256 or details.shape != (b, 256, 3, d) or prefix.shape[0] != b or prefix.shape[2] != d:
        raise ValueError("incompatible token shapes")
    if selected.dtype != torch.bool or selected.shape != (b, 256):
        raise ValueError("selected-parent mask must be bool [B, 256]")
    counts = selected.sum(1)
    effective = prefix.shape[1] + 256 + 3 * counts
    padded_count = int(effective.max().item())
    tokens = parents.new_full((b, padded_count, d), padding_value)
    valid = torch.zeros((b, padded_count), dtype=torch.bool, device=parents.device)
    kind = torch.full((b, padded_count), TokenKind.PADDING, dtype=torch.int8, device=parents.device)
    parent_index = torch.full((b, padded_count), -1, dtype=torch.int64, device=parents.device)
    mode = torch.full((b, padded_count), -1, dtype=torch.int8, device=parents.device)
    base = prefix.shape[1] + 256
    for row in range(b):
        k = int(counts[row])
        chosen = details[row, selected[row]].reshape(3 * k, d)
        sequence = torch.cat((prefix[row], parents[row], chosen))
        length = sequence.shape[0]
        tokens[row, :length] = sequence
        valid[row, :length] = True
        kind[row, :prefix.shape[1]] = TokenKind.PREFIX
        kind[row, prefix.shape[1]:base] = TokenKind.PARENT
        kind[row, base:length] = TokenKind.DETAIL
        parent_index[row, prefix.shape[1]:base] = torch.arange(256, device=parents.device)
        indices = torch.nonzero(selected[row], as_tuple=False).flatten()
        parent_index[row, base:length] = indices.repeat_interleave(3)
        mode[row, base:length] = torch.arange(3, device=parents.device, dtype=torch.int8).repeat(k)
    return AssembledTokens(tokens, valid, kind, parent_index, mode, effective, padded_count,
                           prefix.shape[1], 256, counts)
