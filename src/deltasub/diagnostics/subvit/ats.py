from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .attention import deterministic_topk

PARENT_COUNT = 256


@dataclass
class ATSSequence:
    tokens: torch.Tensor
    positions: torch.Tensor
    valid_mask: torch.Tensor
    token_kind: torch.Tensor  # 0 padding, 1 prefix, 2 parent, 3 direct child
    parent_index: torch.Tensor
    child_index: torch.Tensor
    effective_lengths: torch.Tensor
    selected_indices: tuple[tuple[int, ...], ...]


def direct_child_tokens(children: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Validate direct spatial children in canonical row-major within-parent order."""
    if factor < 1 or children.ndim != 4 or children.shape[1:3] != (PARENT_COUNT, factor * factor):
        raise ValueError("children must be [B,256,f*f,D]")
    if not torch.isfinite(children).all():
        raise ValueError("children must be finite")
    return children


def interpolate_child_positions(parent_positions: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Bicubic interpolation following ViT/DINO positional-resampling convention.

    Parent positions are interpreted as a 16x16 row-major grid and interpolated
    in FP32 to (16*f)x(16*f), then regrouped TL,TR,BL,BR for f=2 (general
    row-major fxf order). ``align_corners=False`` is explicit.
    """
    if parent_positions.ndim != 3 or parent_positions.shape[1] != PARENT_COUNT or factor < 1:
        raise ValueError("parent positions must be [B,256,D] and factor positive")
    if not torch.isfinite(parent_positions).all():
        raise ValueError("parent positions must be finite")
    b, _, d = parent_positions.shape
    grid = parent_positions.float().reshape(b, 16, 16, d).permute(0, 3, 1, 2)
    fine = F.interpolate(grid, size=(16 * factor, 16 * factor), mode="bicubic",
                         align_corners=False).permute(0, 2, 3, 1)
    out = fine.reshape(b, 16, factor, 16, factor, d).permute(0, 1, 3, 2, 4, 5)
    return out.reshape(b, PARENT_COUNT, factor * factor, d)


def assemble_ats(
    prefix_tokens: torch.Tensor, parent_tokens: torch.Tensor, child_tokens: torch.Tensor,
    prefix_positions: torch.Tensor, parent_positions: torch.Tensor,
    child_positions: torch.Tensor, scores: torch.Tensor, k: int | torch.Tensor,
    *, factor: int = 2,
) -> ATSSequence:
    """Retain every original token and append f*f direct children per selected parent."""
    children = direct_child_tokens(child_tokens, factor)
    b, n, d = parent_tokens.shape
    p = prefix_tokens.shape[1]
    expected = (b, PARENT_COUNT, factor * factor, d)
    if n != PARENT_COUNT or prefix_tokens.shape != (b, p, d):
        raise ValueError("token shapes must be [B,P,D] and [B,256,D]")
    if prefix_positions.shape != prefix_tokens.shape or parent_positions.shape != parent_tokens.shape:
        raise ValueError("position shape mismatch")
    if children.shape != expected or child_positions.shape != expected:
        raise ValueError("child shape mismatch")
    for value in (prefix_tokens, parent_tokens, prefix_positions, parent_positions, child_positions):
        if not torch.isfinite(value).all():
            raise ValueError("tokens and positions must be finite")
    selected = deterministic_topk(scores, k)
    lengths = torch.tensor([p + PARENT_COUNT + factor * factor * len(x) for x in selected],
                           dtype=torch.int64, device=parent_tokens.device)
    width = int(lengths.max())
    tokens = parent_tokens.new_zeros((b, width, d)); positions = parent_positions.new_zeros((b, width, d))
    valid = torch.zeros((b, width), dtype=torch.bool, device=parent_tokens.device)
    kind = torch.zeros((b, width), dtype=torch.int8, device=parent_tokens.device)
    parent = torch.full((b, width), -1, dtype=torch.int64, device=parent_tokens.device)
    child = torch.full((b, width), -1, dtype=torch.int8, device=parent_tokens.device)
    for row, indices in enumerate(selected):
        index = torch.tensor(indices, dtype=torch.long, device=parent_tokens.device)
        added = children[row, index].reshape(-1, d)
        added_pos = child_positions[row, index].reshape(-1, d)
        length = int(lengths[row])
        tokens[row, :length] = torch.cat((prefix_tokens[row], parent_tokens[row], added))
        positions[row, :length] = torch.cat((prefix_positions[row], parent_positions[row], added_pos))
        valid[row, :length] = True; kind[row, :p] = 1; kind[row, p:p + PARENT_COUNT] = 2
        kind[row, p + PARENT_COUNT:length] = 3
        parent[row, p:p + PARENT_COUNT] = torch.arange(PARENT_COUNT, device=parent.device)
        if indices:
            parent[row, p + PARENT_COUNT:length] = index.repeat_interleave(factor * factor)
            child[row, p + PARENT_COUNT:length] = torch.arange(
                factor * factor, dtype=torch.int8, device=parent.device).repeat(len(indices))
    return ATSSequence(tokens, positions, valid, kind, parent, child, lengths, selected)
