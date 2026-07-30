from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch

from .assembly import AdaptiveSequence


@dataclass
class ExecutionResult:
    outputs: torch.Tensor
    valid_mask: torch.Tensor
    effective_tokens: int
    padded_tokens: int
    padding_tokens: int
    padding_fraction: float
    bucket_ids: tuple[str, ...]
    bucket_composition: dict[str, tuple[int, ...]]


def _call(model: Callable, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    output = model(tokens, valid)
    if output.shape != tokens.shape:
        raise ValueError("adaptive transformer must return [B,L,D]")
    if not torch.isfinite(output[valid]).all():
        raise ValueError("non-finite valid transformer output")
    return output


def _bucket_id(length: int, boundaries: tuple[int, ...] | None) -> str:
    if boundaries is None:
        return f"length_{length}"
    for boundary in boundaries:
        if length <= boundary:
            return f"le_{boundary}"
    return f"gt_{boundaries[-1]}"


def execute_adaptive(
    model: Callable, sequence: AdaptiveSequence, *, mode: str,
    bucket_boundaries: tuple[int, ...] | None = None,
) -> ExecutionResult:
    if mode not in {"padded", "bucketed"}:
        raise ValueError("execution mode must be padded or bucketed")
    lengths = [int(x) for x in sequence.effective_lengths.tolist()]
    composition: dict[str, list[int]] = {}
    ids = tuple(_bucket_id(length, None if mode == "padded" else bucket_boundaries)
                for length in lengths)
    if mode == "padded":
        composition["padded"] = list(range(len(lengths)))
        output = _call(model, sequence.tokens, sequence.valid_mask)
        padded = sequence.tokens.shape[0] * sequence.tokens.shape[1]
    else:
        for row, bucket in enumerate(ids):
            composition.setdefault(bucket, []).append(row)
        output = torch.zeros_like(sequence.tokens)
        padded = 0
        for bucket in sorted(composition, key=lambda key: (max(
                lengths[i] for i in composition[key]), key)):
            rows = composition[bucket]
            width = max(lengths[i] for i in rows)
            indices = torch.tensor(rows, device=sequence.tokens.device)
            values = sequence.tokens.index_select(0, indices)[:, :width]
            valid = sequence.valid_mask.index_select(0, indices)[:, :width]
            result = _call(model, values, valid)
            output[indices, :width] = result
            padded += len(rows) * width
    effective = sum(lengths)
    padding = padded - effective
    return ExecutionResult(output, sequence.valid_mask, effective, padded, padding,
                           padding / padded if padded else 0.0, ids,
                           {k: tuple(v) for k, v in composition.items()})


class MaskedSelfAttentionBlock(torch.nn.Module):
    """Small DINOv2-compatible fixture block; mask semantics match key padding."""
    def __init__(self, dimension: int, heads: int = 2):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dimension)
        self.attention = torch.nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.ffn = torch.nn.Sequential(torch.nn.LayerNorm(dimension),
                                       torch.nn.Linear(dimension, 2 * dimension),
                                       torch.nn.GELU(), torch.nn.Linear(2 * dimension, dimension))

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     key_padding_mask=~valid_mask, need_weights=False)
        result = tokens + attended
        result = result + self.ffn(result)
        return result.masked_fill(~valid_mask.unsqueeze(-1), 0)
