from __future__ import annotations

import hashlib
import torch

PARENT_COUNT = 256


def _check_finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def extract_cls_patch_attention(
    attention: torch.Tensor, *, prefix_tokens: int, parent_count: int = PARENT_COUNT,
    key_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract per-head CLS-query attention to row-major parent keys.

    ``attention`` is the post-softmax official block attention [B,H,N,N].
    Prefix tokens (CLS and registers) precede parents. Any trailing mask/padding
    keys are excluded. Values are returned in FP32 without averaging heads.
    """
    if attention.ndim != 4 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError("attention must have shape [B,H,N,N]")
    if prefix_tokens < 1 or prefix_tokens + parent_count > attention.shape[-1]:
        raise ValueError("invalid prefix or parent-token count")
    _check_finite(attention, "attention")
    result = attention[:, :, 0, prefix_tokens:prefix_tokens + parent_count].float()
    if key_valid_mask is not None:
        if key_valid_mask.shape != (attention.shape[0], attention.shape[-1]):
            raise ValueError("key_valid_mask shape mismatch")
        parent_valid = key_valid_mask[:, prefix_tokens:prefix_tokens + parent_count]
        if not bool(parent_valid.all()):
            raise ValueError("all 256 parent patch keys must be valid")
    if result.shape[-1] != parent_count:
        raise ValueError("parent attention shape mismatch")
    return result


def deterministic_topk(scores: torch.Tensor, k: int | torch.Tensor) -> tuple[tuple[int, ...], ...]:
    if scores.ndim != 2 or scores.shape[1] != PARENT_COUNT:
        raise ValueError("scores must be [B,256]")
    _check_finite(scores, "scores")
    ks = ([int(k)] * scores.shape[0] if isinstance(k, int)
          else [int(x) for x in k.detach().cpu().tolist()])
    if len(ks) != scores.shape[0] or any(x < 0 or x > PARENT_COUNT for x in ks):
        raise ValueError("K must provide one integer in [0,256] per sample")
    cpu = scores.detach().float().cpu()
    return tuple(tuple(sorted(range(PARENT_COUNT), key=lambda j: (-float(cpu[i, j]), j))[:ks[i]])
                 for i in range(scores.shape[0]))


def sample_attention_heads(head_count: int, count: int, *, seed: int, step: int = 0) -> tuple[int, ...]:
    """Seeded backend-independent head schedule (sampling without replacement)."""
    if head_count < 1 or count < 0 or count > head_count or step < 0:
        raise ValueError("invalid head schedule arguments")
    ranked = sorted(range(head_count), key=lambda h: (
        hashlib.sha256(f"{seed}:{step}:{h}".encode()).digest(), h))
    return tuple(ranked[:count])


@torch.no_grad()
def extract_dinov2_attention(
    model: torch.nn.Module, images: torch.Tensor, *, layer: int,
    parent_count: int = PARENT_COUNT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract pinned official DINOv2 per-head maps and pre-transformer parents.

    This uses the official model's token preparation and block modules without
    patching or mutating them. Attention probabilities are recomputed from the
    target block's normalized qkv input using the block's own scale.
    """
    if images.ndim != 4 or not torch.isfinite(images).all():
        raise ValueError("images must be finite [B,C,H,W]")
    if not hasattr(model, "prepare_tokens_with_masks") or not hasattr(model, "blocks"):
        raise ValueError("model is not a supported official DINOv2 ViT")
    blocks = model.blocks
    if not 0 <= layer < len(blocks):
        raise ValueError("attention layer out of range")
    was_training = model.training
    model.eval()
    try:
        tokens = model.prepare_tokens_with_masks(images, None)
        registers = int(getattr(model, "num_register_tokens", 0))
        prefix = 1 + registers
        if tokens.shape[1] != prefix + parent_count:
            raise ValueError("expected exactly CLS/register prefix plus 256 parents")
        parents = tokens[:, prefix:prefix + parent_count].detach()
        x = tokens
        for index in range(layer):
            x = blocks[index](x)
        block = blocks[layer]
        normalized = block.norm1(x)
        qkv_module = block.attn.qkv
        heads = int(block.attn.num_heads)
        b, n, d = normalized.shape
        qkv = qkv_module(normalized).reshape(b, n, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
        q, key = qkv[0].float(), qkv[1].float()
        probabilities = (q * float(block.attn.scale)) @ key.transpose(-2, -1)
        probabilities = probabilities.softmax(dim=-1, dtype=torch.float32)
        return extract_cls_patch_attention(probabilities, prefix_tokens=prefix,
                                           parent_count=parent_count), parents
    finally:
        model.train(was_training)
