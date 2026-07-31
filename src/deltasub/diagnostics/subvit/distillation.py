from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F

from ...utils.hashing import stable_hash


@dataclass(frozen=True)
class SubViTRouterConfig:
    input_dim: int
    hidden_dim: int = 128
    temperature: float = 1.0
    test_only: bool = False

    def __post_init__(self):
        if self.input_dim < 1 or self.hidden_dim < 1 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("invalid router configuration")


class SubViTRouter(nn.Module):
    """Single-map router using only pre-transformer parent embeddings."""
    def __init__(self, config: SubViTRouterConfig, *, seed: int = 0):
        super().__init__()
        if config.input_dim < 32 and not config.test_only:
            raise ValueError("small router dimensions require explicit test_only")
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(nn.Linear(config.input_dim, config.hidden_dim),
                                         nn.GELU(), nn.Linear(config.hidden_dim, 1))

    @property
    def configuration_hash(self):
        return stable_hash(asdict(self.config))

    def forward(self, parents: torch.Tensor) -> torch.Tensor:
        if parents.ndim != 3 or parents.shape[1:] != (256, self.config.input_dim):
            raise ValueError("router input must be [B,256,input_dim]")
        if not torch.isfinite(parents).all():
            raise ValueError("router input must be finite")
        return self.network(parents).squeeze(-1)


@dataclass
class DistillationLoss:
    total: torch.Tensor
    map_kl: torch.Tensor
    ranking: torch.Tensor
    topk_mask: torch.Tensor
    pair_count: int


def distillation_loss(
    scores: torch.Tensor, teacher_map: torch.Tensor, k: int, *, temperature: float = 1.,
    map_weight: float = 1., ranking_weight: float = 1., topk_weight: float = 1.,
) -> DistillationLoss:
    """Mean-over-batch objective with explicit per-example denominators.

    KL uses normalized softmax maps and the standard T^2 multiplier. Ranking
    averages all strict teacher-order pairs (equal targets make no pair);
    examples with no pairs contribute finite zero. Top-K BCE averages all 256
    entries; teacher ties use ascending parent index.
    """
    if scores.shape != teacher_map.shape or scores.ndim != 2 or scores.shape[1] != 256:
        raise ValueError("scores and teacher_map must be [B,256]")
    if not torch.isfinite(scores).all() or not torch.isfinite(teacher_map).all():
        raise ValueError("loss inputs must be finite")
    if not 0 <= k <= 256 or temperature <= 0:
        raise ValueError("invalid K or temperature")
    log_p = F.log_softmax(scores.float() / temperature, dim=1)
    q = F.softmax(teacher_map.float() / temperature, dim=1)
    kl = F.kl_div(log_p, q, reduction="batchmean") * temperature ** 2
    terms = []
    for row in range(scores.shape[0]):
        hi, lo = torch.where(teacher_map[row, :, None] > teacher_map[row, None, :])
        terms.append(F.softplus(-(scores[row, hi] - scores[row, lo])).mean()
                     if hi.numel() else scores[row].sum() * 0)
    ranking = torch.stack(terms).mean()
    target = torch.zeros_like(scores, dtype=torch.float32)
    order = [sorted(range(256), key=lambda j: (-float(teacher_map[i, j].detach().float().cpu()), j))[:k]
             for i in range(scores.shape[0])]
    for row, indices in enumerate(order):
        target[row, indices] = 1
    mask = F.binary_cross_entropy_with_logits(scores.float(), target, reduction="mean")
    total = map_weight * kl + ranking_weight * ranking + topk_weight * mask
    return DistillationLoss(total, kl, ranking, mask, sum(
        int((teacher_map[r, :, None] > teacher_map[r, None, :]).sum()) for r in range(scores.shape[0])))
