from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CandidateBatch:
    indices: torch.Tensor
    sources: list[list[str]]
    exploration_probability: float


def sample_candidates(
    router_scores: torch.Tensor,
    detail_scores: torch.Tensor,
    total: int,
    random_count: int,
    generator: torch.Generator | None = None,
) -> CandidateBatch:
    b, n = router_scores.shape
    if not 0 < random_count <= total <= n:
        raise ValueError("require 0 < random_count <= total <= patch count")
    all_indices, all_sources = [], []
    for row in range(b):
        random_idx = torch.randperm(n, generator=generator, device="cpu")[:random_count].to(
            router_scores.device
        )
        selected = random_idx.tolist()
        sources = ["exploration"] * random_count
        proposals = [
            ("router", router_scores[row]),
            ("detail", detail_scores[row]),
            ("disagreement", (router_scores[row].argsort().argsort() - detail_scores[row].argsort().argsort()).abs().float()),
        ]
        for source, scores in proposals:
            for index in torch.argsort(scores, descending=True).tolist():
                if index not in selected:
                    selected.append(index)
                    sources.append(source)
                    break
                if len(selected) >= total:
                    break
            if len(selected) >= total:
                break
        while len(selected) < total:
            index = next(i for i in range(n) if i not in selected)
            selected.append(index)
            sources.append("fallback")
        all_indices.append(selected)
        all_sources.append(sources)
    return CandidateBatch(
        indices=torch.tensor(all_indices, device=router_scores.device),
        sources=all_sources,
        exploration_probability=random_count / n,
    )
