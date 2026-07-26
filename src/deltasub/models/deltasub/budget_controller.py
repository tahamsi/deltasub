from __future__ import annotations

import torch


K_BINS = (0, 1, 2, 4, 8, 16, 32)


def bucket_k(k: torch.Tensor) -> torch.Tensor:
    bins = torch.tensor(K_BINS, device=k.device)
    distances = (k.unsqueeze(-1) - bins).abs()
    return bins[distances.argmin(-1)]


class DualBudgetController:
    def __init__(self, target_tokens: float, eta: float = 1e-3, value: float = 0.0):
        self.target_tokens, self.eta, self.value = target_tokens, eta, value

    def update(self, observed_tokens: float) -> float:
        self.value = max(0.0, self.value + self.eta * (observed_tokens - self.target_tokens))
        return self.value


def select_adaptive(scores: torch.Tensor, penalty: float, maximum_k: int = 32) -> list[torch.Tensor]:
    utility = scores - penalty * 3
    result = []
    for row in utility:
        positive = torch.nonzero(row > 0, as_tuple=False).flatten()
        order = positive[torch.argsort(row[positive], descending=True)][:maximum_k]
        result.append(order)
    return result
