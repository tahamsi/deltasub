import torch
from torch.nn import functional as F


def parent_consistency(child_mean: torch.Tensor, parent: torch.Tensor, adapter=None) -> torch.Tensor:
    value = adapter(child_mean) if adapter is not None else child_mean
    return F.mse_loss(value, parent)
