import torch


def transform_regularization(transform: torch.Tensor) -> torch.Tensor:
    zero_mean = transform.sum(-1).square().mean()
    identity = torch.eye(transform.shape[0], device=transform.device, dtype=transform.dtype)
    orthogonal = (transform @ transform.T - identity).square().mean()
    return zero_mean + orthogonal
