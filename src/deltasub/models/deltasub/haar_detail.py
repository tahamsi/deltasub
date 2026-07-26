from __future__ import annotations

import torch


def haar_matrix(*, device=None, dtype=None) -> torch.Tensor:
    return 0.5 * torch.tensor(
        [[1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
        device=device,
        dtype=dtype,
    )


def haar_details(children: torch.Tensor) -> torch.Tensor:
    if children.shape[-2] != 4:
        raise ValueError("children must have four vectors on the penultimate axis")
    q = haar_matrix(device=children.device, dtype=children.dtype)
    return torch.einsum("oc,...cd->...od", q, children)


def reconstruct_children(mean: torch.Tensor, details: torch.Tensor) -> torch.Tensor:
    q = haar_matrix(device=details.device, dtype=details.dtype)
    return mean.unsqueeze(-2) + torch.einsum("oc,...od->...cd", q, details)
