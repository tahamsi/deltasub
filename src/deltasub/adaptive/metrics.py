from __future__ import annotations

import torch


def maximum_valid_output_error(a: torch.Tensor, b: torch.Tensor,
                               valid_mask: torch.Tensor) -> float:
    if a.shape != b.shape or valid_mask.shape != a.shape[:2]:
        raise ValueError("output/mask shape mismatch")
    difference = (a[valid_mask].float() - b[valid_mask].float()).abs().detach()
    return float(difference.max()) if valid_mask.any() else 0.0


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    return {"loss": float(loss.detach()), "accuracy": float((logits.argmax(1) == labels).float().mean()),
            "finite_logits": True, "finite_loss": bool(torch.isfinite(loss))}
