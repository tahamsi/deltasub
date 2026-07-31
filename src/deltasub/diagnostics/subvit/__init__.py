"""Clean-room, paper-described SubViT diagnostic reference.

This package is deliberately isolated from the M4--M6 gain and adaptive paths.
It is not an official implementation and does not establish paper reproduction.
"""

from .attention import extract_cls_patch_attention, sample_attention_heads
from .ats import ATSSequence, assemble_ats, direct_child_tokens, interpolate_child_positions
from .degradation import DegradationResult, evaluate_head_degradation
from .distillation import DistillationLoss, SubViTRouter, SubViTRouterConfig, distillation_loss

__all__ = [
    "ATSSequence", "DegradationResult", "DistillationLoss", "SubViTRouter",
    "SubViTRouterConfig", "assemble_ats", "direct_child_tokens",
    "distillation_loss", "evaluate_head_degradation",
    "extract_cls_patch_attention", "interpolate_child_positions",
    "sample_attention_heads",
]
