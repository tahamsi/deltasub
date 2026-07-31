from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from deltasub.models.backbones.dinov2 import DINOV2_REVISION, construct_official_vitb14
from deltasub.router.model import GainRouter, RouterConfig


@pytest.mark.integration
def test_pinned_official_pretransformer_features_and_router_no_mutation():
    root = os.environ.get("DINOV2_SOURCE_ROOT")
    if not root:
        pytest.skip("DINOV2_SOURCE_ROOT not supplied")
    model, _ = construct_official_vitb14(root)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        parents = model.patch_embed(torch.zeros(1, 3, 224, 224))
        scores = GainRouter(RouterConfig(), seed=5)(parents)
    assert parents.shape == (1, 256, 768)
    assert scores.shape == (1, 256)
    assert torch.isfinite(scores).all()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_m5_cuda_forward_and_loss(precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    router = GainRouter(RouterConfig(), seed=5).cuda()
    parents = torch.randn(1, 256, 768, device="cuda", dtype=dtype)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        scores = router(parents)
        loss = scores.float().square().mean()
    loss.backward()
    assert torch.isfinite(scores).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in router.parameters())
