"""Real-checkpoint M2 regression coverage; never downloads model data."""
from contextlib import nullcontext
import os
from pathlib import Path

import pytest
import torch

from deltasub.baselines.adapters._common import state_hash
from deltasub.models.backbones.dinov2 import DINOv2Adapter, inspect_official_checkpoint


CHECKPOINT = Path("/home/ubuntu/checkpoints/dinov2_vitb14_pretrain.pth")
SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"


def _source() -> Path:
    value = os.environ.get("DINOV2_SOURCE_ROOT")
    if not value or not CHECKPOINT.is_file():
        pytest.skip("pinned source and real official checkpoint are required")
    return Path(value)


@pytest.mark.integration
def test_real_native_checkpoint_and_runtime_interpolation():
    model, report = inspect_official_checkpoint(CHECKPOINT, SHA256, source_root=_source())
    assert model.pos_embed.shape == (1, 1370, 768)
    assert report.compatible and report.strict_checkpoint_load
    assert report.native_checkpoint_image_size == 518
    assert report.native_checkpoint_grid_size == (37, 37)
    assert report.runtime_image_size == 224
    assert report.runtime_grid_size == (16, 16)
    assert report.positional_interpolation_required
    assert report.positional_interpolation_verified


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_real_checkpoint_cuda_224_is_finite_and_does_not_mutate(precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    adapter = DINOv2Adapter.from_official_checkpoint(
        CHECKPOINT, SHA256, source_root=_source()
    ).to("cuda:0").eval()
    before = state_hash(adapter)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16" else nullcontext()
    )
    with torch.inference_mode(), context:
        output = adapter(torch.zeros(1, 3, 224, 224, device="cuda:0"))
    assert output.patch_tokens.shape == (1, 256, 768)
    assert torch.isfinite(output.patch_tokens).all()
    assert torch.isfinite(output.cls_token).all()
    assert state_hash(adapter) == before
