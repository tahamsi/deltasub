from pathlib import Path
import pytest, torch
from deltasub.models.backbones.dinov2 import DINOV2_REVISION, construct_official_vitb14
from deltasub.baselines.fixture import synthetic_input
from deltasub.baselines.registry import build_registry

SOURCE=Path("/home/ubuntu/references/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8")

def test_generated_complete_state_fixture_and_no_mutation():
    adapter=build_registry()["vit_dinov2_selex"]; x=synthetic_input(dimension=8)
    before=x.parent_tokens.clone(); result=adapter.run(x)
    assert result.cls_features.shape==(3,8) and torch.equal(before,x.parent_tokens)

@pytest.mark.integration
def test_pinned_official_dinov2_architecture_if_available():
    if not SOURCE.is_dir(): pytest.skip("pinned DINOv2 checkout absent")
    model, hashes=construct_official_vitb14(SOURCE)
    assert len(model.blocks)==12 and model.embed_dim==768 and len(hashes)==3

@pytest.mark.gpu
@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_cuda_fixture_precision(dtype):
    if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    precision="bf16" if dtype==torch.bfloat16 else "fp32"
    x=synthetic_input(batch_size=1,dimension=16,device="cuda:0",precision=precision)
    y=build_registry()["vit_dinov2_selex"].run(x)
    assert y.cls_features.dtype == dtype
    assert torch.isfinite(y.cls_features).all()
    assert torch.isfinite(y.register_features).all()
    assert y.parent_spatial_features is not None
    assert torch.isfinite(y.parent_spatial_features).all()
