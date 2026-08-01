import os
from pathlib import Path

import pytest
import torch

from deltasub.diagnostic.campaign import preflight


@pytest.mark.integration
@pytest.mark.parametrize("dataset", ["cub", "aircraft"])
def test_real_asset_preflight_without_training(dataset):
    required = [Path(f"/home/ubuntu/datasets/{dataset}/manifest.jsonl"),
                Path("/home/ubuntu/checkpoints/dinov2_vitb14_pretrain.pth"),
                Path("/home/ubuntu/references/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8")]
    if not all(path.exists() for path in required): pytest.skip("provided M9 real assets absent")
    if not torch.cuda.is_available(): pytest.skip("M9 production preflight requires CUDA")
    report = preflight(f"configs/diagnostic/{dataset}_seed0.yaml")
    assert report["status"] == "passed"
    assert report["training_started"] is False
    assert report["seed"] == 0
