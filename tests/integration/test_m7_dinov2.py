from __future__ import annotations

import os
from pathlib import Path
import unittest
import torch

from deltasub.models.backbones.dinov2 import construct_official_vitb14
from deltasub.diagnostics.subvit.attention import extract_dinov2_attention


@unittest.skipUnless(os.environ.get("DINOV2_SOURCE_ROOT"), "DINOV2_SOURCE_ROOT not supplied")
class M7DINOv2Integration(unittest.TestCase):
    def test_official_attention_fp32_and_optional_cuda_bf16(self):
        source = Path(os.environ["DINOV2_SOURCE_ROOT"])
        model, _ = construct_official_vitb14(source)
        maps, parents = extract_dinov2_attention(model, torch.randn(1, 3, 224, 224), layer=0)
        self.assertEqual(maps.shape, (1, 12, 256))
        self.assertEqual(parents.shape, (1, 256, 768))
        self.assertEqual(maps.dtype, torch.float32)
        if torch.cuda.is_available():
            model = model.cuda()
            for dtype in (torch.float32, torch.bfloat16):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                    maps, _ = extract_dinov2_attention(
                        model, torch.randn(1, 3, 224, 224, device="cuda", dtype=dtype), layer=0)
                self.assertEqual(maps.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
