from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import pytest
import torch

from deltasub.models.backbones.dinov2 import (
    DINOV2_REVISION, DINOv2Adapter, construct_official_vitb14,
    inspect_official_checkpoint,
)
from deltasub.models.subtokens.geometry import extract_parent_patches, subdivide_parent_patches
from deltasub.models.subtokens.haar import HaarDetails
from deltasub.models.subtokens.projection import enforce_parent_consistency
from deltasub.utils.hashing import sha256_file


@pytest.mark.integration
class OfficialM3DINOv2Tests(unittest.TestCase):
    def source(self):
        value = os.environ.get("DINOV2_SOURCE_ROOT")
        if not value:
            self.skipTest("DINOV2_SOURCE_ROOT pinned checkout not supplied")
        root = Path(value)
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
            DINOV2_REVISION,
        )
        return root

    def exercise(self, register_tokens: int):
        model, _ = construct_official_vitb14(self.source(), register_tokens=register_tokens)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "complete.pt"
            torch.save(model.state_dict(), checkpoint)
            loaded, report = inspect_official_checkpoint(
                checkpoint, sha256_file(checkpoint), source_root=self.source()
            )
            adapter = DINOv2Adapter(loaded, checkpoint, report)
            projection = adapter.parent_projection
            self.assertEqual(projection.weight.shape, (768, 3, 14, 14))
            before = projection.weight.detach().clone()
            child = adapter.build_child_projector()
            torch.testing.assert_close(projection.weight, before, rtol=0, atol=0)
            image = torch.randn(1, 3, 224, 224)
            parent = adapter.pre_transformer_parent_embeddings(image)
            raw = child(subdivide_parent_patches(extract_parent_patches(image)))
            torch.testing.assert_close(raw.mean(2), parent, rtol=2e-5, atol=3e-6)
            consistent = enforce_parent_consistency(raw, parent)
            details = HaarDetails()(consistent.consistent)
            self.assertEqual(details.shape, (1, 256, 3, 768))
            self.assertEqual(adapter.parent_patch_positions().shape, (1, 256, 768))
            prefix = adapter.prefix_tokens_with_positions(1)
            self.assertEqual(prefix.shape, (1, 1 + register_tokens, 768))
            self.assertEqual(adapter.prefix_token_count, 1 + register_tokens)
            official = adapter.model.prepare_tokens_with_masks(image)
            torch.testing.assert_close(prefix[:, :1], official[:, :1])
            if register_tokens:
                torch.testing.assert_close(prefix[:, 1:], official[:, 1:1 + register_tokens])
            torch.testing.assert_close(projection.weight, before, rtol=0, atol=0)

    def test_generated_complete_official_state_without_registers(self):
        self.exercise(0)

    def test_generated_complete_official_state_with_registers(self):
        self.exercise(4)


if __name__ == "__main__":
    unittest.main()
