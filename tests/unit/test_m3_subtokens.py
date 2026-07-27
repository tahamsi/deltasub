from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest
import torch
from torch import nn

from deltasub.models.subtokens.assembly import TokenKind, assemble_tokens
from deltasub.models.subtokens.diagnostic import load_subtoken_config, run_fixture_diagnostic
from deltasub.models.subtokens.geometry import (
    extract_parent_patches, reconstruct_images, reconstruct_parent_patches,
    subdivide_parent_patches,
)
from deltasub.models.subtokens.haar import HaarDetails
from deltasub.models.subtokens.positions import ParentAwareDetailPositions
from deltasub.models.subtokens.projection import ChildProjector, enforce_parent_consistency


class GeometryTests(unittest.TestCase):
    def test_exact_parent_and_image_geometry(self):
        image = torch.arange(3 * 224 * 224).reshape(1, 3, 224, 224)
        parents = extract_parent_patches(image)
        self.assertEqual(parents.shape, (1, 256, 3, 14, 14))
        torch.testing.assert_close(parents[0, 17], image[0, :, 14:28, 14:28])
        torch.testing.assert_close(reconstruct_images(parents), image)
        # Every coordinate appears exactly once.
        self.assertEqual(torch.unique(parents[0, :, 0]).numel(), 224 * 224)

    def test_direct_child_order_and_inverse(self):
        parent = torch.arange(14 * 14).reshape(1, 1, 1, 14, 14).expand(1, 256, 3, 14, 14)
        children = subdivide_parent_patches(parent)
        expected = (parent[0, 0, :, :7, :7], parent[0, 0, :, :7, 7:],
                    parent[0, 0, :, 7:, :7], parent[0, 0, :, 7:, 7:])
        for q in range(4):
            torch.testing.assert_close(children[0, 0, q], expected[q])
        torch.testing.assert_close(reconstruct_parent_patches(children), parent)

    def test_noncontiguous_and_malformed(self):
        wide = torch.randn(1, 256, 3, 14, 28)
        parents = wide[..., ::2]
        self.assertFalse(parents.is_contiguous())
        torch.testing.assert_close(reconstruct_parent_patches(subdivide_parent_patches(parents)), parents)
        for bad in (torch.randn(1, 3, 223, 224), torch.randn(3, 224, 224)):
            with self.assertRaisesRegex(ValueError, "shape"):
                extract_parent_patches(bad)
        with self.assertRaisesRegex(ValueError, "shape"):
            subdivide_parent_patches(torch.randn(1, 255, 3, 14, 14))


class ProjectionConsistencyTests(unittest.TestCase):
    def make(self, dim=8, dtype=torch.float32):
        torch.manual_seed(4)
        return nn.Conv2d(3, dim, 14, 14, dtype=dtype)

    def test_quadrants_bias_initial_mean_and_no_mutation(self):
        parent = self.make()
        before_w, before_b = parent.weight.detach().clone(), parent.bias.detach().clone()
        child = ChildProjector(parent, provenance="TEST-ONLY", trainable=True)
        slices = ((slice(0, 7), slice(0, 7)), (slice(0, 7), slice(7, 14)),
                  (slice(7, 14), slice(0, 7)), (slice(7, 14), slice(7, 14)))
        for layer, (rows, cols) in zip(child.projections, slices):
            torch.testing.assert_close(layer.weight, 4 * parent.weight[:, :, rows, cols].reshape(8, -1))
            torch.testing.assert_close(layer.bias, parent.bias)
        image = torch.randn(2, 3, 224, 224)
        raw = child(subdivide_parent_patches(extract_parent_patches(image)))
        expected = parent(image).flatten(2).transpose(1, 2)
        torch.testing.assert_close(raw.mean(2), expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(parent.weight, before_w)
        torch.testing.assert_close(parent.bias, before_b)
        self.assertTrue(child.equals_initialization)
        child.projections[0].weight.data.add_(1)
        self.assertFalse(child.equals_initialization)

    def test_trainability_roundtrip_dtype_and_rejections(self):
        parent = self.make(dtype=torch.float64)
        child = ChildProjector(parent, provenance="fixture", trainable=False)
        self.assertTrue(all(not p.requires_grad for p in child.parameters()))
        self.assertEqual(child.projections[0].weight.dtype, torch.float64)
        copy = ChildProjector(parent, provenance="fixture", trainable=True)
        copy.load_state_dict(child.state_dict())
        for a, b in zip(child.state_dict().values(), copy.state_dict().values()):
            torch.testing.assert_close(a, b)
        with self.assertRaisesRegex(ValueError, "provenance"):
            ChildProjector(parent, provenance="")
        with self.assertRaisesRegex(ValueError, "kernel"):
            ChildProjector(nn.Conv2d(3, 8, 7, 7), provenance="fixture")
        patch = nn.Module()
        patch.proj, patch.norm = parent, nn.LayerNorm(8)
        with self.assertRaisesRegex(ValueError, "non-identity"):
            ChildProjector.from_patch_embed(patch, provenance="fixture")

    def test_consistency_diagnostics_and_gradients(self):
        parent_proj = self.make()
        child = ChildProjector(parent_proj, provenance="fixture")
        pixels = torch.randn(1, 256, 4, 3, 7, 7)
        parent = torch.randn(1, 256, 8, requires_grad=True)
        raw = child(pixels)
        result = enforce_parent_consistency(raw, parent)
        torch.testing.assert_close(result.consistent.mean(2), parent, rtol=1e-5, atol=2e-7)
        self.assertGreater(float(result.raw_consistency_loss.detach()), 0)
        self.assertTrue(torch.isfinite(result.consistent).all())
        (result.consistent.square().mean() + result.raw_consistency_loss).backward()
        self.assertTrue(all(layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
                            for layer in child.projections))
        self.assertIsNotNone(parent.grad)
        self.assertGreater(float(parent.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "raw"):
            enforce_parent_consistency(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 4))


class HaarPositionTests(unittest.TestCase):
    def test_haar_invariants_signs_inverse_gradient_determinism(self):
        haar = HaarDetails()
        torch.testing.assert_close(haar.q.sum(1), torch.zeros(3))
        torch.testing.assert_close(haar.q @ haar.q.T, torch.eye(3))
        self.assertEqual(haar.q.tolist(), [[.5, -.5, .5, -.5], [.5, .5, -.5, -.5],
                                          [.5, -.5, -.5, .5]])
        parent = torch.randn(2, 256, 7, requires_grad=True)
        residual = torch.randn(2, 256, 4, 7, requires_grad=True)
        children = residual - residual.mean(2, keepdim=True) + parent.unsqueeze(2)
        first, second = haar(children), haar(children)
        self.assertEqual(first.shape, (2, 256, 3, 7))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        rebuilt = haar.reconstruct(parent, first)
        torch.testing.assert_close(rebuilt, children, rtol=1e-5, atol=5e-7)
        rebuilt.sum().backward()
        self.assertIsNotNone(parent.grad)
        self.assertIsNotNone(residual.grad)

    def test_positions_policy_prefix_roundtrip_and_validation(self):
        module = ParentAwareDetailPositions(8, test_only=True)
        parent = torch.randn(1, 256, 8)
        prefix = torch.randn(1, 5, 8)
        detail = module(parent)
        self.assertEqual(detail.shape, (1, 256, 3, 8))
        for mode in range(3):
            torch.testing.assert_close(detail[:, :, mode], parent)
        kept, _ = module.split_official_positions(prefix, parent)
        self.assertIs(kept, prefix)
        module.mode_embeddings.data.copy_(torch.arange(24).reshape(3, 8))
        restored = ParentAwareDetailPositions(8, test_only=True)
        restored.load_state_dict(module.state_dict())
        torch.testing.assert_close(restored(parent), module(parent))
        with self.assertRaisesRegex(ValueError, "768"):
            ParentAwareDetailPositions(8)
        with self.assertRaisesRegex(ValueError, "parent positions"):
            module(torch.randn(1, 255, 8))


class AssemblyTests(unittest.TestCase):
    def make(self, counts):
        b, d = len(counts), 4
        prefix = torch.arange(b * 2 * d, dtype=torch.float32).reshape(b, 2, d).requires_grad_()
        parents = (100 + torch.arange(b * 256 * d, dtype=torch.float32)).reshape(b, 256, d).requires_grad_()
        details = (10000 + torch.arange(b * 256 * 3 * d, dtype=torch.float32)).reshape(b, 256, 3, d).requires_grad_()
        selected = torch.zeros(b, 256, dtype=torch.bool)
        for row, count in enumerate(counts):
            selected[row, :count] = True
        return prefix, parents, details, selected

    def test_zero_one_multiple_all_heterogeneous_order_padding(self):
        values = self.make([0, 1, 3, 256])
        result = assemble_tokens(*values)
        self.assertEqual(result.effective_token_count.tolist(), [258, 261, 267, 1026])
        self.assertEqual(result.padded_token_count, 1026)
        self.assertEqual(result.selected_parent_count.tolist(), [0, 1, 3, 256])
        self.assertTrue((result.token_kind[0, 258:] == TokenKind.PADDING).all())
        self.assertTrue((result.parent_index[0, 258:] == -1).all())
        self.assertTrue((result.detail_mode[0, 258:] == -1).all())
        self.assertFalse(result.valid_mask[0, 258:].any())
        torch.testing.assert_close(result.tokens[:, :2], values[0])
        torch.testing.assert_close(result.tokens[:, 2:258], values[1])
        self.assertEqual(result.parent_index[2, 258:267].tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(result.detail_mode[2, 258:267].tolist(), [0, 1, 2] * 3)
        self.assertEqual(result.unpadded(1)["tokens"].shape[0], 261)
        result.tokens[result.valid_mask].sum().backward()
        self.assertIsNotNone(values[0].grad)
        self.assertIsNotNone(values[1].grad)
        self.assertIsNotNone(values[2].grad)

    def test_malformed(self):
        values = list(self.make([1]))
        values[3] = values[3].long()
        with self.assertRaisesRegex(ValueError, "bool"):
            assemble_tokens(*values)
        values = list(self.make([1]))
        values[2] = torch.randn(1, 255, 3, 4)
        with self.assertRaisesRegex(ValueError, "shapes"):
            assemble_tokens(*values)


class DiagnosticConfigTests(unittest.TestCase):
    def test_fixture_deterministic_and_strict_config(self):
        path = Path("configs/smoke/m3_subtokens.yaml")
        first, second = run_fixture_diagnostic(path), run_fixture_diagnostic(path)
        self.assertEqual(first, second)
        self.assertIn("NON-REPORTABLE", first["label"])
        self.assertEqual(first["effective_token_count"], 256 + 15)
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.yaml"
            bad.write_text(path.read_text() + "\nrouter: forbidden\n")
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_subtoken_config(bad)


@pytest.mark.gpu
def test_m3_cuda_fp32_and_bf16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    props = torch.cuda.get_device_properties(0)
    assert "A100" in props.name and props.total_memory >= 75 * 1024**3
    for dtype, atol in ((torch.float32, 2e-6), (torch.bfloat16, .05)):
        parent_proj = nn.Conv2d(3, 8, 14, 14, device="cuda:0", dtype=dtype)
        image = torch.randn(1, 3, 224, 224, device="cuda:0", dtype=dtype)
        children = subdivide_parent_patches(extract_parent_patches(image))
        raw = ChildProjector(parent_proj, provenance="CUDA TEST")(children)
        parent = parent_proj(image).flatten(2).transpose(1, 2)
        result = enforce_parent_consistency(raw, parent)
        assert float(result.max_absolute_error.float()) <= atol
        assert torch.isfinite(HaarDetails().to("cuda:0", dtype)(result.consistent)).all()


if __name__ == "__main__":
    unittest.main()
