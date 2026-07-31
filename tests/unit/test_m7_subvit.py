from __future__ import annotations

import tempfile
import unittest
import torch

from deltasub.diagnostics.subvit.attention import (
    deterministic_topk, extract_cls_patch_attention, sample_attention_heads)
from deltasub.diagnostics.subvit.ats import assemble_ats, interpolate_child_positions
from deltasub.diagnostics.subvit.degradation import evaluate_head_degradation
from deltasub.diagnostics.subvit.distillation import (
    SubViTRouter, SubViTRouterConfig, distillation_loss)
from deltasub.diagnostics.subvit.fixture import run_fixture
from deltasub.diagnostics.subvit.schema import Provenance, make_record


class M7SubViTTests(unittest.TestCase):
    def test_attention_register_exclusion_ties_and_schedule(self):
        raw = torch.zeros(2, 3, 261, 261)
        raw[:, :, 0, 5:] = torch.arange(256)
        maps = extract_cls_patch_attention(raw, prefix_tokens=5)
        self.assertEqual(maps.shape, (2, 3, 256))
        self.assertEqual(maps[0, 0, 0], 0)
        self.assertEqual(maps[0, 0, -1], 255)
        self.assertEqual(deterministic_topk(torch.zeros(1, 256), 3)[0], (0, 1, 2))
        self.assertEqual(sample_attention_heads(12, 4, seed=9), sample_attention_heads(12, 4, seed=9))

    def test_direct_geometry_parent_retention_order_counts_and_positions(self):
        b, p, d = 4, 5, 3
        prefix = torch.randn(b, p, d); parents = torch.randn(b, 256, d)
        children = torch.arange(b * 256 * 4 * d).reshape(b, 256, 4, d).float()
        pp = torch.randn(b, p, d); positions = torch.randn(b, 256, d)
        child_positions = interpolate_child_positions(positions)
        result = assemble_ats(prefix, parents, children, pp, positions, child_positions,
                              torch.zeros(b, 256), torch.tensor([0, 1, 2, 256]))
        torch.testing.assert_close(result.tokens[:, :p], prefix)
        torch.testing.assert_close(result.tokens[:, p:p + 256], parents)
        self.assertEqual(result.effective_lengths.tolist(), [261, 265, 269, 1285])
        self.assertEqual(result.parent_index[2, p + 256:p + 264].tolist(), [0] * 4 + [1] * 4)
        self.assertEqual(result.child_index[2, p + 256:p + 264].tolist(), [0, 1, 2, 3] * 2)
        self.assertEqual(child_positions.shape, (b, 256, 4, d))

    def test_degradation_isolation_tie_chunk(self):
        parents = torch.ones(2, 256, 2)
        maps = torch.zeros(2, 3, 256)
        maps[:, 1, 7] = 2
        def teacher(x, valid):
            return (x * valid[:, :, None]).sum(1)
        a = evaluate_head_degradation(parents, maps, 1, teacher)
        b = evaluate_head_degradation(parents, maps, 1, teacher, chunk_size=2)
        self.assertEqual(a.distances, b.distances)
        self.assertEqual(a.selected_heads, (0, 0))  # equal feature drops -> lowest head
        self.assertEqual(a.selected_masks[:, :, :].sum(2).tolist(), [[255] * 3, [255] * 3])

    def test_router_losses_gradients_no_pair_and_schema(self):
        router = SubViTRouter(SubViTRouterConfig(4, 8, test_only=True), seed=1)
        scores = router(torch.randn(2, 256, 4))
        target = torch.stack((torch.arange(256), torch.ones(256))).float()
        loss = distillation_loss(scores, target, 4)
        loss.total.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters()))
        constant = distillation_loss(torch.zeros(1, 256), torch.ones(1, 256), 0)
        self.assertEqual(constant.ranking, 0)
        provenance = Provenance("model", "a" * 64, "revision", 1, None, "s", "v", 4,
                                "b" * 64, "c" * 64)
        make_record("attention_extraction", provenance, {"shape": [1, 2, 256]}).validate()

    def test_fixture_independent_determinism(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = run_fixture(a), run_fixture(b)
            self.assertEqual(first["deterministic_hash"], second["deterministic_hash"])
            self.assertTrue(first["teacher_equal"])
            self.assertFalse(first["token_semantics"]["mechanisms_equivalent"])


if __name__ == "__main__":
    unittest.main()
