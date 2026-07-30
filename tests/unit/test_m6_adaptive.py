from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import torch

from deltasub.adaptive.accounting import account_samples, summarize_accounting
from deltasub.adaptive.assembly import assemble_adaptive
from deltasub.adaptive.budget import BudgetSpec, BudgetUnit, token_counts, usage_to_k
from deltasub.adaptive.config import load_adaptive_config
from deltasub.adaptive.controller import BudgetController, ControllerConfig
from deltasub.adaptive.schema import SelectionPlan, SELECTION_PLAN_SCHEMA_VERSION
from deltasub.adaptive.selection import deterministic_select

SHA = "a" * 64


class BudgetControllerTests(unittest.TestCase):
    def test_units_bounds_and_impossible(self):
        self.assertEqual(token_counts(4, 5), {
            "selected_parents": 4, "added_detail_tokens": 12,
            "spatial_tokens": 268, "total_tokens": 273})
        self.assertEqual(usage_to_k(273, BudgetUnit.TOTAL_TOKENS, 5), 4)
        self.assertEqual(BudgetSpec("fixed_k", "selected_parents", 7).target_selected_parents, 7)
        with self.assertRaises(ValueError):
            BudgetSpec("fixed_k", "added_detail_tokens", 7)
        with self.assertRaises(ValueError):
            BudgetSpec("mean_total_tokens", "total_tokens", 200, prefix_tokens=5)

    def test_controller_modes_update_no_validation_serialization(self):
        fixed = BudgetController(ControllerConfig(
            "fixed_k", BudgetSpec("fixed_k", "selected_parents", 3)))
        scores = torch.tensor([[1.] * 256, [-1.] * 256])
        self.assertEqual(fixed.choose_k(scores).tolist(), [3, 3])
        threshold = BudgetController(ControllerConfig(
            "threshold_with_bounds", BudgetSpec("per_sample_k", "selected_parents", 2, 1, 4),
            threshold=0.0))
        self.assertEqual(threshold.choose_k(scores).tolist(), [4, 1])
        dual = BudgetController(ControllerConfig(
            "dual_threshold", BudgetSpec("mean_detail_tokens", "added_detail_tokens", 3),
            initial_lambda=.2, dual_lr=.1, lambda_min=0, lambda_max=1, update_interval=2))
        self.assertFalse(dual.accumulate(torch.tensor([2]), training=False,
                                         optimizer_step_succeeded=True))
        self.assertFalse(dual.accumulate(torch.tensor([2]), training=True,
                                         optimizer_step_succeeded=False))
        self.assertFalse(dual.accumulate(torch.tensor([2]), training=True,
                                         optimizer_step_succeeded=True))
        self.assertTrue(dual.accumulate(torch.tensor([2]), training=True,
                                        optimizer_step_succeeded=True))
        self.assertAlmostEqual(dual.lambda_value, .5)  # .2 + .1 * (6 - 3)
        restored = BudgetController(dual.config); restored.load_state_dict(dual.state_dict())
        self.assertEqual(restored.state_dict(), dual.state_dict())
        bad = dict(dual.state_dict()); bad["lambda_value"] = .1
        with self.assertRaisesRegex(ValueError, "hash"):
            restored.load_state_dict(bad)


class SelectionAssemblySchemaTests(unittest.TestCase):
    def test_selection_ties_extrema_masks_bf16(self):
        scores = torch.zeros(4, 256, dtype=torch.bfloat16)
        scores[2, 9] = 2; scores[2, 3] = 2
        valid = torch.ones(4, 256, dtype=torch.bool); valid[2, 0] = False
        result = deterministic_select(scores, [0, 256, 2, 3], valid)
        self.assertEqual(result.selected_indices[0], ())
        self.assertEqual(len(result.selected_indices[1]), 256)
        self.assertEqual(result.selected_indices[2], (3, 9))
        self.assertEqual(result.selected_indices[3], (0, 1, 2))
        self.assertEqual(result.selected_mask.sum(1).tolist(), [0, 256, 2, 3])
        with self.assertRaises(ValueError):
            deterministic_select(torch.full((1, 256), float("nan")), [1])
        with self.assertRaises(ValueError):
            deterministic_select(scores[:1], [257])

    def test_assembly_preservation_order_gradient_noncontiguous(self):
        b, p, d = 2, 2, 4
        prefix = torch.randn(b, p, d, requires_grad=True)
        parents = torch.randn(b, 256, d, requires_grad=True)
        details = torch.randn(b, 256, 3, d, requires_grad=True)
        selection = deterministic_select(torch.zeros(b, 256), [1, 3])
        position = torch.randn(b, 256, d)
        result = assemble_adaptive(prefix, parents, details, prefix.clone(), position,
                                   position[:, :, None].expand(-1, -1, 3, -1), selection)
        torch.testing.assert_close(result.tokens[:, :p], prefix)
        torch.testing.assert_close(result.tokens[:, p:p + 256], parents)
        self.assertEqual(result.parent_index[1, p + 256:p + 265].tolist(),
                         [0, 0, 0, 1, 1, 1, 2, 2, 2])
        result.tokens[result.valid_mask].sum().backward()
        self.assertIsNotNone(details.grad)

    def test_plan_hash_and_rejections(self):
        scores = [0.] * 256
        order = tuple(range(256))
        plan = SelectionPlan(
            SELECTION_PLAN_SCHEMA_VERSION, "s", "v", SHA, SHA, SHA, SHA, SHA,
            "selected_parents", 1, 0, 256, 1, (0,), SHA, order, 1, 256, 3, 260,
            None, SHA, "revision", "projector").with_hash()
        plan.validate(scores)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            replace(plan, selected_parent_indices=(0, 0), realized_k=2,
                    added_detail_token_count=6, effective_total_token_count=263).with_hash().validate()

    def test_accounting_and_config(self):
        budget = BudgetSpec("mean_total_tokens", "total_tokens", 260, prefix_tokens=1)
        records = account_samples([0, 1], 1, [260, 260], ["a", "a"])
        summary = summarize_accounting(records, budget)
        self.assertEqual(summary["mean_k"], .5)
        self.assertEqual(summary["units"]["effective"], "total_tokens")
        self.assertTrue(summary["token_counts_are_not_flops"])
        self.assertEqual(load_adaptive_config("configs/smoke/m6_adaptive.yaml")["mode"], "fixture")


if __name__ == "__main__":
    unittest.main()
