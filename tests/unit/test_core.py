from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from deltasub.data.gcd_splits import deterministic_class_split
from deltasub.data.manifests import validate_manifest, write_manifest
from deltasub.losses.parent_consistency import parent_consistency
from deltasub.losses.per_anchor_selex import per_anchor_selex
from deltasub.losses.router_loss import pairwise_ranking_loss, router_loss
from deltasub.losses.selex import scalar_selex
from deltasub.models.deltasub.bucketed_inference import (
    bucketed_forward,
    token_counts,
)
from deltasub.models.deltasub.budget_controller import (
    DualBudgetController,
    bucket_k,
    select_adaptive,
)
from deltasub.models.deltasub.candidate_sampler import sample_candidates
from deltasub.models.deltasub.child_patch_embed import (
    ChildPatchEmbed,
    extract_parent_patches,
    subdivide_parent_patches,
)
from deltasub.models.deltasub.detail_position import DetailPosition
from deltasub.models.deltasub.gain_collector import (
    normalized_gain,
    repeated_base_check,
)
from deltasub.models.deltasub.haar_detail import (
    haar_details,
    haar_matrix,
    reconstruct_children,
)
from deltasub.models.deltasub.model import TinyDeltaSub
from deltasub.models.deltasub.replay_buffer import StratifiedReplayBuffer
from deltasub.reporting.tables import format_best_second, summarize_runs
from deltasub.utils.checkpointing import atomic_torch_save, load_checkpoint
from deltasub.utils.hashing import stable_hash


class GeometryTests(unittest.TestCase):
    def test_parent_patch_extraction(self):
        images = torch.arange(2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
        patches = extract_parent_patches(images)
        self.assertEqual(patches.shape, (2, 256, 3, 14, 14))
        torch.testing.assert_close(patches[0, 0], images[0, :, :14, :14])

    def test_direct_subdivision(self):
        parents = torch.randn(2, 3, 3, 14, 14)
        children = subdivide_parent_patches(parents)
        self.assertEqual(children.shape, (2, 3, 4, 3, 7, 7))
        torch.testing.assert_close(children[:, :, 0], parents[:, :, :, :7, :7])

    def test_child_projection(self):
        children = torch.randn(2, 3, 4, 3, 7, 7)
        self.assertEqual(ChildPatchEmbed(16)(children).shape, (2, 3, 4, 16))

    def test_haar_properties_and_reconstruction(self):
        q = haar_matrix(dtype=torch.float64)
        torch.testing.assert_close(q @ torch.ones(4, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
        torch.testing.assert_close(q @ q.T, torch.eye(3, dtype=torch.float64))
        children = torch.randn(2, 5, 4, 7, dtype=torch.float64)
        details = haar_details(children)
        reconstructed = reconstruct_children(children.mean(-2), details)
        torch.testing.assert_close(reconstructed, children)

    def test_position_and_parent_consistency(self):
        module = DetailPosition(8)
        parent = torch.randn(256, 8)
        selected = torch.tensor([0, 9])
        self.assertEqual(module(parent, selected).shape, (2, 3, 8))
        value = torch.randn(2, 8)
        self.assertEqual(parent_consistency(value, value).item(), 0.0)


class LossTests(unittest.TestCase):
    @staticmethod
    def inputs():
        torch.manual_seed(3)
        n = 6
        logits = torch.randn(n, n, dtype=torch.float64)
        valid = ~torch.eye(n, dtype=torch.bool)
        unsup = torch.roll(torch.eye(n, dtype=torch.bool), 1, 1) & valid
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        supervised = labels[:, None].eq(labels[None, :]) & valid
        hierarchical = [labels.div(2, rounding_mode="floor")[:, None].eq(labels.div(2, rounding_mode="floor")[None, :]) & valid]
        labelled = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
        confidence = torch.linspace(0.5, 1, n, dtype=torch.float64)
        return logits, unsup, supervised, hierarchical, valid, labelled, confidence

    def test_per_anchor_equals_scalar(self):
        inputs = self.inputs()
        vector = per_anchor_selex(*inputs).total
        scalar = scalar_selex(*inputs)
        torch.testing.assert_close(vector.mean(), scalar, rtol=0, atol=1e-12)

    def test_router_loss_and_margin(self):
        prediction = torch.tensor([0.1, 0.2, 0.3, 0.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0, 1.0, -1.0])
        image_ids = torch.tensor([0, 0, 1, 1])
        ignored = pairwise_ranking_loss(prediction[:2], target[:2], image_ids[:2], 0.01)
        self.assertEqual(ignored.item(), 0.0)
        total, parts = router_loss(prediction, target, torch.tensor([1, 1, 0, 0], dtype=torch.bool), image_ids)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(parts), {"huber", "ranking", "calibration"})


class DeterminismAndSamplingTests(unittest.TestCase):
    def test_deterministic_repeated_base_and_synthetic_gains(self):
        torch.manual_seed(0)
        model = TinyDeltaSub(embed_dim=8, classes=2)
        images = torch.rand(2, 3, 224, 224)
        labels = torch.tensor([0, 1])
        report = repeated_base_check(model, images, labels, 1e-5, 1e-5)
        self.assertLess(report.loss_difference, 1e-5)
        base = torch.tensor([2.0, 2.0, 2.0])
        split = torch.tensor([1.0, 2.0, 3.0])
        raw, gain = normalized_gain(base, split)
        self.assertEqual(raw.sign().tolist(), [1.0, 0.0, -1.0])
        self.assertEqual(gain.sign().tolist(), [1.0, 0.0, -1.0])

    def test_candidate_streams(self):
        router = torch.arange(32, dtype=torch.float32).reshape(2, 16)
        detail = router.flip(1)
        generator = torch.Generator().manual_seed(1)
        batch = sample_candidates(router, detail, total=4, random_count=2, generator=generator)
        self.assertEqual(batch.indices.shape, (2, 4))
        self.assertEqual(batch.exploration_probability, 2 / 16)
        for indices, sources in zip(batch.indices, batch.sources):
            self.assertEqual(len(indices.unique()), 4)
            self.assertEqual(sources.count("exploration"), 2)
            self.assertTrue(any(source != "exploration" for source in sources))

    def test_replay_buffer_stratification(self):
        buffer = StratifiedReplayBuffer(2)
        for record in [
            {"normalized_gain": 1.0},
            {"normalized_gain": 0.0},
            {"normalized_gain": -1.0},
            {"normalized_gain": 0.0, "disagreement": 1.0},
            {"normalized_gain": 0.0, "low_confidence": True},
        ]:
            buffer.add(record)
        self.assertEqual(set(buffer.counts()), {"positive", "near_zero", "negative", "high_disagreement", "low_confidence"})


class BudgetAndCacheTests(unittest.TestCase):
    def test_adaptive_constraints_and_dual_update(self):
        scores = torch.tensor([[1.0, -1.0, 0.5]])
        selected = select_adaptive(scores, penalty=0.0, maximum_k=1)
        self.assertEqual(len(selected[0]), 1)
        torch.testing.assert_close(bucket_k(torch.tensor([0, 3, 7, 30])), torch.tensor([0, 2, 8, 32]))
        controller = DualBudgetController(100, eta=0.1)
        self.assertEqual(controller.update(90), 0.0)
        self.assertEqual(controller.update(120), 2.0)

    def test_effective_and_padded_tokens(self):
        selections = [torch.arange(1), torch.arange(4)]
        self.assertEqual(token_counts(selections), (256 + 3 + 256 + 12, 2 * (256 + 12)))

    def test_bucketed_matches_unbucketed(self):
        torch.manual_seed(2)
        model = TinyDeltaSub(embed_dim=8, classes=2).eval()
        images = torch.rand(3, 3, 224, 224)
        selections = [torch.arange(0), torch.arange(1), torch.arange(1)]
        with torch.no_grad():
            direct = model(images, selections)
            bucketed = bucketed_forward(model, images, selections)
        torch.testing.assert_close(bucketed, direct, atol=1e-5, rtol=1e-5)

    def test_cache_hash_and_atomic_checkpoint(self):
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))
        self.assertNotEqual(stable_hash({"a": 1}), stable_hash({"a": 2}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            atomic_torch_save({"step": 3}, path)
            self.assertEqual(load_checkpoint(path)["step"], 3)
            self.assertFalse(any(item.name.startswith(".state.pt") for item in path.parent.iterdir()))


class DataAndReportingTests(unittest.TestCase):
    def test_manifest_split_and_hash(self):
        split = deterministic_class_split(["d", "a", "c", "b"])
        self.assertEqual(split, {"known": ["a", "b"], "novel": ["c", "d"]})
        records = [
            {
                "sample_id": "x",
                "image_path": "/not/accessed/x.jpg",
                "original_class": "a",
                "class_status": "known",
                "label_status": "labelled",
                "split": "train",
                "dataset_version": "synthetic-v1",
                "source_checksum": "0" * 64,
            }
        ]
        self.assertEqual(validate_manifest(records), validate_manifest(records))
        with tempfile.TemporaryDirectory() as directory:
            checksum = write_manifest(records, Path(directory) / "manifest.jsonl")
            self.assertEqual(len(checksum), 64)

    def test_table_aggregation_and_formatting(self):
        frame = pd.DataFrame(
            [
                {"method": "a", "dataset": "cub", "seed": 0, "all": 1, "known": 2, "novel": 3},
                {"method": "a", "dataset": "cub", "seed": 1, "all": 3, "known": 4, "novel": 5},
            ]
        )
        summary = summarize_runs(frame)
        self.assertEqual(summary.loc[0, "all_mean"], 2)
        self.assertEqual(format_best_second([3, 2, 1]), ["**3.00**", "<u>2.00</u>", "1.00"])


if __name__ == "__main__":
    unittest.main()
