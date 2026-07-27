from __future__ import annotations

from dataclasses import replace
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from deltasub.gains.cache import GainCache, inspect_cache, open_cache
from deltasub.gains.config import load_gain_config
from deltasub.gains.counterfactual import BatchContext, module_sha256
from deltasub.gains.fixture import (
    fixture_inputs, make_fixture_evaluator, run_fixture_collection,
)
from deltasub.gains.planning import CandidatePlan
from deltasub.utils.hashing import stable_hash


class CounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fixture_inputs()
        cls.evaluator, cls.model, cls.child, cls.positions, cls.head = make_fixture_evaluator()
        images, labels, labelled, hierarchy, confusion, confidence = cls.inputs
        cls.context = BatchContext.build(
            images=images, sample_ids=("a", "b", "c"), view_ids=("v0", "v1"),
            augmentation_seeds=(1, 2), augmentation_parameters={"fixed": True},
            labels=labels, labelled=labelled, hierarchy_labels=hierarchy,
            confusion_factor=confusion,
            pseudo_label_confidence=confidence, model=cls.model,
            child_projector=cls.child, position_module=cls.positions, head=cls.head,
            configuration_hash="0" * 64, precision="fp32", device="cpu",
            source_git_commit="test",
        )

    def evaluate(self, anchor=1, parent=7, cache=None):
        images, labels, labelled, hierarchy, confusion, confidence = self.inputs
        return self.evaluator.evaluate(
            images=images, labels=labels, labelled=labelled,
            hierarchy_labels=hierarchy, confusion_factor=confusion,
            pseudo_label_confidence=confidence, context=self.context,
            anchor=anchor, parent=parent, base_cache=cache,
        )

    def test_exact_tokens_gain_sign_spillover_and_repeat(self):
        first = self.evaluate()
        second = self.evaluate()
        self.assertEqual(first.gain, first.base_anchor_loss - first.counterfactual_anchor_loss)
        self.assertEqual(first.counterfactual_effective_token_count,
                         first.base_effective_token_count + 3)
        self.assertEqual(first.gain, second.gain)
        self.assertEqual(first.spillover_sum, second.spillover_sum)
        self.assertTrue(torch.isfinite(torch.tensor(first.gain)))

    def test_candidate_isolation_chunk_and_base_reuse(self):
        reference = [self.evaluate(0, parent, {}) for parent in (0, 7, 255)]
        images, labels, labelled, hierarchy, confusion, confidence = self.inputs
        chunk = self.evaluator.evaluate_chunk(
            [(0, 0), (0, 7), (0, 255)], images=images, labels=labels,
            labelled=labelled, hierarchy_labels=hierarchy,
            confusion_factor=confusion, pseudo_label_confidence=confidence,
            context=self.context,
        )
        self.assertEqual([x.gain for x in reference], [x.gain for x in chunk])
        self.assertEqual([x.parent for x in chunk], [0, 7, 255])

    def test_rng_mode_and_parameter_state_restoration(self):
        self.model.train()
        context = replace(self.context, model_mode="train")
        before = module_sha256(self.model)
        python_state, numpy_state, torch_state = (
            random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        )
        old = self.context
        self.__class__.context = context
        try:
            self.evaluate()
        finally:
            self.__class__.context = old
        self.assertTrue(self.model.training)
        self.assertEqual(before, module_sha256(self.model))
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(numpy_state[0], np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        torch.testing.assert_close(torch_state, torch.get_rng_state(), rtol=0, atol=0)
        self.model.eval()

    def test_context_hash_stability_sensitivity_and_mismatch_rejection(self):
        self.assertEqual(self.context.sha256, self.context.sha256)
        for field in self.context.__dataclass_fields__:
            value = getattr(self.context, field)
            if isinstance(value, tuple):
                changed = value + ("changed",)
            else:
                changed = str(value) + "-changed"
            self.assertNotEqual(self.context.sha256, replace(self.context, **{field: changed}).sha256)
        bad = replace(self.context, configuration_hash="1" * 64)
        # Context hashes are opaque but module-bound fields fail closed.
        bad = replace(bad, model_state_hash="2" * 64)
        old = self.context
        self.__class__.context = bad
        try:
            with self.assertRaisesRegex(ValueError, "immutable"):
                self.evaluate()
        finally:
            self.__class__.context = old


class PlanningConfigTests(unittest.TestCase):
    def test_plans_and_order(self):
        plan = CandidatePlan.from_config({
            "mode": "list", "parent_indices": [7, 0, 255],
            "sample_limit": 2, "batch_limit": 1,
        })
        self.assertEqual(plan.parents, (0, 7, 255))
        self.assertEqual(
            list(plan.iter_keys([["a", "b", "c"], ["d"]])),
            [(0, 0, 0), (0, 0, 7), (0, 0, 255),
             (0, 1, 0), (0, 1, 7), (0, 1, 255)],
        )
        for value, message in (
            ({"mode": "list", "parent_indices": []}, "empty"),
            ({"mode": "list", "parent_indices": [1, 1]}, "duplicate"),
            ({"mode": "list", "parent_indices": [256]}, r"\[0, 255\]"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                CandidatePlan.from_config(value)

    def test_strict_config_guards(self):
        config = yaml.safe_load(Path("configs/smoke/m4_gains.yaml").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            loaded, plan = load_gain_config(path)
            self.assertTrue(loaded["test_only"])
            self.assertEqual(plan.parents, (0, 7, 255))
            for mutate, message in (
                (lambda x: x.update(router={}), "unknown"),
                (lambda x: x["collection"].update(precision="fp16"), "precision"),
                (lambda x: x["collection"]["candidate"].update(parent_indices=[0, 0]), "duplicate"),
                (lambda x: x["backbone"].update(checkpoint_sha256="bad"), "SHA256"),
            ):
                bad = yaml.safe_load(yaml.safe_dump(config))
                mutate(bad)
                path.write_text(yaml.safe_dump(bad))
                with self.assertRaisesRegex(ValueError, message):
                    load_gain_config(path)


class CacheFixtureTests(unittest.TestCase):
    def test_independent_rerun_resume_stream_inspect_and_hashes(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run_fixture_collection(first)
            b = run_fixture_collection(second)
            for key in (
                "cache_id", "batch_context_hash", "record_count",
                "maximum_repeated_evaluation_error", "maximum_chunk_equivalence_error",
                "shard_sha256_values", "deterministic_validation_hash",
                "model_state_hash_before", "model_state_hash_after",
            ):
                self.assertEqual(a[key], b[key], key)
            cache = open_cache(a["cache_path"])
            self.assertEqual(sum(1 for _ in cache.iter_records()), 9)
            resumed = run_fixture_collection(first, resume=True)
            self.assertEqual(resumed["processed_candidates"], 0)
            self.assertEqual(resumed["skipped_existing_candidates"], 9)
            report = inspect_cache(a["cache_path"])
            self.assertEqual(report["record_count"], 9)
            self.assertEqual(report["candidate_coverage"], [0, 7, 255])

    def test_corruption_metadata_temp_and_manual_status_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_fixture_collection(directory)
            root = Path(result["cache_path"])
            temporary = root / "shards" / "orphan.parquet.tmp"
            temporary.write_bytes(b"partial")
            envelope = json.loads((root / "metadata.json").read_text())
            GainCache(directory, envelope["metadata"], resume=True)
            self.assertFalse(temporary.exists())
            validation = json.loads((root / "validation.json").read_text())
            validation["complete"] = False
            (root / "validation.json").write_text(json.dumps(validation))
            # Completeness is recomputed from the immutable planned key digest.
            checked = open_cache(root).validate()
            self.assertTrue(checked["complete"])
            shard = next((root / "shards").glob("*.parquet"))
            shard.write_bytes(shard.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                open_cache(root)


if __name__ == "__main__":
    unittest.main()
