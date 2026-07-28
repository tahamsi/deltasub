from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest

import torch

from deltasub.router.losses import deterministic_pairs, router_loss
from deltasub.router.metrics import evaluate_metrics
from deltasub.router.model import GainRouter, RouterConfig
from deltasub.router.replay import DeterministicReplayBuffer
from deltasub.router.sampling import plan_epoch
from deltasub.router.schema import ROUTER_EXAMPLE_SCHEMA_VERSION, RouterExample


SHA = "a" * 64


def example(index=0, sample="sample", split="train", gain=1., labelled=True):
    return RouterExample(
        ROUTER_EXAMPLE_SCHEMA_VERSION, SHA, f"{sample}|{index}", SHA, sample, 0,
        index, index // 16, index % 16, labelled, "known", gain, 2., 1., True,
        SHA, SHA, split, dataset_manifest_sha256=SHA, model_checkpoint_sha256=SHA,
        gain_configuration_sha256=SHA,
    )


class RouterModelTests(unittest.TestCase):
    def test_shape_initialization_features_count_compute_and_roundtrip(self):
        config = RouterConfig(8, 16, 2, test_only=True)
        first, second = GainRouter(config, seed=9), GainRouter(config, seed=9)
        parents = torch.randn(2, 256, 8)
        self.assertEqual(first(parents).shape, (2, 256))
        self.assertEqual(first.construct_features(parents).shape, (2, 256, 18))
        self.assertEqual(first.parameter_count, 657)
        self.assertEqual(first.multiply_add_estimate(), 143360)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        second.load_state_dict(first.state_dict())
        torch.testing.assert_close(first(parents), second(parents), rtol=0, atol=0)

    def test_guards_and_no_leakage_surface(self):
        with self.assertRaisesRegex(ValueError, "production"):
            GainRouter(RouterConfig(input_dim=8))
        router = GainRouter(RouterConfig(8, test_only=True))
        with self.assertRaises(ValueError):
            router(torch.randn(1, 255, 8))
        self.assertNotIn("gain", RouterConfig.__dataclass_fields__)
        self.assertNotIn("counterfactual", RouterConfig.__dataclass_fields__)


class SamplingReplayLossMetricTests(unittest.TestCase):
    def test_sampling_is_deterministic_two_stream_no_leakage(self):
        values = [example(i, f"s{i // 3}", gain=(-1) ** i * i) for i in range(9)]
        first = plan_epoch(values, batch_size=4, informative_fraction=.5, seed=3, epoch=0)
        second = plan_epoch(values, batch_size=4, informative_fraction=.5, seed=3, epoch=0)
        self.assertEqual(first.sha256, second.sha256)
        self.assertGreater(first.coverage_count, 0)
        self.assertGreater(first.informative_count, 0)
        for batch in first.batches:
            self.assertEqual(len({x.gain_record_key for x in batch}), len(batch))
            self.assertTrue(all(x.sample_weight == 1 for x in batch))
        with self.assertRaisesRegex(ValueError, "validation"):
            plan_epoch([example(split="validation")], batch_size=1,
                       informative_fraction=0, seed=0, epoch=0)

    def test_replay_eviction_priority_roundtrip_and_guards(self):
        replay = DeterministicReplayBuffer(3, SHA, "b" * 64)
        for i, gain in enumerate((1., -1., 0., 2.)):
            replay.add(example(i, gain=gain), priority=i)
        self.assertEqual(len(replay.entries), 3)
        key = next(iter(replay.entries)); replay.update_priority(key, 7)
        state = replay.state_dict()
        restored = DeterministicReplayBuffer(3, SHA, "b" * 64)
        restored.load_state_dict(state)
        self.assertEqual(restored.checksum, replay.checksum)
        with self.assertRaisesRegex(ValueError, "validation"):
            replay.add(example(5, split="validation"))
        bad = DeterministicReplayBuffer(3, "c" * 64, "b" * 64)
        with self.assertRaisesRegex(ValueError, "gain_cache_id"):
            bad.load_state_dict(state)

    def test_losses_exact_mask_pairs_no_pair_and_gradients(self):
        scores = torch.tensor([0., 1., 2., 3.], requires_grad=True)
        targets = torch.tensor([0., 2., 1., 9.])
        groups = torch.tensor([0, 0, 0, 1])
        valid = torch.tensor([True, True, True, False])
        pairs = deterministic_pairs(targets, groups, valid, target_margin=.1,
                                    maximum_pairs_per_anchor=10)
        self.assertEqual(pairs, [(1, 0), (1, 2), (2, 0)])
        output = router_loss(scores, targets, groups, valid, sign_weight=.2,
                             target_margin=.1, maximum_pairs_per_anchor=10)
        output.total.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(output.valid_count, 3)
        self.assertEqual(output.pair_count, 3)
        no_pairs = router_loss(scores[:2], torch.ones(2), torch.zeros(2, dtype=torch.long),
                               torch.ones(2, dtype=torch.bool), ranking_weight=1)
        self.assertEqual(no_pairs.ranking.item(), 0.)

    def test_metrics_perfect_reverse_constant_quantiles(self):
        perfect = evaluate_metrics([0, 1, 2], [0, 1, 2], ["a"] * 3)
        reverse = evaluate_metrics([2, 1, 0], [0, 1, 2], ["a"] * 3)
        constant = evaluate_metrics([1, 1], [1, 1], ["a", "a"])
        self.assertEqual(perfect["pairwise_ranking_accuracy"]["value"], 1.)
        self.assertEqual(reverse["pairwise_ranking_accuracy"]["value"], 0.)
        self.assertFalse(constant["spearman"]["defined"])
        self.assertTrue(perfect["quantile_calibration"])
        self.assertTrue(perfect["top_k_is_diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
