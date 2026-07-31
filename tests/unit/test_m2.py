"""M2 tests. Production DINOv2 integration is marked and requires a pinned checkout."""
from __future__ import annotations
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn
import yaml

from deltasub.losses.per_anchor_selex import (
    euclidean_cdist,
    masked_contrastive_per_anchor,
    selex_per_anchor,
)
from deltasub.losses.selex import selex_loss
from deltasub.models.backbones.dinov2 import (
    DINOV2_REVISION, TestOnlyTinyBackbone as TinyBackboneFixture, construct_official_vitb14,
    inspect_official_checkpoint,
)
from deltasub.training.baseline import run_baseline_training, validate_baseline
from deltasub.training import selex_equivalence
from deltasub.training.selex_equivalence import THRESHOLDS, verify_equivalence
from deltasub.utils.hashing import sha256_file


class SelExTests(unittest.TestCase):
    def make(self, batch=6, labelled="mixed", levels=1, device="cpu", dtype=torch.float32):
        generator = torch.Generator(device=device).manual_seed(9 + batch + levels)
        features = torch.randn(batch, 2, 16, generator=generator, device=device, dtype=dtype)
        labels = torch.arange(batch, device=device) // 2
        if labelled == "mixed":
            flag = torch.arange(batch, device=device) % 2 == 0
        else:
            flag = torch.full((batch,), labelled == "all", dtype=torch.bool, device=device)
        hierarchy = tuple(labels // 2 ** (i + 1) for i in range(levels))
        confusion = torch.rand(batch * 2, batch * 2, generator=generator, device=device, dtype=dtype)
        confusion /= confusion.sum(1, keepdim=True)
        return features, labels, flag, hierarchy, confusion

    def test_exact_pinned_reference_cpu_and_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                selex_equivalence.torch.cuda, "is_available",
                side_effect=AssertionError("CPU verification queried CUDA"),
            ):
                result = verify_equivalence(Path(directory) / "gate.json")
        self.assertEqual(result["status"], "passed")
        self.assertLessEqual(result["fp32_max_absolute_error"], THRESHOLDS["fp32_atol"])
        self.assertLessEqual(result["fp32_max_relative_error"], THRESHOLDS["fp32_rtol"])
        self.assertEqual(len(result["reference_source_sha256"]), 64)
        self.assertEqual(result["modes_run"], ["cpu_fp32"])
        self.assertEqual(result["cuda_fp32"]["status"], "not_requested")
        self.assertEqual(result["cuda_bf16"]["status"], "not_requested")
        self.assertEqual(
            result["numerical_policy"]["distance"], "euclidean_torch_cdist_p2"
        )

    def test_all_reductions_are_finite_and_deterministic(self):
        for batch in (2, 4, 6, 8):
            for labelled in ("mixed", "all", "none"):
                for levels in (0, 1, 2):
                    args = self.make(batch, labelled, levels)
                    first = selex_per_anchor(*args)
                    second = selex_per_anchor(*args)
                    self.assertTrue(torch.isfinite(first.total).all())
                    torch.testing.assert_close(first.total, second.total, rtol=0, atol=0)
                    torch.testing.assert_close(selex_loss(*args), first.total[first.valid].mean())

    def test_invalid_anchor_zero_and_exact_valid_denominator(self):
        logits = torch.tensor([[1., 2., 3.], [3., 1., 0.]])
        positive = torch.tensor([[False, False, False], [False, True, False]])
        valid = torch.ones_like(positive)
        values = masked_contrastive_per_anchor(logits, positive, valid)
        self.assertEqual(float(values[0]), 0.)
        self.assertTrue(torch.isfinite(values).all())
        expected = -(logits[1, 1] - torch.logsumexp(logits[1], 0))
        torch.testing.assert_close(values[1], expected)

    def test_unsupported_distance_dtype_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "FP32 on CPU/CUDA"):
            euclidean_cdist(torch.randn(3, 4, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "floating inputs must both"):
            selex_loss(*self.make(dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "requires include_cuda"):
            verify_equivalence(
                Path(tempfile.mkdtemp()) / "gate.json", include_bf16=True
            )

    @pytest.mark.gpu
    def test_cuda_fp32_and_bf16(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        result = verify_equivalence(
            Path(tempfile.mkdtemp()) / "gate.json",
            include_cuda=True,
            include_bf16=True,
        )
        self.assertEqual(result["cuda_fp32"]["status"], "passed")
        self.assertLessEqual(result["cuda_fp32"]["max_absolute_error"], THRESHOLDS["fp32_atol"])
        self.assertEqual(result["cuda_bf16"]["status"], "passed")
        self.assertLessEqual(result["cuda_bf16"]["max_absolute_error"], THRESHOLDS["bf16_atol"])
        self.assertIn("cuda_bf16_input_fp32_distance", result["modes_run"])
        self.assertIn("bf16_input_fp32", result["numerical_policy"]["cuda_bf16"])

        args = list(self.make(device="cuda:0", dtype=torch.bfloat16))
        args[0].requires_grad_(True)
        first = selex_loss(*args)
        first.backward()
        first_gradient = args[0].grad.detach().clone()
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(torch.isfinite(first))
        self.assertTrue(torch.isfinite(first_gradient).all())

        args[0].grad = None
        second = selex_loss(*args)
        second.backward()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(
            first_gradient, args[0].grad, rtol=0, atol=0
        )


@pytest.mark.integration
class OfficialDINOv2Tests(unittest.TestCase):
    def source(self):
        value = os.environ.get("DINOV2_SOURCE_ROOT")
        if not value:
            self.skipTest("DINOV2_SOURCE_ROOT pinned checkout not supplied")
        return Path(value)

    def test_official_architecture_complete_state_and_failures(self):
        source = self.source()
        model, _ = construct_official_vitb14(source)
        self.assertEqual(model.patch_embed(torch.zeros(1, 3, 224, 224)).shape, (1, 256, 768))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "complete.pt"
            torch.save(model.state_dict(), checkpoint)
            digest = sha256_file(checkpoint)
            _, report = inspect_official_checkpoint(checkpoint, digest, source_root=source)
            self.assertTrue(report.compatible)
            for mutation, message in (("missing", "missing"), ("unexpected", "unexpected"), ("shape", "shape")):
                state = model.state_dict()
                if mutation == "missing":
                    state.pop("cls_token")
                elif mutation == "unexpected":
                    state["critical.extra"] = torch.zeros(1)
                else:
                    state["cls_token"] = torch.zeros(1, 1, 1)
                path = Path(directory) / f"{mutation}.pt"
                torch.save({"state_dict": state}, path)
                _, bad = inspect_official_checkpoint(path, sha256_file(path), source_root=source)
                self.assertFalse(bad.compatible, message)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                inspect_official_checkpoint(checkpoint, "0" * 64, source_root=source)
            malformed = Path(directory) / "malformed.pt"
            malformed.write_bytes(b"not torch")
            with self.assertRaisesRegex(ValueError, "malformed"):
                inspect_official_checkpoint(malformed, sha256_file(malformed), source_root=source)


class BaselineTrainingTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow unavailable")
        records = []
        for index in range(4):
            image = root / f"{index}.png"
            Image.new("RGB", (32, 32), (index * 40, 20, 100)).save(image)
            records.append({
                "sample_id": str(index), "dataset": "test-only", "image_path": str(image),
                "original_class_id": index // 2, "original_class_name": str(index // 2),
                "known_or_novel": "known", "labelled_or_unlabelled": "labelled" if index < 2 else "unlabelled",
                "train_or_test_split": "train", "bounding_box": None,
                "source_archive_checksum": "0" * 64, "split_source": "test-only",
                "split_revision": "0" * 40, "manifest_schema_version": 1,
            })
        manifest = root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records))
        split = root / "split.json"
        split.write_text('{"validation_outcome":"passed"}')
        gate = root / "gate.json"
        verify_equivalence(gate)
        model = TinyBackboneFixture()
        checkpoint = root / "tiny.pt"
        torch.save(model.state_dict(), checkpoint)
        output = root / "run"
        config = {
            "schema_version": 1, "test_only": True, "seed": 2,
            "dataset": {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
                        "split_validation_report": str(split), "root": str(root)},
            "backbone": {"name": "test_only_tiny_backbone", "checkpoint": str(checkpoint),
                         "checkpoint_sha256": sha256_file(checkpoint), "embed_dim": 16},
            "training": {"physical_batch_size": 2, "gradient_accumulation": 2,
                         "effective_batch_size": 4, "optimizer": "sgd",
                         "learning_rate": .1, "epochs": 1, "num_workers": 0,
                         "device": "cpu"},
            "selex": {"equivalence_gate": str(gate), "temperature": 1.,
                      "supervised_weight": .35, "unsupervised_smoothing": 1.},
            "output_directory": str(output),
        }
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        return config, config_path, checkpoint, output

    def test_actual_optimization_artifacts_resume_and_incompatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, config_path, initial, output = self.make_fixture(root)
            before = torch.load(initial, map_location="cpu", weights_only=True)
            result = run_baseline_training(config_path)
            self.assertTrue(torch.isfinite(torch.tensor(result["best_loss"])))
            last = torch.load(
                output / "checkpoint_last.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(last["training_device"], "cpu")
            self.assertTrue(any(not torch.equal(before[k], last["model"][k]) for k in before))
            for name in ("checkpoint_last.pt", "checkpoint_best.pt", "metrics.jsonl",
                         "environment.json", "resolved_config.yaml"):
                self.assertTrue((output / name).is_file(), name)
            previous = last["global_step"]
            last["epoch"] = 0
            torch.save(last, output / "checkpoint_last.pt")
            resumed = run_baseline_training(config_path, resume=True)
            self.assertGreater(resumed["global_step"], previous)
            incompatible = dict(config)
            incompatible["seed"] = 99
            bad = root / "bad.yaml"
            bad.write_text(yaml.safe_dump(incompatible))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                run_baseline_training(bad, resume=True)

    def test_device_policy_validation_and_cpu_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, config_path, _, output = self.make_fixture(root)
            config["training"]["device"] = "tpu"
            config_path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "training.device"):
                validate_baseline(config_path)

            config["training"]["device"] = "cuda"
            config_path.write_text(yaml.safe_dump(config))
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                    validate_baseline(config_path)

            config["training"]["device"] = "cpu"
            config_path.write_text(yaml.safe_dump(config))
            run_baseline_training(config_path)
            environment = json.loads((output / "environment.json").read_text())
            self.assertEqual(environment["requested_device"], "cpu")
            self.assertEqual(environment["resolved_device"], "cpu")

    def test_gate_status_alone_is_rejected_and_test_model_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, config_path, _, _ = self.make_fixture(root)
            Path(config["selex"]["equivalence_gate"]).write_text('{"status":"passed"}')
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_baseline(config_path)
            verify_equivalence(config["selex"]["equivalence_gate"])
            config["test_only"] = False
            config_path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "test-only"):
                validate_baseline(config_path)


if __name__ == "__main__":
    unittest.main()
