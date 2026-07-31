from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch import nn

from .cache import GainCache, inspect_cache
from .counterfactual import (
    BatchContext, CounterfactualGainEvaluator, module_sha256, tensor_sha256,
)
from .schema import GAIN_SCHEMA_VERSION, GainRecord
from ..models.subtokens.positions import ParentAwareDetailPositions
from ..models.subtokens.projection import ChildProjector
from ..training.selex_equivalence import PRODUCTION, REFERENCE
from ..utils.hashing import sha256_file, stable_hash

FIXTURE_SHA = "f" * 64
FIXTURE_TIMESTAMP = "2000-01-01T00:00:00+00:00"


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


class TestOnlyM4Transformer(nn.Module):
    test_only = True

    def __init__(self, dim: int = 6):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, 14, 14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.parent_positions = nn.Parameter(torch.zeros(1, 256, dim))
        layer = nn.TransformerEncoderLayer(
            dim, 2, dim_feedforward=12, dropout=0.0, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, 1)
        self.norm = nn.LayerNorm(dim)

    def pre_transformer_parent_embeddings(self, images):
        return self.patch_embed(images).flatten(2).transpose(1, 2)

    def parent_patch_positions(self):
        return self.parent_positions

    def prefix_tokens_with_positions(self, batch_size):
        return self.cls_token.expand(batch_size, -1, -1)

    def encode_tokens(self, sequence):
        return self.norm(self.transformer(sequence.unsqueeze(0))[0, 0])


def fixture_inputs(seed: int = 17):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(3, 3, 224, 224, generator=generator)
    # Deterministic two-view tensors are materialized once and reused exactly.
    images = torch.stack((base, base.flip(-1)), dim=1)
    labels = torch.tensor([0, 0, 1])
    labelled = torch.tensor([True, False, False])
    hierarchy = (torch.tensor([0, 0, 2]),)
    confusion = torch.rand(6, 6, generator=generator)
    confusion /= confusion.sum(1, keepdim=True)
    confidence = torch.tensor([1.0, 0.8, 0.2])
    return images, labels, labelled, hierarchy, confusion, confidence


def make_fixture_evaluator(seed: int = 17):
    torch.manual_seed(seed)
    model = TestOnlyM4Transformer()
    child = ChildProjector(model.patch_embed, trainable=False, provenance="TEST-ONLY M4")
    positions = ParentAwareDetailPositions(6, test_only=True)
    head = nn.Sequential(nn.Linear(6, 6), nn.GELU(), nn.Linear(6, 6))
    for module in (model, child, positions, head):
        module.eval()
        module.requires_grad_(False)
    evaluator = CounterfactualGainEvaluator(
        model, child, positions, head, model.encode_tokens
    )
    return evaluator, model, child, positions, head


def run_fixture_collection(output_root: str | Path, *, resume: bool = False,
                           candidates=(0, 7, 255), seed: int = 17) -> dict:
    started = time.perf_counter()
    images, labels, labelled, hierarchy, confusion, confidence = fixture_inputs(seed)
    evaluator, model, child, positions, head = make_fixture_evaluator(seed)
    state_before = stable_hash([
        module_sha256(module) for module in (model, child, positions, head)
    ])
    configuration_sha = stable_hash({
        "fixture": "m4", "seed": seed, "candidates": list(candidates),
        "precision": "fp32", "chunk_size": 2,
    })
    context = BatchContext.build(
        images=images, sample_ids=("fixture-0", "fixture-1", "fixture-2"),
        view_ids=("view-0", "view-1"), augmentation_seeds=(17, 18),
        augmentation_parameters={"view-0": "identity", "view-1": "horizontal_flip"},
        labels=labels, labelled=labelled, hierarchy_labels=hierarchy, confusion_factor=confusion,
        pseudo_label_confidence=confidence, model=model, child_projector=child,
        position_module=positions, head=head, configuration_hash=configuration_sha, precision="fp32",
        device="cpu", source_git_commit=git_commit(),
    )
    expected = {
        (context.sha256, sample_id, anchor, parent)
        for anchor, sample_id in enumerate(context.sample_ids) for parent in candidates
    }
    metadata = {
        "schema_version": GAIN_SCHEMA_VERSION, "dataset_name": "m4-test-only-fixture",
        "synthetic_only": True, "reportable": False,
        "label": "SYNTHETIC DIAGNOSTIC NON-REPORTABLE",
        "dataset_manifest_sha256": FIXTURE_SHA, "split_report_sha256": FIXTURE_SHA,
        "model_checkpoint_sha256": FIXTURE_SHA, "dinov2_source_revision": "TEST-ONLY",
        "child_projector_state_sha256": module_sha256(child),
        "selex_implementation_sha256": sha256_file(PRODUCTION),
        "selex_reference_sha256": sha256_file(REFERENCE),
        "selex_equivalence_gate_sha256": FIXTURE_SHA,
        "configuration_sha256": configuration_sha,
        "batch_context_hashes": [context.sha256],
        "candidate_subset": list(candidates), "candidate_subset_is_full": False,
        "sample_ids": list(context.sample_ids), "view_layout_identifier": "two-view-stacked-v1",
        "precision": "fp32", "resolved_device": "cpu", "seed": seed,
        "source_git_commit": git_commit(), "collection_timestamp": FIXTURE_TIMESTAMP,
        "deterministic_algorithm_exceptions": [],
        "planned_record_count": len(expected),
        "planned_key_sha256": stable_hash(sorted([list(key) for key in expected])),
    }
    cache = GainCache(output_root, metadata, shard_size=4, resume=resume)
    existing = len(cache.existing_keys)
    base_cache = {}
    evaluations = []
    repeated_error = 0.0
    chunk_error = 0.0
    for anchor in range(3):
        candidate_pairs = [(anchor, parent) for parent in candidates]
        reference = [
            evaluator.evaluate(
                images=images, labels=labels, labelled=labelled,
                hierarchy_labels=hierarchy, confusion_factor=confusion,
                pseudo_label_confidence=confidence, context=context,
                anchor=anchor, parent=parent, base_cache=base_cache,
            ) for parent in candidates
        ]
        if anchor == 0:
            repeated = evaluator.evaluate(
                images=images, labels=labels, labelled=labelled,
                hierarchy_labels=hierarchy, confusion_factor=confusion,
                pseudo_label_confidence=confidence, context=context,
                anchor=anchor, parent=candidates[0], base_cache={},
            )
            repeated_error = abs(repeated.gain - reference[0].gain)
            chunked = evaluator.evaluate_chunk(
                candidate_pairs, images=images, labels=labels, labelled=labelled,
                hierarchy_labels=hierarchy, confusion_factor=confusion,
                pseudo_label_confidence=confidence, context=context,
            )
            chunk_error = max(abs(a.gain - b.gain) for a, b in zip(reference, chunked))
        evaluations.extend(reference)
    records = []
    for evaluation in evaluations:
        parent = evaluation.parent
        records.append(GainRecord(
            GAIN_SCHEMA_VERSION, "m4-test-only-fixture", FIXTURE_SHA, FIXTURE_SHA,
            context.sample_ids[evaluation.anchor], evaluation.anchor, "two-view-stacked-v1",
            parent, parent // 16, parent % 16, bool(labelled[evaluation.anchor]),
            "known" if evaluation.anchor < 2 else "novel",
            int(labels[evaluation.anchor]) if labelled[evaluation.anchor] else None,
            evaluation.base_anchor_loss, evaluation.counterfactual_anchor_loss,
            evaluation.gain, evaluation.base_batch_loss, evaluation.counterfactual_batch_loss,
            evaluation.batch_loss_change, evaluation.spillover_sum,
            evaluation.spillover_max_abs, evaluation.anchor_valid,
            evaluation.base_effective_token_count, evaluation.counterfactual_effective_token_count,
            evaluation.base_padded_token_count, evaluation.counterfactual_padded_token_count,
            FIXTURE_SHA, "TEST-ONLY", module_sha256(child), sha256_file(PRODUCTION),
            sha256_file(REFERENCE), FIXTURE_SHA, context.sha256, configuration_sha,
            "fp32", "cpu", seed, FIXTURE_TIMESTAMP, git_commit(),
        ))
    cache.append(records)
    validation = cache.validate(expected_keys=expected)
    state_after = stable_hash([
        module_sha256(module) for module in (model, child, positions, head)
    ])
    if state_before != state_after:
        raise RuntimeError("model state changed during gain collection")
    valid_gains = [item.gain for item in evaluations if item.anchor_valid]
    result = {
        "label": "SYNTHETIC DIAGNOSTIC NON-REPORTABLE", "cache_id": cache.cache_id,
        "cache_path": str(cache.root), "batch_context_hash": context.sha256,
        "record_count": len(records), "sample_count": 3, "candidate_count": len(candidates),
        "gain_minimum": min(valid_gains), "gain_maximum": max(valid_gains),
        "gain_mean": statistics.fmean(valid_gains),
        "gain_standard_deviation": statistics.pstdev(valid_gains),
        "positive_gain_count": sum(x > 0 for x in valid_gains),
        "negative_gain_count": sum(x < 0 for x in valid_gains),
        "zero_gain_count": sum(x == 0 for x in valid_gains),
        "invalid_anchor_count": sum(not x.anchor_valid for x in evaluations),
        "maximum_repeated_evaluation_error": repeated_error,
        "maximum_chunk_equivalence_error": chunk_error,
        "model_state_hash_before": state_before, "model_state_hash_after": state_after,
        "model_state_equal": state_before == state_after,
        "shard_sha256_values": [x["sha256"] for x in cache._index()],
        "deterministic_validation_hash": validation["deterministic_validation_sha256"],
        "processed_candidates": max(0, len(records) - existing),
        "skipped_existing_candidates": existing,
        "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": 0,
    }
    Path(cache.root / "fixture_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
