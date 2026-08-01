from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import time

import torch
from torch import nn

from .cache import GainCache
from .config import load_gain_config
from .counterfactual import BatchContext, CounterfactualGainEvaluator, module_sha256
from .fixture import run_fixture_collection
from .schema import GAIN_SCHEMA_VERSION, GainRecord
from ..data.manifests import read_manifest
from ..models.backbones.dinov2 import DINOv2Adapter
from ..models.subtokens.positions import ParentAwareDetailPositions
from ..training.baseline import ManifestImageDataset
from ..router.features import RouterFeatureBatch, save_feature_cache
from ..training.selex_equivalence import PRODUCTION, REFERENCE, recompute_fp32_equivalence
from ..utils.hashing import sha256_file, stable_hash


def _verified_file(path: str, expected: str, label: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    if sha256_file(value) != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return value


def validate_collection_config(config_path: str | Path, checkpoint_override=None,
                               expected_override=None, source_override=None) -> dict:
    config, plan = load_gain_config(config_path)
    dataset = config["dataset"]
    backbone = config["backbone"]
    selex = config["selex"]
    subtokens = config["subtokens"]
    if config["test_only"]:
        # Fixture pseudo-paths are provenance labels; exact fixed hashes make this explicit.
        if any(value != "f" * 64 for value in (
            dataset["manifest_sha256"], dataset["split_report_sha256"],
            backbone["checkpoint_sha256"], selex["equivalence_gate_sha256"],
        )):
            raise ValueError("test-only fixture provenance must use explicit fixture hashes")
    else:
        _verified_file(dataset["manifest"], dataset["manifest_sha256"], "dataset manifest")
        split = _verified_file(dataset["split_report"], dataset["split_report_sha256"], "split report")
        report = json.loads(split.read_text(encoding="utf-8"))
        if report.get("validation_outcome") != "passed":
            raise ValueError("dataset split report did not pass")
        gate = _verified_file(selex["equivalence_gate"], selex["equivalence_gate_sha256"], "SelEx gate")
        gate_value = json.loads(gate.read_text(encoding="utf-8"))
        if gate_value.get("status") != "passed":
            raise ValueError("SelEx equivalence gate did not pass")
        recomputed = recompute_fp32_equivalence()
        if recomputed["max_absolute_error"] > gate_value["thresholds"]["fp32_atol"]:
            raise ValueError("SelEx equivalence re-execution failed")
        _verified_file(subtokens["config"], subtokens["config_sha256"], "M3 subtoken config")
        checkpoint = checkpoint_override or backbone["checkpoint"]
        expected = expected_override or backbone["checkpoint_sha256"]
        source = source_override or backbone["source_root"]
        adapter = DINOv2Adapter.from_official_checkpoint(
            checkpoint, expected, source_root=source
        )
        adapter.requires_grad_(False).eval()
        if adapter.parameter_report()["trainable"] != 0:
            raise RuntimeError("production DINOv2 did not freeze")
    return {
        "status": "validated", "test_only": config["test_only"],
        "candidate_count": len(plan.parents), "candidate_subset": list(plan.parents),
        "device": config["collection"]["device"],
        "no_optimizer": True, "no_training": True,
    }


def collect(config_path: str | Path, *, checkpoint=None, expected_sha256=None,
            source_root=None, resume=False, validate_only=False) -> dict:
    config, _ = load_gain_config(config_path)
    validation = validate_collection_config(
        config_path, checkpoint, expected_sha256, source_root
    )
    if validate_only:
        return validation
    if config["test_only"]:
        candidate = config["collection"]["candidate"]
        from .planning import CandidatePlan
        plan = CandidatePlan.from_config(candidate)
        return run_fixture_collection(
            config["output_root"], resume=resume, candidates=plan.parents,
            seed=config["collection"]["seed"],
        )
    return _collect_production(
        config_path, config, checkpoint=checkpoint, expected_sha256=expected_sha256,
        source_root=source_root, resume=resume,
    )


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def _collect_production(config_path, config, *, checkpoint, expected_sha256, source_root,
                        resume):
    collection, backbone, dataset_config = (
        config["collection"], config["backbone"], config["dataset"]
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("production M4 requires exactly one visible CUDA device at cuda:0")
    device = torch.device("cuda:0")
    checkpoint_path = checkpoint or backbone["checkpoint"]
    checkpoint_sha = expected_sha256 or backbone["checkpoint_sha256"]
    source = source_root or backbone["source_root"]
    adapter = DINOv2Adapter.from_official_checkpoint(
        checkpoint_path, checkpoint_sha, source_root=source
    ).to(device).eval()
    adapter.requires_grad_(False)
    child = adapter.build_child_projector(trainable=False).to(device).eval()
    positions = ParentAwareDetailPositions().to(device).eval()
    positions.requires_grad_(False)
    head = nn.Identity().to(device).eval()  # Exact M2 baseline feature contract is CLS.
    precision = collection["precision"]
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    if dtype == torch.bfloat16:
        adapter.to(dtype=dtype)
        child.to(dtype=dtype)
        positions.to(dtype=dtype)

    def encode(sequence):
        value = sequence.unsqueeze(0)
        for block in adapter.model.blocks:
            value = block(value)
        return adapter.model.norm(value)[0, 0]

    evaluator = CounterfactualGainEvaluator(adapter, child, positions, head, encode)
    records = read_manifest(dataset_config["manifest"])
    image_dataset = ManifestImageDataset(
        records, dataset_config.get("root") or Path(dataset_config["manifest"]).parent,
        train=True, seed=int(collection["seed"])
    )
    records = image_dataset.records
    batch_size = int(collection["batch_size"])
    plan = load_gain_config(config_path)[1]
    if plan.sample_limit is not None:
        records = records[:plan.sample_limit]
        image_dataset.records = records
    batches = [
        list(range(start, min(start + batch_size, len(records))))
        for start in range(0, len(records), batch_size)
    ]
    if plan.batch_limit is not None:
        batches = batches[:plan.batch_limit]
    if not batches:
        raise ValueError("collection plan selected no samples")
    configuration_sha = stable_hash(config)
    git = _git_commit()
    contexts = []
    feature_batches = []
    materialized = []
    for batch_number, indices in enumerate(batches):
        items = [image_dataset[index] for index in indices]
        images = torch.stack([item["views"] for item in items]).to(device=device, dtype=dtype)
        labels = torch.tensor([item["class_id"] for item in items], device=device)
        labelled = torch.tensor([item["labelled"] for item in items], device=device)
        pseudo = torch.tensor([item["pseudo_label"] for item in items], device=device)
        confidence = torch.ones(len(items), device=device, dtype=dtype)
        confusion = torch.eye(2 * len(items), device=device, dtype=dtype)
        sample_ids = tuple(records[index]["sample_id"] for index in indices)
        context = BatchContext.build(
            images=images, sample_ids=sample_ids, view_ids=("view-0", "view-1"),
            augmentation_seeds=(
                int(collection["seed"]) + 97 * indices[0],
                int(collection["seed"]) + 97 * indices[0] + 1,
            ),
            augmentation_parameters={
                "implementation": "M2 ManifestImageDataset deterministic resize/flip",
                "batch_number": batch_number, "dataset_indices": indices,
            },
            labels=labels, labelled=labelled, hierarchy_labels=(pseudo,),
            confusion_factor=confusion, pseudo_label_confidence=confidence,
            model=adapter, child_projector=child, position_module=positions, head=head,
            configuration_hash=configuration_sha, precision=precision,
            device=device, source_git_commit=git,
        )
        contexts.append(context)
        with torch.inference_mode():
            parent_features = adapter.pre_transformer_parent_embeddings(
                images.flatten(0, 1)
            ).reshape(len(items), 2, 256, 768).float().mean(1).cpu()
        feature_batches.append(RouterFeatureBatch(
            sample_ids, "two-view-stacked-v1", context.sha256, parent_features,
            checkpoint_sha, adapter.inspection.source_revision,
            dataset_config["manifest_sha256"], configuration_sha,
            module_sha256(child),
        ))
        materialized.append(indices)
    expected = {
        (contexts[b].sha256, contexts[b].sample_ids[anchor], anchor, parent)
        for b in range(len(contexts)) for anchor in range(len(contexts[b].sample_ids))
        for parent in plan.parents
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": GAIN_SCHEMA_VERSION, "dataset_name": dataset_config["name"],
        "synthetic_only": False, "reportable": True,
        "dataset_manifest_sha256": dataset_config["manifest_sha256"],
        "split_report_sha256": dataset_config["split_report_sha256"],
        "model_checkpoint_sha256": checkpoint_sha,
        "dinov2_source_revision": adapter.inspection.source_revision,
        "child_projector_state_sha256": module_sha256(child),
        "selex_implementation_sha256": sha256_file(PRODUCTION),
        "selex_reference_sha256": sha256_file(REFERENCE),
        "selex_equivalence_gate_sha256": config["selex"]["equivalence_gate_sha256"],
        "configuration_sha256": configuration_sha,
        "batch_context_hashes": [context.sha256 for context in contexts],
        "candidate_subset": list(plan.parents),
        "candidate_subset_is_full": plan.parents == tuple(range(256)),
        "precision": precision, "resolved_device": "cuda:0", "seed": collection["seed"],
        "source_git_commit": git, "deterministic_algorithm_exceptions": [],
        "planned_record_count": len(expected),
        "planned_key_sha256": stable_hash(sorted([list(key) for key in expected])),
    }
    cache = GainCache(
        config["output_root"], metadata, shard_size=collection["shard_size"], resume=resume
    )
    state_before = stable_hash([
        module_sha256(module) for module in (adapter, child, positions, head)
    ])
    started = time.perf_counter()
    processed = skipped = invalid = errors = forward_passes = 0
    torch.cuda.reset_peak_memory_stats(device)
    for batch_number, indices in enumerate(materialized):
        items = [image_dataset[index] for index in indices]
        images = torch.stack([item["views"] for item in items]).to(device=device, dtype=dtype)
        labels = torch.tensor([item["class_id"] for item in items], device=device)
        labelled = torch.tensor([item["labelled"] for item in items], device=device)
        hierarchy = (torch.tensor([item["pseudo_label"] for item in items], device=device),)
        confidence = torch.ones(len(items), device=device, dtype=dtype)
        confusion = torch.eye(2 * len(items), device=device, dtype=dtype)
        context = contexts[batch_number]
        base_cache = {}
        for anchor, source_index in enumerate(indices):
            source_record = records[source_index]
            for parent in plan.parents:
                key = (context.sha256, source_record["sample_id"], anchor, parent)
                if key in cache.existing_keys:
                    skipped += 1
                    continue
                try:
                    forward_passes += len(indices) * 2 * (1 + int(not base_cache))
                    value = evaluator.evaluate(
                        images=images, labels=labels, labelled=labelled,
                        hierarchy_labels=hierarchy, confusion_factor=confusion,
                        pseudo_label_confidence=confidence, context=context,
                        anchor=anchor, parent=parent, base_cache=base_cache,
                    )
                    gain_record = GainRecord(
                        GAIN_SCHEMA_VERSION, dataset_config["name"],
                        dataset_config["manifest_sha256"], dataset_config["split_report_sha256"],
                        source_record["sample_id"], anchor, "two-view-stacked-v1",
                        parent, parent // 16, parent % 16, bool(labelled[anchor]),
                        source_record.get("known_or_novel"),
                        int(labels[anchor]) if bool(labelled[anchor]) else None,
                        value.base_anchor_loss, value.counterfactual_anchor_loss, value.gain,
                        value.base_batch_loss, value.counterfactual_batch_loss,
                        value.batch_loss_change, value.spillover_sum, value.spillover_max_abs,
                        value.anchor_valid, value.base_effective_token_count,
                        value.counterfactual_effective_token_count,
                        value.base_padded_token_count, value.counterfactual_padded_token_count,
                        checkpoint_sha, adapter.inspection.source_revision,
                        module_sha256(child), sha256_file(PRODUCTION), sha256_file(REFERENCE),
                        config["selex"]["equivalence_gate_sha256"], context.sha256,
                        configuration_sha, precision, "cuda:0", int(collection["seed"]),
                        timestamp, git,
                    )
                    cache.append([gain_record])
                    processed += 1
                    invalid += int(not value.anchor_valid)
                    if processed % int(collection["shard_size"]) == 0:
                        elapsed_now = time.perf_counter() - started
                        remaining_items = len(expected) - len(cache.existing_keys) - len(cache.pending)
                        rate = processed / max(elapsed_now, 1e-12)
                        cache._atomic_text(
                            cache.root / "progress.json",
                            json.dumps({
                                "processed_candidates": processed,
                                "skipped_existing_candidates": skipped,
                                "remaining_candidates": remaining_items,
                                "elapsed_seconds": elapsed_now,
                                "estimated_remaining_seconds": remaining_items / max(rate, 1e-12),
                                "gpu_hours": elapsed_now / 3600,
                                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                                "candidate_throughput_per_second": rate,
                                "forward_pass_count": forward_passes,
                            }, indent=2, sort_keys=True) + "\n",
                        )
                except (FloatingPointError, ValueError):
                    errors += 1
                    raise
    cache.flush()
    feature_cache_path = Path(config["output_root"]) / "router_features.pt"
    feature_cache_hash = save_feature_cache(feature_batches, feature_cache_path)
    validation = cache.validate(expected_keys=expected)
    state_after = stable_hash([
        module_sha256(module) for module in (adapter, child, positions, head)
    ])
    if state_before != state_after:
        raise RuntimeError("model/projector/head state changed during collection")
    elapsed = time.perf_counter() - started
    return {
        "status": "completed", "cache_id": cache.cache_id, "cache_path": str(cache.root),
        "record_count": validation["record_count"], "processed_candidates": processed,
        "skipped_existing_candidates": skipped, "invalid_anchors": invalid,
        "numerical_errors": errors, "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": 0.0, "gpu_hours": elapsed / 3600,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "forward_pass_count": forward_passes,
        "candidate_throughput_per_second": processed / max(elapsed, 1e-12),
        "router_feature_cache": str(feature_cache_path),
        "router_feature_cache_hash": feature_cache_hash,
        "model_state_hash_before": state_before, "model_state_hash_after": state_after,
        "model_state_equal": state_before == state_after,
    }
