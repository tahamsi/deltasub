from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Iterable
from dataclasses import replace

import numpy as np
import torch

from .config import load_router_config
from .features import RouterFeatureBatch, join_features
from .losses import router_loss
from .metrics import evaluate_metrics
from .model import GainRouter, RouterConfig
from .replay import DeterministicReplayBuffer
from .sampling import plan_epoch, sampling_summary
from .schema import RouterExample
from .splits import create_split_manifest, require_matching_manifest
from ..gains.cache import open_cache
from ..gains.counterfactual import module_sha256
from ..utils.checkpointing import atomic_torch_save, load_checkpoint
from ..utils.hashing import sha256_file, stable_hash
from ..utils.reproducibility import seed_everything


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def validate_router_cache(path: str | Path, expected_cache_id: str | None = None):
    cache = open_cache(path)
    validation = cache.validate()
    if not validation["complete"]:
        raise ValueError("M4 gain cache is incomplete")
    if expected_cache_id and cache.cache_id != expected_cache_id:
        raise ValueError("M4 cache ID mismatch")
    return cache, validation


def _state_hash(value) -> str:
    return stable_hash({
        key: stable_hash(tensor.detach().cpu().numpy().tobytes().hex())
        for key, tensor in sorted(value.state_dict().items())
    })


def inspect_router_checkpoint(path: str | Path) -> dict:
    value = load_checkpoint(path, map_location="cpu")
    required = {
        "schema_version", "router_state", "optimizer_state", "scheduler_state", "epoch",
        "global_step", "best_metric", "router_configuration_hash", "m4_cache_id",
        "m4_metadata_hash", "m4_validation_hash", "feature_source_hash",
        "split_manifest_hash", "replay_buffer_hash", "dinov2_checkpoint_hash",
        "dinov2_source_revision", "git_commit", "training_device", "precision",
        "rng_states", "trainable_parameter_count",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
    return {key: value[key] for key in required if key not in {"router_state", "optimizer_state",
                                                               "scheduler_state", "rng_states"}}


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(value: dict) -> None:
    random.setstate(value["python"]); np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])


def train_from_cache(
    config_path: str | Path, feature_batches: Iterable[RouterFeatureBatch], *,
    resume: bool = False,
) -> dict:
    config = load_router_config(config_path)
    cache, validation = validate_router_cache(config["gain_cache_path"], config["expected_cache_id"] or None)
    return train_records(config, list(cache.iter_records()), list(feature_batches),
                         cache, validation, resume=resume)


def train_records(config: dict, records, feature_batches, cache, validation, *, resume=False) -> dict:
    started = time.perf_counter()
    seed_everything(config["seed"])
    output = Path(config["checkpoint_directory"]); output.mkdir(parents=True, exist_ok=True)
    manifest = create_split_manifest(
        records, train_fraction=config["splits"]["train_fraction"],
        validation_fraction=config["splits"]["validation_fraction"],
        test_fraction=config["splits"]["test_fraction"], seed=config["splits"]["seed"],
    )
    manifest_path = output / "split_manifest.json"
    if manifest_path.exists():
        require_matching_manifest(manifest, json.loads(manifest_path.read_text()))
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    joined = join_features(records, feature_batches)
    batch_by_context = {(b.batch_context_sha256, sample): b for b in feature_batches for sample in b.sample_ids}
    examples = []
    for record in records:
        source = batch_by_context[(record.batch_context_sha256, record.sample_id)].source_hash
        examples.append(RouterExample.from_gain(
            record, cache_id=cache.cache_id, feature_source_hash=source,
            router_feature_hash=stable_hash(joined[record.key].cpu().tolist()),
            split=manifest.assignments[record.sample_id],
        ))
    invalid_count = sum(not x.anchor_valid for x in examples)
    train = [x for x in examples if x.split_assignment == "train" and x.anchor_valid]
    validation_examples = [x for x in examples if x.split_assignment == "validation" and x.anchor_valid]
    feature_dim = config["router"]["input_dim"]
    model_config = RouterConfig(**config["router"], test_only=config["test_only"])
    model = GainRouter(model_config, seed=config["seed"]).to(config["training"]["device"])
    device = torch.device(config["training"]["device"])
    optimizer_class = torch.optim.AdamW if config["training"]["optimizer"] == "adamw" else torch.optim.SGD
    optimizer = optimizer_class(model.parameters(), lr=config["training"]["learning_rate"],
                                weight_decay=config["training"]["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["training"]["epochs"])
                 if config["training"]["scheduler"] == "cosine" else None)
    replay_config_hash = stable_hash(config["replay"])
    replay = DeterministicReplayBuffer(config["replay"]["capacity"], cache.cache_id,
                                       replay_config_hash, mode=config["replay"]["priority_mode"],
                                       near_zero_epsilon=config["replay"]["near_zero_epsilon"])
    epoch_start = global_step = 0
    best = -float("inf")
    feature_source_hash = stable_hash(sorted({b.source_hash for b in feature_batches}))
    critical = {
        "router_configuration_hash": model.configuration_hash, "m4_cache_id": cache.cache_id,
        "m4_metadata_hash": cache.metadata_sha256,
        "m4_validation_hash": validation["deterministic_validation_sha256"],
        "feature_source_hash": feature_source_hash, "split_manifest_hash": manifest.sha256,
        "dinov2_checkpoint_hash": records[0].model_checkpoint_sha256,
        "dinov2_source_revision": records[0].dinov2_source_revision,
        "training_device": str(device), "precision": config["training"]["precision"],
    }
    last_path = output / "checkpoint_last.pt"
    # --resume is idempotent: restore an existing run or start cleanly.
    if resume and last_path.is_file():
        checkpoint = load_checkpoint(last_path, map_location=device)
        for key, expected in critical.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"incompatible resume: {key}")
        model.load_state_dict(checkpoint["router_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        replay.load_state_dict(checkpoint["replay_state"])
        epoch_start, global_step, best = checkpoint["epoch"], checkpoint["global_step"], checkpoint["best_metric"]
        _restore_rng(checkpoint["rng_states"])
    initial_state = _state_hash(model)
    frozen_before = stable_hash({"cache": validation["record_content_sha256"],
                                 "features": [stable_hash(b.parent_embeddings.cpu().tolist()) for b in feature_batches]})
    environment = {"python": os.sys.version, "torch": torch.__version__, "device": str(device),
                   "cuda_available": torch.cuda.is_available()}
    (output / "environment.json").write_text(json.dumps(environment, indent=2, default=str) + "\n")
    import yaml
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    metrics_path = output / "metrics.jsonl"

    def tensors(items):
        parent = torch.stack([
            batch_by_context[(x.batch_context_sha256, x.sample_id)].parent_embeddings[
                batch_by_context[(x.batch_context_sha256, x.sample_id)].sample_ids.index(x.sample_id)
            ] for x in items
        ]).to(device)
        indices = torch.tensor([x.candidate_parent_index for x in items], device=device)
        targets = torch.tensor([x.target_gain for x in items], device=device)
        groups_map = {key: index for index, key in enumerate(sorted({(x.batch_context_sha256, x.sample_id) for x in items}))}
        groups = torch.tensor([groups_map[(x.batch_context_sha256, x.sample_id)] for x in items], device=device)
        return parent, indices, targets, groups

    def evaluate(items):
        model.eval(); scores = []
        with torch.no_grad():
            for start in range(0, len(items), config["training"]["physical_batch_size"]):
                chunk = items[start:start + config["training"]["physical_batch_size"]]
                parent, indices, _, _ = tensors(chunk)
                values = model(parent)
                scores.extend(values[torch.arange(len(chunk), device=device), indices].float().cpu().tolist())
        model.train()
        return evaluate_metrics(scores, [x.target_gain for x in items],
                                [f"{x.batch_context_sha256}:{x.sample_id}" for x in items],
                                labelled=[x.labelled for x in items],
                                known_or_novel=[x.known_or_novel for x in items],
                                parents=[x.candidate_parent_index for x in items],
                                invalid_anchor_count=invalid_count)

    initial_metrics = evaluate(validation_examples)
    last_loss = None; last_summary = {}; replay_used = 0
    for epoch in range(epoch_start, config["training"]["epochs"]):
        plan = plan_epoch(train, batch_size=config["training"]["physical_batch_size"],
                          informative_fraction=config["sampling"]["informative_fraction"],
                          seed=config["seed"], epoch=epoch)
        last_summary = sampling_summary(plan)
        optimizer.zero_grad(set_to_none=True)
        train_by_key = {x.gain_record_key: x for x in train}
        for batch_index, planned_items in enumerate(plan.batches):
            items = list(planned_items)
            if epoch > 0 and replay.entries and config["replay"]["ratio"] > 0:
                selected = {x.gain_record_key for x in items}
                candidates = sorted(
                    (entry for entry in replay.entries.values()
                     if entry.gain_record_key not in selected),
                    key=lambda x: (-x.priority, x.insertion_order, x.gain_record_key),
                )
                if candidates and items:
                    items[-1] = replace(train_by_key[candidates[0].gain_record_key],
                                        sampling_stream="replay")
                    replay_used += 1
            parent, indices, targets, groups = tensors(items)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16,
                                      enabled=device.type == "cuda" and config["training"]["precision"] == "bf16")
            with autocast:
                all_scores = model(parent)
                scores = all_scores[torch.arange(len(items), device=device), indices]
                loss = router_loss(
                    scores, targets, groups, torch.ones_like(targets, dtype=torch.bool),
                    regression_weight=config["loss"]["regression_weight"],
                    ranking_weight=config["loss"]["ranking_weight"],
                    sign_weight=config["loss"]["sign_weight"],
                    huber_delta=config["loss"]["huber_delta"],
                    target_margin=config["loss"]["ranking_margin"],
                    maximum_pairs_per_anchor=config["loss"]["maximum_pairs_per_anchor"],
                    sign_epsilon=config["loss"]["sign_epsilon"],
                    gain_clip=config["loss"]["gain_clip"],
                )
                scaled = loss.total / config["training"]["gradient_accumulation"]
            scaled.backward()
            if (batch_index + 1) % config["training"]["gradient_accumulation"] == 0 or batch_index + 1 == len(plan.batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clipping"])
                optimizer.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            for item, residual in zip(items, (scores.detach().float() - targets.float()).abs().cpu().tolist()):
                if item.gain_record_key not in replay.entries:
                    replay.add(item, residual)
                else:
                    replay.update_priority(item.gain_record_key, residual)
            last_loss = loss
        if scheduler:
            scheduler.step()
        validation_metrics = evaluate(validation_examples)
        metric_value = validation_metrics["spearman"]["value"]
        if metric_value is None:
            metric_value = validation_metrics["pairwise_ranking_accuracy"]["value"]
        metric_value = -float("inf") if metric_value is None else metric_value
        record = {
            "epoch": epoch + 1, "global_step": global_step,
            "regression_loss": float(last_loss.regression.detach()),
            "ranking_loss": float(last_loss.ranking.detach()),
            "sign_loss": float(last_loss.sign.detach()),
            "total_loss": float(last_loss.total.detach()),
            "validation": validation_metrics, **last_summary,
            "replay_examples": len(replay.entries), "replay_examples_used": replay_used,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        checkpoint = {
            "schema_version": 1, "router_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "epoch": epoch + 1, "global_step": global_step, "best_metric": max(best, metric_value),
            **critical, "replay_buffer_hash": replay.checksum, "replay_state": replay.state_dict(),
            "git_commit": _git_commit(), "rng_states": _rng_state(),
            "trainable_parameter_count": model.parameter_count,
            "model_state_hash": _state_hash(model), "optimizer_state_hash": stable_hash(str(optimizer.state_dict())),
        }
        atomic_torch_save(checkpoint, last_path)
        if metric_value > best:
            best = metric_value
            atomic_torch_save(checkpoint, output / "checkpoint_best.pt")
        (output / "replay_state.json").write_text(json.dumps(replay.state_dict(), indent=2, sort_keys=True) + "\n")
    final_metrics = evaluate(validation_examples)
    frozen_after = stable_hash({"cache": validation["record_content_sha256"],
                                "features": [stable_hash(b.parent_embeddings.cpu().tolist()) for b in feature_batches]})
    parameter_change = 0.
    checkpoint = load_checkpoint(last_path)
    if initial_state != checkpoint["model_state_hash"]:
        # Exact norm is computed against a fresh deterministic initial model only for non-resume runs.
        reference = GainRouter(model_config, seed=config["seed"])
        parameter_change = float(torch.sqrt(sum(
            ((model.state_dict()[k].detach().cpu() - v.detach().cpu()) ** 2).sum()
            for k, v in reference.state_dict().items()
        )))
    return {
        "label": "SYNTHETIC DIAGNOSTIC NON-REPORTABLE" if config["test_only"] else "M5 ROUTER TRAINING",
        "non_reportable": bool(config["test_only"]), "gain_cache_id": cache.cache_id,
        "split_manifest_hash": manifest.sha256,
        "train_sample_count": manifest.summary["sample_counts"].get("train", 0),
        "validation_sample_count": manifest.summary["sample_counts"].get("validation", 0),
        "train_record_count": len(train), "validation_record_count": len(validation_examples),
        "router_parameter_count": model.parameter_count,
        "router_multiply_add_estimate": model.multiply_add_estimate(),
        "initial_validation_metrics": initial_metrics, "final_validation_metrics": final_metrics,
        "best_validation_metric": best,
        "regression_loss": float(last_loss.regression.detach()),
        "ranking_loss": float(last_loss.ranking.detach()),
        "sign_loss": float(last_loss.sign.detach()),
        **last_summary, "replay_examples": len(replay.entries),
        "replay_examples_used": replay_used, "replay_buffer_hash": replay.checksum,
        "parameter_change_norm": parameter_change, "frozen_state_hash_before": frozen_before,
        "frozen_state_hash_after": frozen_after, "frozen_state_equal": frozen_before == frozen_after,
        "global_step": global_step, "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "no_adaptive_budget_executed": True, "no_token_selection_executed": True,
    }
