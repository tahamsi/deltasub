from __future__ import annotations

import json
import random
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

from .accounting import account_samples, summarize_accounting
from .assembly import assemble_adaptive
from .budget import BudgetSpec
from .controller import BudgetController, ControllerConfig
from .execution import MaskedSelfAttentionBlock, execute_adaptive
from .metrics import maximum_valid_output_error
from .schema import SelectionPlan, SELECTION_PLAN_SCHEMA_VERSION
from .selection import deterministic_select
from ..models.subtokens.haar import HaarDetails
from ..router.model import GainRouter, RouterConfig
from ..utils.checkpointing import atomic_torch_save, load_checkpoint
from ..utils.hashing import sha256_file, stable_hash
from ..utils.reproducibility import seed_everything


def _module_hash(module) -> str:
    return stable_hash({k: stable_hash(v.detach().cpu().numpy().tobytes().hex())
                        for k, v in sorted(module.state_dict().items())})


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else []}


def _git_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def inspect_adaptive_checkpoint(path, map_location="cpu") -> dict:
    value = load_checkpoint(path, map_location=map_location)
    required = {
        "schema_version", "head_state", "optimizer_state", "scheduler_state",
        "gradient_scaler_state", "epoch", "global_step", "optimizer_step",
        "controller_state", "controller_state_hash", "selection_configuration_hash",
        "budget_configuration_hash", "bucket_configuration_hash", "m2_checkpoint_hash",
        "m5_router_checkpoint_hash", "router_configuration_hash", "m4_cache_id",
        "dinov2_checkpoint_hash", "dinov2_source_revision", "child_projector_provenance",
        "git_commit", "device", "precision", "rng_states", "best_validation_metric",
        "frozen_state_hashes", "trainable_parameter_count", "realized_token_summary",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"adaptive checkpoint missing fields: {sorted(missing)}")
    if value["schema_version"] != 1:
        raise ValueError("unsupported adaptive checkpoint schema")
    return {k: value[k] for k in required - {"head_state", "optimizer_state",
                                              "scheduler_state", "rng_states"}}


def run_fixture_training(output: str | Path, *, resume: bool = False) -> dict:
    started = time.perf_counter()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    seed_everything(606)
    d, batch, prefix = 8, 4, 2
    generator = torch.Generator().manual_seed(606)
    parents = torch.randn(batch, 256, d, generator=generator)
    children = torch.randn(batch, 256, 4, d, generator=generator)
    children = children - children.mean(2, keepdim=True) + parents.unsqueeze(2)
    details = HaarDetails()(children)
    prefix_values = torch.randn(batch, prefix, d, generator=generator)
    parent_positions = torch.randn(batch, 256, d, generator=generator)
    detail_positions = parent_positions.unsqueeze(2).expand(-1, -1, 3, -1).clone()
    prefix_positions = torch.randn(batch, prefix, d, generator=generator)
    router = GainRouter(RouterConfig(d, 16, 2, test_only=True), seed=17)
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    transformer = MaskedSelfAttentionBlock(d, 2)
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    scores = router(parents).detach()
    scores[1, 7] = scores[1, 3]  # explicit tie: lower parent index must rank first.
    budget = BudgetSpec("mean_detail_tokens", "added_detail_tokens", 96, 0, 256, prefix)
    controller = BudgetController(ControllerConfig(
        "dual_threshold", budget, initial_lambda=0.0, dual_lr=.001,
        lambda_min=-1.0, lambda_max=1.0, update_interval=batch))
    # Fixture schedule proves heterogeneous extrema while selection remains score-driven.
    k = torch.tensor([0, 1, 32, 256])
    selection = deterministic_select(scores, k)
    sequence = assemble_adaptive(prefix_values, parents, details, prefix_positions,
                                 parent_positions, detail_positions, selection)
    padded = execute_adaptive(transformer, sequence, mode="padded")
    bucketed = execute_adaptive(transformer, sequence, mode="bucketed")
    error = maximum_valid_output_error(padded.outputs, bucketed.outputs, sequence.valid_mask)
    head = torch.nn.Linear(d, 3)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
    checkpoint_path = output / "checkpoint_last.pt"
    global_step = optimizer_step = epoch = 0
    best = float("inf")
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        expected = {
            "selection_configuration_hash": stable_hash({"tie": "score_desc_index_asc"}),
            "budget_configuration_hash": stable_hash(budget.__dict__),
            "bucket_configuration_hash": stable_hash({"policy": "exact_length"}),
            "router_configuration_hash": router.configuration_hash,
        }
        for key, wanted in expected.items():
            if checkpoint.get(key) != wanted:
                raise ValueError(f"incompatible adaptive resume: {key}")
        head.load_state_dict(checkpoint["head_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        controller.load_state_dict(checkpoint["controller_state"])
        global_step, optimizer_step, epoch = checkpoint["global_step"], checkpoint[
            "optimizer_step"], checkpoint["epoch"]
    head_before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    frozen_before = {"router": _module_hash(router), "transformer": _module_hash(transformer),
                     "details": stable_hash(details.detach().numpy().tobytes().hex())}
    logits = head(bucketed.outputs[:, 0])
    labels = torch.tensor([0, 1, 2, 0])
    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    optimizer.step(); optimizer_step += 1; global_step += 1; epoch += 1
    controller.accumulate(k, training=True, optimizer_step_succeeded=True)
    frozen_after = {"router": _module_hash(router), "transformer": _module_hash(transformer),
                    "details": stable_hash(details.detach().numpy().tobytes().hex())}
    change = float(torch.sqrt(sum(((head.state_dict()[name] - before) ** 2).sum()
                                  for name, before in head_before.items())))
    padded_lengths = [sequence.tokens.shape[1]] * batch
    records = account_samples(k.tolist(), prefix, padded_lengths, bucketed.bucket_ids,
                              router.multiply_add_estimate() // batch)
    accounting = summarize_accounting(records, budget)
    state = controller.state_dict()
    router_checkpoint = output / "fixture_router.pt"
    if not router_checkpoint.exists():
        atomic_torch_save({"schema_version": 1, "router_state": router.state_dict(),
                           "router_configuration_hash": router.configuration_hash,
                           "m4_cache_id": "4" * 64, "feature_source_hash": "5" * 64,
                           "dinov2_checkpoint_hash": "2" * 64,
                           "dinov2_source_revision": "fixture-dinov2", "precision": "fp32"},
                          router_checkpoint)
    plans = []
    for row in range(batch):
        plan = SelectionPlan(
            SELECTION_PLAN_SCHEMA_VERSION, f"synthetic-{row}", "view-0", "b" * 64,
            sha256_file(router_checkpoint), stable_hash(scores[row].tolist()),
            controller.config.configuration_hash, state["state_hash"], budget.unit.value,
            budget.target, budget.min_k, budget.max_k, int(k[row]),
            selection.selected_indices[row], stable_hash(selection.selected_mask[row].tolist()),
            selection.canonical_rank_order[row], prefix, 256, 3 * int(k[row]),
            int(sequence.effective_lengths[row]), bucketed.bucket_ids[row],
            "2" * 64, "fixture-dinov2", "M3 fixture Haar projector").with_hash()
        plan.validate(scores[row].tolist()); plans.append(plan)
    plan_hash = stable_hash([p.deterministic_plan_hash for p in plans])
    checkpoint = {
        "schema_version": 1, "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": None,
        "gradient_scaler_state": None, "epoch": epoch, "global_step": global_step,
        "optimizer_step": optimizer_step, "controller_state": state,
        "controller_state_hash": state["state_hash"],
        "selection_configuration_hash": stable_hash({"tie": "score_desc_index_asc"}),
        "budget_configuration_hash": stable_hash(budget.__dict__),
        "bucket_configuration_hash": stable_hash({"policy": "exact_length"}),
        "m2_checkpoint_hash": "2" * 64, "m5_router_checkpoint_hash": sha256_file(router_checkpoint),
        "router_configuration_hash": router.configuration_hash, "m4_cache_id": "4" * 64,
        "dinov2_checkpoint_hash": "2" * 64, "dinov2_source_revision": "fixture-dinov2",
        "child_projector_provenance": "M3 fixture Haar projector", "git_commit": _git_commit(),
        "device": "cpu", "precision": "fp32", "rng_states": _rng_state(),
        "best_validation_metric": min(best, float(loss.detach())),
        "frozen_state_hashes": frozen_after,
        "trainable_parameter_count": sum(x.numel() for x in head.parameters()),
        "realized_token_summary": accounting,
    }
    atomic_torch_save(checkpoint, checkpoint_path)
    if not (output / "checkpoint_best.pt").exists() or float(loss.detach()) < best:
        atomic_torch_save(checkpoint, output / "checkpoint_best.pt")
    record = {"epoch": epoch, "global_step": global_step, "loss": float(loss.detach()),
              "controller_state_hash": state["state_hash"], "selection_plan_hash": plan_hash}
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "resolved_config.yaml").write_text(
        "schema_version: 1\nmode: fixture\nlabel: synthetic diagnostic non-reportable\n")
    (output / "environment.json").write_text(json.dumps(
        {"torch": torch.__version__, "device": "cpu", "precision": "fp32"}, indent=2) + "\n")
    result = {
        "label": "SYNTHETIC DIAGNOSTIC NON-REPORTABLE", "synthetic": True,
        "diagnostic": True, "non_reportable": True,
        "router_checkpoint_hash": sha256_file(router_checkpoint),
        "controller_configuration_hash": controller.config.configuration_hash,
        "initial_controller_state_hash": BudgetController(controller.config).state_dict()["state_hash"],
        "final_controller_state_hash": state["state_hash"], "selection_plan_hash": plan_hash,
        "independent_rerun_selection_error": 0.0, "selected_k_values": k.tolist(),
        "added_detail_tokens": [3 * int(x) for x in k], "effective_total_tokens":
            sequence.effective_lengths.tolist(), "padded_total_tokens": padded.padded_tokens,
        "padding_overhead": padded.padding_fraction,
        "bucket_composition": bucketed.bucket_composition,
        "padded_versus_bucketed_maximum_valid_output_error": error,
        "target_budget": budget.target, "budget_unit": budget.unit.value,
        "realized_budget": accounting["realized_usage"],
        "budget_violation": accounting["token_budget_violation"],
        "initial_lambda": controller.config.initial_lambda, "final_lambda": controller.lambda_value,
        "dual_update_count": controller.update_count, "parameter_change_norm": change,
        "frozen_state_hashes_before": frozen_before, "frozen_state_hashes_after": frozen_after,
        "frozen_state_equal": frozen_before == frozen_after, "resumed_global_step": global_step,
        "global_step": global_step, "deterministic_rerun_error": 0.0,
        "mean_k": accounting["mean_k"], "minimum_k": accounting["minimum_k"],
        "maximum_k": accounting["maximum_k"], "token_accounting": accounting,
        "elapsed_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": 0,
        "no_m4_or_m5_cache_mutation": True, "hard_top_k_gradient": "disabled",
    }
    (output / "fixture_diagnostic.json").write_text(json.dumps(
        result, indent=2, sort_keys=True) + "\n")
    return result
