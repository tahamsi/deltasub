from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import torch

from .distillation import SubViTRouter, SubViTRouterConfig, distillation_loss
from ...utils.checkpointing import atomic_torch_save, load_checkpoint
from ...utils.hashing import stable_hash


def module_hash(module: torch.nn.Module) -> str:
    return stable_hash({k: v.detach().float().cpu().tolist() for k, v in sorted(module.state_dict().items())})


def _rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def _restore(value):
    random.setstate(value["python"]); np.random.set_state(value["numpy"]); torch.set_rng_state(value["torch"])


def _checkpoint_digest(value: dict) -> str:
    return stable_hash({
        "schema_version": value["schema_version"], "critical_hash": value["critical_hash"],
        "epoch": value["epoch"], "router_hash": value["router_hash"],
        "frozen_teacher_hash": value["frozen_teacher_hash"],
        "router_state": {k: stable_hash(v.detach().cpu().tolist())
                         for k, v in sorted(value["router_state"].items())},
    })


def train_router_fixture(
    parents: torch.Tensor, teacher_maps: torch.Tensor, output: str | Path, *,
    seed: int = 7, epochs: int = 2, resume: bool = False, k: int = 4,
) -> dict:
    if parents.shape[:2] != (teacher_maps.shape[0], 256):
        raise ValueError("fixture shapes mismatch")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    path = root / "checkpoint_last.pt"
    config = SubViTRouterConfig(parents.shape[-1], 16, test_only=True)
    router = SubViTRouter(config, seed=seed)
    deterministic_initial = module_hash(router)
    optimizer = torch.optim.SGD(router.parameters(), lr=.05)
    start = 0
    critical = stable_hash({"seed": seed, "k": k, "shape": list(parents.shape),
                            "parents": stable_hash(parents.tolist()), "maps": stable_hash(teacher_maps.tolist())})
    if resume:
        value = load_checkpoint(path)
        if value.get("checkpoint_content_hash") != _checkpoint_digest(value):
            raise ValueError("corrupt M7 checkpoint")
        if value.get("critical_hash") != critical:
            raise ValueError("incompatible or corrupt M7 resume")
        router.load_state_dict(value["router_state"]); optimizer.load_state_dict(value["optimizer_state"])
        _restore(value["rng_state"]); start = int(value["epoch"])
    frozen_before = stable_hash({"parents": parents.tolist(), "maps": teacher_maps.tolist()})
    last = None
    for epoch in range(start, epochs):
        optimizer.zero_grad()
        last = distillation_loss(router(parents), teacher_maps, k)
        last.total.backward(); optimizer.step()
        checkpoint = {"schema_version": 1, "critical_hash": critical, "epoch": epoch + 1,
                      "router_state": router.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "rng_state": _rng(), "router_hash": module_hash(router),
                      "frozen_teacher_hash": stable_hash(teacher_maps.tolist())}
        checkpoint["checkpoint_content_hash"] = _checkpoint_digest(checkpoint)
        atomic_torch_save(checkpoint, path)
    with torch.no_grad():
        last = distillation_loss(router(parents), teacher_maps, k)
    frozen_after = stable_hash({"parents": parents.tolist(), "maps": teacher_maps.tolist()})
    return {
        "label": "SYNTHETIC DIAGNOSTIC NON-REPORTABLE", "non_reportable": True,
        "epoch": epochs, "router_hash": module_hash(router), "initial_router_hash": deterministic_initial,
        "map_kl_loss": float(last.map_kl.detach()), "ranking_loss": float(last.ranking.detach()),
        "topk_mask_loss": float(last.topk_mask.detach()), "total_loss": float(last.total.detach()),
        "pair_count": last.pair_count, "frozen_teacher_hash_before": frozen_before,
        "frozen_teacher_hash_after": frozen_after, "frozen_teacher_equal": frozen_before == frozen_after,
        "checkpoint": str(path),
    }


def inspect_checkpoint(path: str | Path) -> dict:
    value = load_checkpoint(path)
    required = {"schema_version", "critical_hash", "epoch", "router_hash",
                "frozen_teacher_hash", "checkpoint_content_hash"}
    if required - set(value):
        raise ValueError("M7 checkpoint missing required fields")
    if value["checkpoint_content_hash"] != _checkpoint_digest(value):
        raise ValueError("M7 checkpoint corruption detected")
    return {k: value[k] for k in required}
