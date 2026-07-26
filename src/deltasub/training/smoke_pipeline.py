from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.nn import functional as F

from ..data.datasets import SyntheticGCDDataset
from ..models.deltasub.candidate_sampler import sample_candidates
from ..models.deltasub.detail_score import cheap_detail_features
from ..models.deltasub.gain_collector import collect_tiny_gains, repeated_base_check
from ..models.deltasub.model import TinyDeltaSub
from ..models.deltasub.router import GainRouter, normalized_coordinates
from ..reporting.schema import REQUIRED_RUN_FILES, SCHEMA_VERSION
from ..utils.checkpointing import atomic_torch_save, load_checkpoint
from ..utils.hashing import stable_hash
from ..utils.reproducibility import seed_everything


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def run_smoke_pipeline(
    output: str | Path,
    *,
    size: int = 8,
    seed: int = 0,
    device: str = "cpu",
    resume: bool = False,
) -> dict:
    start = time.time()
    seed_everything(seed)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "checkpoint_last.pt"
    dataset = SyntheticGCDDataset(size=size, seed=seed)
    images, labels = dataset.images.to(device), dataset.labels.to(device)
    model = TinyDeltaSub().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step = 0
    if resume and checkpoint.exists():
        state = load_checkpoint(checkpoint)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]

    model.train()
    logits = model(images[:2])
    loss = F.cross_entropy(logits, labels[:2])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step += 1
    atomic_torch_save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        checkpoint,
    )
    atomic_torch_save({"model": model.state_dict(), "step": step}, target / "checkpoint_best.pt")

    report = repeated_base_check(model, images[:2], labels[:2], 1e-5, 1e-5)
    with torch.no_grad():
        parents = model.parent_tokens(images[:2])
        cheap = cheap_detail_features(images[:2])
        router = GainRouter(parents.shape[-1]).to(device)
        coords = normalized_coordinates(16, device=images.device, dtype=images.dtype)
        scores = router(parents, cheap, coords)
        candidates = sample_candidates(scores, cheap.mean(-1), total=4, random_count=2)
        gain_records = collect_tiny_gains(model, images[:2], labels[:2], candidates.indices)
    gain_frame = pd.DataFrame(gain_records)
    gain_path = target / "selection_statistics.parquet"
    temporary_gain = gain_path.with_suffix(".parquet.tmp")
    gain_frame.to_parquet(temporary_gain, index=False)
    os.replace(temporary_gain, gain_path)

    predictions = model(images).argmax(-1)
    accuracy = float((predictions == labels).float().mean())
    known = dataset.labels < 2
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "synthetic_only": True,
        "all": accuracy,
        "known": float((predictions[known] == labels[known]).float().mean()),
        "novel": float((predictions[~known] == labels[~known]).float().mean()),
        "loss": float(loss.detach()),
    }
    config = {"synthetic": True, "size": size, "seed": seed, "device": device}
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    efficiency = {
        "effective_tokens": 256 + 3 * 4,
        "padded_tokens": 256 + 3 * 4,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
    }
    compute = {
        "elapsed_seconds": time.time() - start,
        "forward_passes": 1 + 2 + len(gain_records) + 1,
        "gpu_hours": (time.time() - start) / 3600 if device.startswith("cuda") else 0.0,
        "status": "synthetic_smoke_complete",
        "determinism": report.__dict__,
    }
    (target / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (target / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    for name, value in (
        ("environment.json", environment),
        ("metrics.json", metrics),
        ("efficiency.json", efficiency),
        ("compute.json", compute),
    ):
        (target / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (target / "metrics.jsonl").write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    (target / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
    (target / "dataset_manifest_checksum.txt").write_text(stable_hash(config) + "\n", encoding="utf-8")
    (target / "backbone_checkpoint_hash.txt").write_text("synthetic-tiny-model\n", encoding="utf-8")
    (target / "stdout.log").write_text("synthetic smoke completed\n", encoding="utf-8")
    (target / "stderr.log").write_text("", encoding="utf-8")
    missing = [name for name in REQUIRED_RUN_FILES if not (target / name).exists()]
    if missing:
        raise RuntimeError(f"smoke pipeline failed to create artifacts: {missing}")
    return {"run": str(target), "metrics": metrics, "compute": compute}
