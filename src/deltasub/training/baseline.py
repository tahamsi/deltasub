"""Single-device Stage-0 SelEx baseline training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from ..data.manifests import read_manifest
from ..losses.selex import selex_loss
from ..models.backbones.dinov2 import DINOv2Adapter, TestOnlyTinyBackbone
from .selex_equivalence import PRODUCTION, REFERENCE, SELEX_COMMIT, recompute_fp32_equivalence
from ..utils.checkpointing import atomic_torch_save, load_checkpoint
from ..utils.hashing import sha256_file, stable_hash

CONFIG_VERSION = 1
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]


@dataclass(frozen=True)
class BaselineValidation:
    config_sha256: str
    manifest_sha256: str
    checkpoint_sha256: str
    physical_batch_size: int
    gradient_accumulation: int
    effective_batch_size: int
    output_directory: str


def load_baseline_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    allowed = {"schema_version", "dataset", "backbone", "training", "selex", "output_directory", "seed", "test_only"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown critical configuration keys: {sorted(unknown)}")
    if value.get("schema_version") != CONFIG_VERSION:
        raise ValueError("unsupported baseline configuration schema_version")
    return value


def _validate_equivalence_gate(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"SelEx equivalence gate is absent: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status", "selex_commit", "reference_source_path", "reference_source_sha256",
        "production_source_path", "production_source_sha256", "configuration_sha256",
        "fp32_max_absolute_error", "fp32_max_relative_error", "thresholds",
        "modes_run", "numerical_policy",
    }
    if required - gate.keys():
        raise ValueError(f"SelEx equivalence gate is incomplete: {sorted(required - gate.keys())}")
    reference = Path(gate["reference_source_path"])
    production = Path(gate["production_source_path"])
    if reference.resolve() != REFERENCE.resolve() or production.resolve() != PRODUCTION.resolve():
        raise ValueError("SelEx equivalence gate references non-canonical source paths")
    if gate["selex_commit"] != SELEX_COMMIT:
        raise ValueError("SelEx equivalence gate has the wrong pinned commit")
    if not reference.is_file() or sha256_file(reference) != gate["reference_source_sha256"]:
        raise ValueError("SelEx reference source hash is stale or forged")
    if not production.is_file() or sha256_file(production) != gate["production_source_sha256"]:
        raise ValueError("SelEx production source hash is stale or forged")
    thresholds = gate["thresholds"]
    if "cpu_fp32" not in gate["modes_run"]:
        raise ValueError("SelEx equivalence gate lacks required CPU FP32 evidence")
    if gate["numerical_policy"].get("distance") != "euclidean_torch_cdist_p2":
        raise ValueError("SelEx equivalence gate has an incompatible distance policy")
    recomputed = recompute_fp32_equivalence()
    if (
        recomputed["configuration_sha256"] != gate["configuration_sha256"]
        or recomputed["max_absolute_error"] != gate["fp32_max_absolute_error"]
        or recomputed["max_relative_error"] != gate["fp32_max_relative_error"]
    ):
        raise ValueError("SelEx equivalence measurements do not match a fresh execution")
    if gate["status"] != "passed" or gate["fp32_max_absolute_error"] > thresholds["fp32_atol"] or gate["fp32_max_relative_error"] > thresholds["fp32_rtol"]:
        raise ValueError("SelEx scalar-equivalence gate has not passed")


def validate_baseline(config_path: str | Path, checkpoint: str | Path | None = None) -> BaselineValidation:
    config = load_baseline_config(config_path)
    _resolve_training_device(config["training"].get("device", "cpu"))
    dataset = config["dataset"]
    manifest, report = Path(dataset["manifest"]), Path(dataset["split_validation_report"])
    records = read_manifest(manifest)
    if not records:
        raise ValueError("dataset manifest is empty")
    manifest_hash = sha256_file(manifest)
    if manifest_hash != dataset["manifest_sha256"]:
        raise ValueError("dataset manifest checksum mismatch")
    if not report.is_file():
        raise FileNotFoundError(f"split validation report is absent: {report}")
    split = json.loads(report.read_text(encoding="utf-8"))
    if split.get("validation_outcome", split.get("status")) != "passed":
        raise ValueError("split validation report has not passed")
    _validate_equivalence_gate(Path(config["selex"]["equivalence_gate"]))
    checkpoint_path = Path(checkpoint or config["backbone"]["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"backbone checkpoint is absent: {checkpoint_path}")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != config["backbone"]["checkpoint_sha256"]:
        raise ValueError("backbone checkpoint checksum mismatch")
    train = config["training"]
    effective = int(train["physical_batch_size"]) * int(train["gradient_accumulation"])
    if effective != int(train["effective_batch_size"]):
        raise ValueError("effective batch size is inconsistent")
    is_test = bool(config.get("test_only"))
    if config["backbone"]["name"].startswith("test_only_") != is_test:
        raise ValueError("test-only backbones require an explicit test_only configuration and cannot be used in production")
    output = Path(config["output_directory"])
    provenance = output / "resolved_config.yaml"
    if provenance.exists() and yaml.safe_load(provenance.read_text()) != config:
        raise ValueError("run directory contains incompatible configuration or provenance")
    return BaselineValidation(stable_hash(config), manifest_hash, checkpoint_hash,
                              int(train["physical_batch_size"]), int(train["gradient_accumulation"]),
                              effective, str(output))


class ManifestImageDataset(Dataset):
    def __init__(self, records: list[dict], root: str | Path | None, *, train: bool, seed: int):
        self.records = [r for r in records if r["train_or_test_split"] == ("train" if train else "test")]
        self.root, self.train, self.seed, self.epoch = Path(root or "."), train, seed, 0
        if not self.records:
            raise ValueError(f"manifest contains no {'train' if train else 'test'} records")

    def __len__(self):
        return len(self.records)

    def _view(self, path: Path, index: int, view: int):
        try:
            from PIL import Image
            import numpy as np
        except ImportError as error:
            raise RuntimeError("baseline image loading requires Pillow and NumPy") from error
        with Image.open(path) as image:
            image = image.convert("RGB").resize((224, 224))
            rng = random.Random(self.seed + 1000003 * self.epoch + 97 * index + view)
            if self.train and rng.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            array = np.asarray(image, dtype="float32").copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 255
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, index):
        record = self.records[index]
        path = Path(record["image_path"])
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            raise FileNotFoundError(f"manifest image is missing: {path}")
        return {
            "views": torch.stack((self._view(path, index, 0), self._view(path, index, 1))),
            "class_id": int(record["original_class_id"]),
            "pseudo_label": int(record.get("pseudo_label", record["original_class_id"])),
            "labelled": record["labelled_or_unlabelled"] == "labelled",
        }


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _resolve_training_device(policy: str) -> torch.device:
    if not isinstance(policy, str) or policy not in {"cpu", "cuda", "cuda:0", "auto"}:
        raise ValueError(
            "training.device must be one of: auto, cpu, cuda, cuda:0"
        )
    wants_cuda = policy in {"cuda", "cuda:0"} or (
        policy == "auto" and torch.cuda.is_available()
    )
    if wants_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"training.device={policy!r} requested CUDA, but CUDA is unavailable"
            )
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Stage-0 training requires exactly one visible CUDA device"
            )
        return torch.device("cuda:0")
    return torch.device("cpu")


def record_environment(*, requested_device: str, resolved_device: torch.device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "git_commit": _git_commit(), "cuda_available": torch.cuda.is_available(),
            "requested_device": requested_device, "resolved_device": str(resolved_device)}


def save_training_checkpoint(
    path, *, model, optimizer, scheduler, epoch, global_step, config,
    training_device,
):
    atomic_torch_save({"schema_version": 1, "epoch": epoch, "global_step": global_step,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict() if scheduler else None,
                       "config_sha256": stable_hash(config),
                       "training_device": str(training_device)}, path)


def resume_training_checkpoint(
    path, *, model, optimizer, scheduler, config, target_device
):
    state = load_checkpoint(path, map_location=target_device)
    if state.get("config_sha256") != stable_hash(config):
        raise ValueError("resume checkpoint configuration is incompatible")
    if state.get("training_device") != str(target_device):
        raise ValueError(
            "resume checkpoint training device is incompatible with the resolved device"
        )
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["epoch"]), int(state.get("global_step", 0))


def _features(model, images):
    value = model(images)
    if hasattr(value, "cls_token"):
        return value.cls_token
    if torch.is_tensor(value):
        return value
    raise TypeError("backbone did not return usable embeddings")


def run_baseline_training(config_path: str | Path, checkpoint: str | Path | None = None, *, resume=False, seed=None):
    validation = validate_baseline(config_path, checkpoint)
    config = load_baseline_config(config_path)
    if seed is not None:
        config["seed"] = seed
    torch.manual_seed(int(config["seed"]))
    device_policy = config["training"].get("device", "cpu")
    device = _resolve_training_device(device_policy)
    backbone = config["backbone"]
    checkpoint_path = checkpoint or backbone["checkpoint"]
    if backbone["name"] == "test_only_tiny_backbone":
        model = TestOnlyTinyBackbone(int(backbone.get("embed_dim", 16)))
        # The test checkpoint is genuinely loaded, never paired as unrelated provenance.
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    else:
        model = DINOv2Adapter.from_official_checkpoint(
            checkpoint_path, backbone["checkpoint_sha256"], source_root=backbone["source_root"],
            model_name=backbone["name"],
        )
        model.set_trainable_blocks(backbone.get("trainable_blocks", "final"))
    model.to(device)
    records = read_manifest(config["dataset"]["manifest"])
    dataset = ManifestImageDataset(records, config["dataset"].get("root"), train=True, seed=int(config["seed"]))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(dataset, batch_size=validation.physical_batch_size, shuffle=True,
                        num_workers=int(config["training"].get("num_workers", 0)), generator=generator,
                        drop_last=False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("baseline has no trainable parameters")
    training = config["training"]
    if training.get("optimizer", "adamw") == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=float(training["learning_rate"]), momentum=.9)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=float(training["learning_rate"]),
                                      weight_decay=float(training.get("weight_decay", .05)))
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs * len(loader)))
    output = Path(validation.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    resolved = yaml.safe_dump(config, sort_keys=True)
    if not (output / "resolved_config.yaml").exists():
        (output / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
    (output / "environment.json").write_text(
        json.dumps(
            record_environment(
                requested_device=device_policy, resolved_device=device
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
    (output / "dataset_manifest_checksum.txt").write_text(validation.manifest_sha256 + "\n", encoding="utf-8")
    (output / "backbone_checkpoint_hash.txt").write_text(validation.checkpoint_sha256 + "\n", encoding="utf-8")
    start_epoch = global_step = 0
    if resume:
        start_epoch, global_step = resume_training_checkpoint(
            output / "checkpoint_last.pt", model=model, optimizer=optimizer,
            scheduler=scheduler, config=config, target_device=device
        )
    optimizer.zero_grad(set_to_none=True)
    start, best = time.monotonic(), float("inf")
    images_seen = forward_passes = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics_path = output / "metrics.jsonl"
    accumulation = validation.gradient_accumulation
    try:
        for epoch in range(start_epoch, epochs):
            dataset.epoch = epoch
            model.train()
            for batch_index, batch in enumerate(loader):
                views = batch["views"].to(device)
                batch_size = views.shape[0]
                labels = batch["class_id"].to(device)
                labelled = batch["labelled"].to(device)
                pseudo = batch["pseudo_label"].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    embeddings = _features(model, views.flatten(0, 1)).reshape(batch_size, 2, -1)
                    forward_passes += 1
                    images_seen += batch_size * 2
                    count = batch_size * 2
                    confusion = torch.eye(count, device=device, dtype=embeddings.dtype)
                    loss = selex_loss(embeddings, labels, labelled, (pseudo,), confusion,
                                      temperature=float(config["selex"].get("temperature", 1.0)),
                                      sup_con_weight=float(config["selex"].get("supervised_weight", .35)),
                                      unsupervised_smoothing=float(config["selex"].get("unsupervised_smoothing", 1.0)))
                    scaled = loss / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite baseline loss at epoch {epoch}, batch {batch_index}")
                scaled.backward()
                boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
                if boundary:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    elapsed_now = time.monotonic() - start
                    completed_batches = epoch * len(loader) + batch_index + 1
                    total_batches = epochs * len(loader)
                    remaining = elapsed_now / completed_batches * max(0, total_batches - completed_batches)
                    metric = {"epoch": epoch, "global_step": global_step, "loss": float(loss.detach()),
                              "learning_rate": optimizer.param_groups[0]["lr"],
                              "elapsed_seconds": elapsed_now, "estimated_remaining_seconds": remaining,
                              "images_per_second": images_seen / max(elapsed_now, 1e-12),
                              "forward_passes": forward_passes}
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(metric, sort_keys=True) + "\n")
                    save_training_checkpoint(output / "checkpoint_last.pt", model=model, optimizer=optimizer,
                                             scheduler=scheduler, epoch=epoch + 1, global_step=global_step,
                                             config=config, training_device=device)
                    if metric["loss"] < best:
                        best = metric["loss"]
                        save_training_checkpoint(output / "checkpoint_best.pt", model=model, optimizer=optimizer,
                                                 scheduler=scheduler, epoch=epoch + 1, global_step=global_step,
                                                 config=config, training_device=device)
    except Exception:
        (output / "failure.json").write_text(json.dumps({"status": "failed", "elapsed_seconds": time.monotonic() - start}, indent=2), encoding="utf-8")
        raise
    elapsed = time.monotonic() - start
    report = {"status": "completed", "epochs": epochs, "global_step": global_step, "best_loss": best,
              "elapsed_seconds": elapsed, "estimated_remaining_seconds": 0.0,
              "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0.0,
              "images_per_second": images_seen / max(elapsed, 1e-12),
              "forward_passes": forward_passes,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
