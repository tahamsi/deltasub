"""Single-A100, resume-safe common-protocol training for M9.

This module deliberately keeps ground-truth test labels outside every training API.
They enter only ``evaluate`` after the selected checkpoint has been restored.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from ..adaptive.selection import deterministic_select
from ..data.manifests import read_manifest
from ..evaluation.gcd_v2 import evaluate_gcd_v2, provenance as gcd_provenance
from ..losses.selex import selex_loss
from ..models.backbones.dinov2 import DINOv2Adapter
from ..models.subtokens.geometry import extract_parent_patches, subdivide_parent_patches
from ..models.subtokens.haar import HaarDetails
from ..models.subtokens.positions import ParentAwareDetailPositions
from ..models.subtokens.projection import enforce_parent_consistency
from ..router.model import GainRouter, RouterConfig
from ..utils.checkpointing import atomic_torch_save, load_checkpoint
from ..utils.hashing import sha256_file, stable_hash
from ..utils.reproducibility import seed_everything

METHODS = {"baseline", "deltasub"}
MEAN, STD = (0.485, .456, .406), (.229, .224, .225)


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


class ProductionManifestDataset(Dataset):
    """Manifest-only image dataset. Test targets are never returned in train mode."""
    def __init__(self, records, root, *, split: str, seed: int, train: bool):
        if split not in {"train", "test"}: raise ValueError("split must be train or test")
        self.records = [r for r in records if r["train_or_test_split"] == split]
        if not self.records: raise ValueError(f"empty {split} partition")
        self.root, self.seed, self.train, self.epoch = Path(root), int(seed), train, 0
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]; path = Path(record["image_path"])
        if path.is_absolute(): raise ValueError("production manifests must contain relative image paths")
        path = self.root / path
        if not path.is_file(): raise FileNotFoundError(f"manifest image missing: {path}")
        with Image.open(path) as source: image = source.convert("RGB")
        if self.train:
            views = []
            for view in range(2):
                state = torch.random.get_rng_state()
                torch.manual_seed(self.seed + self.epoch * 1_000_003 + index * 97 + view)
                views.append(self.train_transform(image))
                torch.random.set_rng_state(state)
            result = {"views": torch.stack(views), "labelled": record["labelled_or_unlabelled"] == "labelled",
                      "sample_id": record["sample_id"]}
            # Only genuinely labelled training targets cross this boundary.
            result["target"] = int(record["original_class_id"]) if result["labelled"] else -1
            return result
        return {"image": self.eval_transform(image), "target": int(record["original_class_id"]),
                "old": record["known_or_novel"] == "known", "sample_id": record["sample_id"]}


class FrozenBaseline(nn.Module):
    def __init__(self, backbone: DINOv2Adapter, classes: int):
        super().__init__(); self.backbone = backbone; self.head = nn.Linear(768, classes)
        self.backbone.requires_grad_(False).eval()
    def train(self, mode=True):
        super().train(mode); self.backbone.eval(); return self
    def features(self, images):
        with torch.no_grad(): return self.backbone(images).cls_token
    def forward(self, images): return self.head(self.features(images))


class ProductionDeltaSub(nn.Module):
    """M3 Haar details selected by the trained M5 router and encoded by DINOv2 blocks."""
    def __init__(self, backbone, router, classes, fixed_k):
        super().__init__(); self.backbone, self.router = backbone, router
        self.child = backbone.build_child_projector(trainable=True)
        self.positions = ParentAwareDetailPositions(); self.haar = HaarDetails()
        self.head = nn.Linear(768, classes); self.fixed_k = int(fixed_k)
        if not 0 <= self.fixed_k <= 256: raise ValueError("DeltaSub K must be in [0,256]")
        backbone.requires_grad_(False).eval(); router.requires_grad_(False).eval()
    def train(self, mode=True):
        super().train(mode); self.backbone.eval(); self.router.eval(); return self
    def features(self, images):
        parents = self.backbone.pre_transformer_parent_embeddings(images)
        with torch.no_grad(): scores = self.router(parents.float())
        k = torch.full((len(images),), self.fixed_k, device=images.device, dtype=torch.long)
        selected = deterministic_select(scores, k).selected_mask
        children = self.child(subdivide_parent_patches(extract_parent_patches(images)))
        consistent = enforce_parent_consistency(children, parents).consistent
        details = self.haar(consistent)
        parent_pos = self.backbone.parent_patch_positions().to(parents).expand(len(images), -1, -1)
        prefix = self.backbone.prefix_tokens_with_positions(len(images)).to(parents)
        detail_pos = self.positions(parent_pos)
        sequences = []
        for row in range(len(images)):
            sequences.append(torch.cat((prefix[row], parents[row] + parent_pos[row],
                                        (details[row, selected[row]] + detail_pos[row, selected[row]]).reshape(-1, 768))))
        tokens = torch.stack(sequences)  # fixed K gives an exact rectangular batch
        for block in self.backbone.model.blocks: tokens = block(tokens)
        return self.backbone.model.norm(tokens)[:, 0]
    def forward(self, images): return self.head(self.features(images))


def construct_production_model(config: dict, method: str, *, device="cpu") -> nn.Module:
    if method not in METHODS: raise ValueError(f"unsupported production method: {method}")
    backbone_cfg = config["backbone"]
    backbone = DINOv2Adapter.from_official_checkpoint(
        backbone_cfg["checkpoint"], backbone_cfg["checkpoint_sha256"],
        source_root=backbone_cfg["source_root"], model_name=backbone_cfg["name"])
    classes = int(config["dataset"]["classes"])
    if method == "baseline": model = FrozenBaseline(backbone, classes)
    else:
        path = Path(config["deltasub"]["router_checkpoint"])
        if not path.is_file(): raise FileNotFoundError(f"required M5 router checkpoint absent: {path}")
        state = load_checkpoint(path, map_location="cpu")
        required = {"router_state", "router_configuration_hash", "m4_cache_id", "dinov2_checkpoint_hash", "dinov2_source_revision"}
        if required - state.keys(): raise ValueError("M5 router checkpoint lacks production provenance")
        if state["dinov2_checkpoint_hash"] != backbone_cfg["checkpoint_sha256"]: raise ValueError("router/backbone checkpoint mismatch")
        router_cfg = RouterConfig(**config["deltasub"]["router_architecture"])
        router = GainRouter(router_cfg, seed=int(config["seed"]))
        if router.configuration_hash != state["router_configuration_hash"]: raise ValueError("router configuration mismatch")
        router.load_state_dict(state["router_state"], strict=True)
        model = ProductionDeltaSub(backbone, router, classes, config["deltasub"]["fixed_k"])
    return model.to(device)


def _worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32; np.random.seed(seed); random.seed(seed)


def _checkpoint(path, model, optimizer, scheduler, epoch, step, best, config, generator):
    atomic_torch_save({"schema_version": "m9.run.v1", "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch,
        "global_step": step, "best_selection_loss": best, "config_hash": stable_hash(config),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()}, "loader_generator": generator.get_state()}, path)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); predictions=[]; targets=[]; masks=[]
    for batch in loader:
        logits=model(batch["image"].to(device)); predictions.extend(logits.float().argmax(1).cpu().tolist())
        targets.extend(batch["target"].tolist()); masks.extend(batch["old"].tolist())
    return evaluate_gcd_v2(targets, predictions, np.asarray(masks, dtype=bool)).as_dict()


def run_production(config: dict, method: str, output: str | Path, *, resume=False) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("production runs require exactly cuda:0")
    device=torch.device("cuda:0"); seed=int(config["seed"]); seed_everything(seed)
    output=Path(output); output.mkdir(parents=True, exist_ok=True)
    import yaml
    resolved_text=yaml.safe_dump(config,sort_keys=True)
    existing=output/"resolved_config.yaml"
    if existing.exists() and existing.read_text(encoding="utf-8")!=resolved_text: raise ValueError("run directory contains a different resolved config")
    _atomic_text(existing,resolved_text)
    records=read_manifest(config["dataset"]["manifest"], dataset_root=config["dataset"]["root"], check_images=True)
    train=ProductionManifestDataset(records, config["dataset"]["root"], split="train", seed=seed, train=True)
    test=ProductionManifestDataset(records, config["dataset"]["root"], split="test", seed=seed, train=False)
    model=construct_production_model(config, method, device=device); spec=config[method]
    commit=subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
    dirty_text=subprocess.run(["git","diff","--binary","HEAD"],check=True,capture_output=True).stdout
    split_path=Path(config["dataset"]["split_validation"]); split=json.loads(split_path.read_text(encoding="utf-8"))
    provenance={"repository_commit":commit,"repository_dirty":bool(dirty_text),"dirty_diff_sha256":stable_hash(dirty_text),
        "dataset_root":config["dataset"]["root"],"manifest_path":config["dataset"]["manifest"],"manifest_sha256":sha256_file(config["dataset"]["manifest"]),
        "split_validation_path":str(split_path),"split_validation_sha256":sha256_file(split_path),"backbone_checkpoint":config["backbone"]["checkpoint"],
        "backbone_checkpoint_sha256":sha256_file(config["backbone"]["checkpoint"]),"dinov2_source_revision":"7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "gcd":gcd_provenance(implementation_path=Path(__file__).parents[1]/"evaluation/gcd_v2.py"),"selex_revision":config["selex"]["revision"],
        "selex_equivalence_report_sha256":sha256_file(config["selex"]["equivalence_report"]),"seed":seed,
        "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"precision_policy":"CUDA BF16 forward; FP32 losses, reductions, and metrics",
        "known_class_count":len(split["known_class_ids"]),"novel_class_count":len(split["novel_class_ids"]),
        "environment":{"python":platform.python_version(),"torch":str(torch.__version__),"cuda":str(torch.version.cuda),"cudnn":str(torch.backends.cudnn.version()),
                       "device":torch.cuda.get_device_name(device),"driver":subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()}}
    _atomic_json(output/"provenance.json",provenance)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters, lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(train,batch_size=int(spec["physical_batch_size"]),shuffle=True,generator=generator,
        num_workers=int(spec["num_workers"]),worker_init_fn=_worker_seed,persistent_workers=int(spec["num_workers"])>0)
    evaluation=DataLoader(test,batch_size=int(spec["evaluation_batch_size"]),shuffle=False,
        num_workers=int(spec["num_workers"]),worker_init_fn=_worker_seed)
    epochs=int(spec["epochs"]); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,epochs*len(loader)))
    start=step=0; best=float("inf"); last=output/"checkpoint_last.pt"; history=[]
    # --resume is idempotent: resume when a checkpoint exists, otherwise start fresh.
    if resume and last.is_file():
        state=load_checkpoint(last,map_location=device)
        if state.get("config_hash")!=stable_hash(config): raise ValueError("resume config mismatch")
        model.load_state_dict(state["model"],strict=True); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start,step,best=state["epoch"],state["global_step"],state["best_selection_loss"]
        random.setstate(state["rng"]["python"]); np.random.set_state(state["rng"]["numpy"]); torch.set_rng_state(state["rng"]["torch"]); torch.cuda.set_rng_state_all(state["rng"]["cuda"]); generator.set_state(state["loader_generator"])
        history.append({"resumed_at":datetime.now(timezone.utc).isoformat(),"epoch":start,"global_step":step,"checkpoint_sha256":sha256_file(last)})
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device); metrics=output/"metrics.jsonl"
    accumulation=int(spec["gradient_accumulation"]); optimizer.zero_grad(set_to_none=True)
    for epoch in range(start,epochs):
        train.epoch=epoch; model.train(); epoch_loss=0.; batches=0
        for index,batch in enumerate(loader):
            views=batch["views"].to(device); labelled=batch["labelled"].to(device); targets=batch["target"].to(device); b=len(views)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                features=model.features(views.flatten(0,1)).reshape(b,2,-1); logits=model.head(features)
                supervised=torch.nn.functional.cross_entropy(logits[:,0][labelled].float(),targets[labelled]) if labelled.any() else logits.float().sum()*0
                pseudo=logits.detach().mean(1).argmax(1); confusion=torch.eye(2*b,device=device,dtype=features.dtype)
                unsupervised=selex_loss(features,targets.clamp_min(0),labelled,(pseudo,),confusion,
                    temperature=float(spec["temperature"]),sup_con_weight=float(spec["supervised_weight"]))
                loss=(float(spec["classification_weight"])*supervised+float(spec["selex_weight"])*unsupervised)/accumulation
            if not torch.isfinite(loss): raise FloatingPointError("non-finite production loss")
            loss.backward(); epoch_loss+=float(loss.detach())*accumulation; batches+=1
            if (index+1)%accumulation==0 or index+1==len(loader):
                torch.nn.utils.clip_grad_norm_(parameters,float(spec["gradient_clipping"])); optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); step+=1
        selection=epoch_loss/max(batches,1); record={"epoch":epoch+1,"global_step":step,"selection_loss":selection,"learning_rate":optimizer.param_groups[0]["lr"],"elapsed_seconds":time.perf_counter()-started}
        with metrics.open("a",encoding="utf-8") as stream: stream.write(json.dumps(record,sort_keys=True)+"\n"); stream.flush(); os.fsync(stream.fileno())
        _checkpoint(last,model,optimizer,scheduler,epoch+1,step,min(best,selection),config,generator)
        if selection<best: best=selection; _checkpoint(output/"checkpoint_best.pt",model,optimizer,scheduler,epoch+1,step,best,config,generator)
    best_state=load_checkpoint(output/"checkpoint_best.pt",map_location=device); model.load_state_dict(best_state["model"],strict=True)
    final=evaluate(model,evaluation,device); elapsed=time.perf_counter()-started
    checkpoint_size=(output/"checkpoint_best.pt").stat().st_size
    result={"status":"completed","method":method,"seed":seed,"metrics":final,"best_checkpoint_selection_rule":"minimum training objective (test labels never used)",
        "train_samples":len(train),"eval_samples":len(test),"labelled_samples":sum(r["labelled_or_unlabelled"]=="labelled" for r in train.records),
        "unlabelled_samples":sum(r["labelled_or_unlabelled"]=="unlabelled" for r in train.records),"runtime_seconds":elapsed,"gpu_hours":elapsed/3600,
        "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(device),"total_parameters":sum(p.numel() for p in model.parameters()),
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"checkpoint_sha256":sha256_file(output/"checkpoint_best.pt"),
        "checkpoint_size_bytes":checkpoint_size,"evaluation_samples_per_second":len(test)/max(elapsed,1e-12),"resume_history":history}
    _atomic_json(output/"result.json",result); return result
