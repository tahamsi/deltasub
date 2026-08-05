#!/usr/bin/env python
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from deltasub.data.manifests import read_manifest
from deltasub.evaluation.gcd_v2 import evaluate_gcd_v2
from deltasub.experiment.deltasub_residual_compare import bootstrap_hmean_delta
from deltasub.experiment.training import MEAN, STD, _atomic_json, _worker_seed
from deltasub.utils.checkpointing import load_checkpoint
from deltasub.utils.hashing import sha256_file
from deltasub.utils.reproducibility import seed_everything

SCHEMA = "deltasub.prototype-conditional-final.v1"
PATCHES, MODES, DIM = 256, 3, 768
VIEWS = ("original", "low_saturation", "mild_blur", "dimmed")
METHODS = (
    "prototype_conditional", "shuffled_prototype",
    "predictive_information_gain", "feature_shift",
    "detail_energy", "random",
)

def load_ig_module():
    path = Path(__file__).with_name("information_gain_append_gate.py")
    spec = importlib.util.spec_from_file_location("deltasub_ig_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

IG = load_ig_module()

class AlignedViews(Dataset):
    def __init__(self, records, root, limit=0):
        self.records = [r for r in records if r["train_or_test_split"] == "test"]
        if limit:
            self.records = self.records[:limit]
        self.root = Path(root)
        self.resize = transforms.Resize(
            256, interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.crop = transforms.CenterCrop(224)

    def __len__(self):
        return len(self.records)

    @staticmethod
    def tensor(image):
        return TF.normalize(TF.to_tensor(image), MEAN, STD)

    def __getitem__(self, index):
        record = self.records[index]
        relative = Path(record["image_path"])
        if relative.is_absolute():
            raise ValueError("manifest paths must be relative")
        with Image.open(self.root / relative) as source:
            image = source.convert("RGB")
        original = self.crop(self.resize(image))
        views = (
            original,
            ImageEnhance.Color(original).enhance(0.25),
            original.filter(ImageFilter.GaussianBlur(radius=1.5)),
            ImageEnhance.Brightness(original).enhance(0.80),
        )
        return {
            "index": index,
            "views": torch.stack([self.tensor(v) for v in views]),
            "target": int(record["original_class_id"]),
            "old": record["known_or_novel"] == "known",
            "sample_id": str(record["sample_id"]),
        }

def normalize_np(value):
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)

def metrics(target, prediction, old):
    value = evaluate_gcd_v2(target, prediction, old).as_dict()
    old_score, new_score = float(value["old"]), float(value["new"])
    return {
        "all": float(value["all"]), "old": old_score, "new": new_score,
        "hmean": (
            2 * old_score * new_score / (old_score + new_score)
            if old_score + new_score > 0 else 0.0
        ),
    }

def atomic_npz(path, **arrays):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def random_indices(sample_ids, count, device):
    values = []
    for sample_id in sample_ids:
        digest = hashlib.sha256(("prototype-deltasub:" + str(sample_id)).encode()).digest()
        values.append(int.from_bytes(digest[:8], "little") % count)
    return torch.tensor(values, dtype=torch.long, device=device)

def zscore(value):
    return (
        value - value.mean(1, keepdim=True)
    ) / value.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)

def resolution_score(candidate, pair, temperature=0.10):
    similarity = torch.einsum("bvnd,bkd->bvnk", candidate, pair)
    log_probability = F.log_softmax(similarity / temperature, dim=-1)
    probability = log_probability.exp()
    mean_probability = probability.mean(1, keepdim=True)
    jsd = (
        probability * (log_probability - mean_probability.clamp_min(1e-12).log())
    ).sum(-1).mean(1)
    margin = similarity[..., 0] - similarity[..., 1]
    robust = margin.mean(1).abs() - margin.std(1, unbiased=False)
    return zscore(-jsd) + zscore(robust)

def information_gain(global_logits, candidate_logits):
    global_log = F.log_softmax(global_logits.float(), dim=-1)
    candidate_log = F.log_softmax(candidate_logits.float(), dim=-1)
    probability = candidate_log.exp()
    return (
        probability * (candidate_log - global_log[:, :, None, :])
    ).sum(-1).mean(1)

def candidate_features(model, trunk, details, count, chunk):
    batch, tokens, width = trunk.shape
    values = []
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        size = end - start
        expanded = trunk[:, None].expand(-1, size, -1, -1).reshape(
            batch * size, tokens, width
        )
        selected = details[:, start:end].reshape(batch * size, MODES, width)
        sequence = torch.cat((expanded, selected), 1)
        values.append(model._detail_features(sequence).reshape(batch, size, width))
    return torch.cat(values, 1)

def gather(candidate, index):
    batch, views, _, width = candidate.shape
    indices = index[:, None, None, None].expand(batch, views, 1, width)
    return candidate.gather(2, indices).squeeze(2)

def prototypes(mean_features, pseudo_labels):
    labels, inverse = np.unique(pseudo_labels, return_inverse=True)
    classes, width = labels.size, mean_features.shape[1]
    sums = np.zeros((classes, width), dtype=np.float64)
    counts = np.zeros(classes, dtype=np.int64)
    np.add.at(sums, inverse, mean_features)
    np.add.at(counts, inverse, 1)
    full = normalize_np(sums / counts[:, None])
    leave_one_out = np.empty_like(mean_features)
    singletons = 0
    for i, cluster in enumerate(inverse):
        if counts[cluster] > 1:
            leave_one_out[i] = normalize_np(
                ((sums[cluster] - mean_features[i]) / (counts[cluster] - 1))[None]
            )[0]
        else:
            singletons += 1
            leave_one_out[i] = full[cluster]
    similarity = mean_features @ full.T
    similarity[np.arange(len(mean_features)), inverse] = (
        mean_features * leave_one_out
    ).sum(1)
    top_two = np.argsort(-similarity, axis=1)[:, :2]
    control = labels[similarity.argmax(1)]
    pair = full[top_two].copy()
    for column in range(2):
        own = top_two[:, column] == inverse
        pair[own, column] = leave_one_out[own]
    generator = np.random.default_rng(73019)
    permutation = generator.permutation(classes)
    if classes > 1 and np.array_equal(permutation, np.arange(classes)):
        permutation = np.roll(permutation, 1)
    shuffled_index = permutation[top_two]
    shuffled_pair = full[shuffled_index].copy()
    for column in range(2):
        own = shuffled_index[:, column] == inverse
        shuffled_pair[own, column] = leave_one_out[own]
    return {
        "labels": labels.astype(np.int64),
        "inverse": inverse.astype(np.int64),
        "counts": counts,
        "full": full.astype(np.float32),
        "loo": leave_one_out.astype(np.float32),
        "pair": pair.astype(np.float32),
        "shuffled_pair": shuffled_pair.astype(np.float32),
        "control": control.astype(np.int64),
        "singletons": singletons,
    }

def classify(features, indices, state, device):
    full = torch.from_numpy(state["full"]).to(device)
    loo = torch.from_numpy(state["loo"][indices]).to(device)
    inverse = torch.from_numpy(state["inverse"][indices]).to(device)
    similarity = features @ full.T
    similarity[torch.arange(features.shape[0], device=device), inverse] = (
        features * loo
    ).sum(1)
    cluster = similarity.argmax(1).cpu().numpy()
    return state["labels"][cluster]

@torch.inference_mode()
def global_pass(model, loader, device, cache_path):
    features, targets, old, sample_ids = [], [], [], []
    for batch_number, batch in enumerate(loader, 1):
        views = batch["views"].to(device)
        batch, count = views.shape[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            value = model.backbone(
                views.reshape(batch * count, 3, 224, 224)
            ).cls_token
        features.append(value.reshape(batch, count, DIM).float().cpu().numpy())
        targets.extend(batch["target"].tolist())
        old.extend(batch["old"].tolist())
        sample_ids.extend([str(v) for v in batch["sample_id"]])
        if batch_number % 20 == 0 or batch_number == len(loader):
            print(f"global batch {batch_number:04d}/{len(loader):04d}", flush=True)
    result = {
        "features": np.concatenate(features).astype(np.float32),
        "target": np.asarray(targets, dtype=np.int64),
        "old": np.asarray(old, dtype=bool),
        "sample_id": np.asarray(sample_ids, dtype=str),
    }
    atomic_npz(cache_path, **result)
    return result

@torch.inference_mode()
def candidate_pass(
    model, loader, global_features, state,
    candidate_count, chunk_size, device,
):
    predictions = {name: [] for name in METHODS}
    selections = {name: [] for name in METHODS}
    for batch_number, batch in enumerate(loader, 1):
        indices = batch["index"].numpy().astype(np.int64)
        views = batch["views"].to(device)
        batch_size, view_count = views.shape[:2]
        flat = views.reshape(batch_size * view_count, 3, 224, 224)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            trunk, raw_details, detail_tokens = model.trunk_and_details(flat)
            raw_candidate = candidate_features(
                model, trunk, detail_tokens, candidate_count, chunk_size
            )
        candidate = raw_candidate.float().reshape(
            batch_size, view_count, candidate_count, DIM
        )
        candidate_norm = F.normalize(candidate, dim=-1)
        global_raw = torch.from_numpy(global_features[indices]).to(device)
        global_norm = F.normalize(global_raw, dim=-1)
        pair = torch.from_numpy(state["pair"][indices]).to(device)
        shuffled_pair = torch.from_numpy(state["shuffled_pair"][indices]).to(device)

        proto_score = resolution_score(candidate_norm, pair)
        shuffled_score = resolution_score(candidate_norm, shuffled_pair)
        ig_score = information_gain(
            model.head(global_raw).float(), model.head(candidate).float()
        )
        shift_score = (
            1.0 - (candidate_norm * global_norm[:, :, None, :]).sum(-1)
        ).mean(1)
        energy_score = (
            raw_details.float()
            .reshape(batch_size, view_count, PATCHES, MODES, DIM)[
                :, :, :candidate_count
            ]
            .square().mean((-1, -2)).sqrt().mean(1)
        )
        selected = {
            "prototype_conditional": proto_score.argmax(1),
            "shuffled_prototype": shuffled_score.argmax(1),
            "predictive_information_gain": ig_score.argmax(1),
            "feature_shift": shift_score.argmax(1),
            "detail_energy": energy_score.argmax(1),
            "random": random_indices(
                [str(v) for v in batch["sample_id"]], candidate_count, device
            ),
        }
        for name, selected_index in selected.items():
            selected_views = gather(candidate_norm, selected_index)
            selected_mean = F.normalize(selected_views.mean(1), dim=-1)
            prediction = classify(selected_mean, indices, state, device)
            predictions[name].extend(prediction.tolist())
            selections[name].extend(
                selected_index.cpu().numpy().astype(np.int64).tolist()
            )
        if batch_number % 10 == 0 or batch_number == len(loader):
            print(
                f"candidate batch {batch_number:04d}/{len(loader):04d}",
                flush=True,
            )
    return (
        {n: np.asarray(v, dtype=np.int64) for n, v in predictions.items()},
        {n: np.asarray(v, dtype=np.int64) for n, v in selections.items()},
    )

def run(args):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA GPU is required")
    if not 1 <= args.candidate_count <= PATCHES:
        raise ValueError("candidate-count must be in [1,256]")
    if not 1 <= args.chunk_size <= args.candidate_count:
        raise ValueError("chunk-size must be in [1,candidate-count]")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if args.resume and result_path.is_file():
        value = json.loads(result_path.read_text(encoding="utf-8"))
        print("DELTASUB FINAL VERDICT:", value["verdict"].upper())
        print("result:", result_path)
        return

    seed_everything(0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    model, config, _, provenance = IG.load_matched_baseline(device)
    checkpoint = Path(
        "artifacts/deltasub_information_gain/cub/seed_0/checkpoint_last.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_state = load_checkpoint(checkpoint, map_location=device)
    if "append_model" not in checkpoint_state:
        raise KeyError("append checkpoint lacks append_model")
    model.load_append_state_dict(checkpoint_state["append_model"])
    model.requires_grad_(False).eval()

    records = read_manifest(
        config["dataset"]["manifest"],
        dataset_root=config["dataset"]["root"],
        check_images=True,
    )
    dataset = AlignedViews(records, config["dataset"]["root"], args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=_worker_seed,
        persistent_workers=True,
    )

    cache_path = output / "global_features.npz"
    if args.resume and cache_path.is_file():
        archive = np.load(cache_path, allow_pickle=False)
        global_data = {key: archive[key] for key in archive.files}
        expected = (len(dataset), len(VIEWS), DIM)
        if global_data["features"].shape != expected:
            raise RuntimeError("global feature cache shape mismatch")
        print("reusing global feature cache")
    else:
        global_data = global_pass(model, loader, device, cache_path)

    baseline_path = Path(provenance["result"]).parent / "predictions.npz"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline_archive = np.load(baseline_path, allow_pickle=False)
    sample_count = len(dataset)
    target = global_data["target"].astype(np.int64)
    old = global_data["old"].astype(bool)
    baseline = baseline_archive["fused_prediction"][:sample_count].astype(np.int64)
    if not np.array_equal(
        target, baseline_archive["target"][:sample_count].astype(np.int64)
    ):
        raise RuntimeError("target order mismatch")
    if not np.array_equal(
        old, baseline_archive["old"][:sample_count].astype(bool)
    ):
        raise RuntimeError("Old/New order mismatch")

    normalized_views = normalize_np(global_data["features"])
    mean_features = normalize_np(normalized_views.mean(1))
    prototype_state = prototypes(mean_features, baseline)

    predictions, selections = candidate_pass(
        model, loader, global_data["features"], prototype_state,
        args.candidate_count, args.chunk_size, device,
    )
    all_predictions = {
        "matched_selex": baseline,
        "prototype_control": prototype_state["control"],
        **predictions,
    }
    all_metrics = {
        name: metrics(target, prediction, old)
        for name, prediction in all_predictions.items()
    }

    reference = all_metrics["prototype_control"]
    candidate = all_metrics["prototype_conditional"]
    delta = {
        key: candidate[key] - reference[key]
        for key in ("all", "old", "new", "hmean")
    }
    interval = bootstrap_hmean_delta(
        targets=target,
        old=old,
        baseline=all_predictions["prototype_control"],
        candidate=all_predictions["prototype_conditional"],
        draws=args.bootstrap_draws,
        seed=88041,
    )
    controls = (
        "shuffled_prototype", "predictive_information_gain",
        "feature_shift", "detail_energy", "random",
    )
    best_control = max(
        controls, key=lambda name: all_metrics[name]["hmean"]
    )
    passed = (
        delta["all"] >= 0.0
        and delta["new"] >= 0.005
        and delta["hmean"] >= 0.005
        and interval["lower_95"] > 0.0
        and candidate["hmean"] > all_metrics[best_control]["hmean"]
    )
    verdict = "keep" if passed else "abandon"

    np.savez_compressed(
        output / "predictions.npz",
        target=target,
        old=old,
        **{
            f"{name}_prediction": prediction
            for name, prediction in all_predictions.items()
        },
        **{f"{name}_index": index for name, index in selections.items()},
    )
    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "dataset": "cub",
        "seed": 0,
        "training_used": False,
        "oracle_used": False,
        "test_labels_used_for_selection": False,
        "transductive_test_features_used": True,
        "original_tokens_retained": True,
        "subtokens_appended": True,
        "selected_k": 1,
        "candidate_count": args.candidate_count,
        "views": list(VIEWS),
        "metrics": all_metrics,
        "delta_vs_prototype_control": delta,
        "hmean_paired_bootstrap": interval,
        "best_patch_control": {
            "name": best_control,
            "hmean": all_metrics[best_control]["hmean"],
        },
        "selection_overlap": {
            name: float(
                np.mean(
                    selections["prototype_conditional"] == selections[name]
                )
            )
            for name in controls
        },
        "prototype_diagnostics": {
            "pseudo_cluster_count": int(prototype_state["labels"].size),
            "singleton_count": int(prototype_state["singletons"]),
            "minimum_size": int(prototype_state["counts"].min()),
            "maximum_size": int(prototype_state["counts"].max()),
        },
        "gate_rule": {
            "reference": "cross-fitted multi-view prototype control",
            "all_delta_minimum": 0.0,
            "new_delta_minimum": 0.005,
            "hmean_delta_minimum": 0.005,
            "hmean_lower_95_above_zero": True,
            "strictly_above_all_patch_controls": True,
        },
        "append_checkpoint": str(checkpoint),
        "append_checkpoint_sha256": sha256_file(checkpoint),
        "baseline_archive": str(baseline_path),
        "baseline_archive_sha256": sha256_file(baseline_path),
        "samples": sample_count,
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "verdict": verdict,
    }
    _atomic_json(result_path, result)

    print()
    print("===== PROTOTYPE-CONDITIONAL DELTASUB FINAL GATE =====")
    print("method                            all       old       new      hmean")
    order = (
        "matched_selex", "prototype_control", "prototype_conditional",
        "shuffled_prototype", "predictive_information_gain",
        "feature_shift", "detail_energy", "random",
    )
    for name in order:
        value = all_metrics[name]
        print(
            f"{name:<34}"
            f"{value['all']:>9.6f}"
            f"{value['old']:>10.6f}"
            f"{value['new']:>10.6f}"
            f"{value['hmean']:>11.6f}"
        )
    print()
    print("delta versus cross-fitted prototype control")
    for key in ("all", "old", "new", "hmean"):
        print(f"{key:<5}: {delta[key]:+.6f}")
    print(
        "hmean paired-bootstrap 95% CI: "
        f"[{interval['lower_95']:+.6f}, {interval['upper_95']:+.6f}]"
    )
    print(
        "best patch-selection control: "
        f"{best_control} ({all_metrics[best_control]['hmean']:.6f})"
    )
    print()
    print("DELTASUB FINAL VERDICT:", verdict.upper())
    print("result:", result_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/deltasub_prototype_conditional/cub/seed_0",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=256)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())

if __name__ == "__main__":
    main()
