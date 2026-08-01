from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable

import torch
import yaml

from ..data.manifests import read_manifest
from ..models.backbones.dinov2 import DINOV2_REVISION, inspect_official_checkpoint
from ..evaluation.gcd_v2 import provenance as gcd_provenance
from ..utils.hashing import sha256_file, stable_hash

SCHEMA = "m9.diagnostic.v1"
DATASETS = {"cub", "aircraft"}
METHODS = ("vit_dinov2_selex", "deltasub")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "dataset", "backbone", "selex", "diagnostic", "output_root"}
    if not isinstance(value, dict) or required - value.keys():
        raise ValueError(f"M9 config missing keys: {sorted(required - set(value or {}))}")
    if value["schema_version"] != SCHEMA: raise ValueError("unsupported M9 schema_version")
    if value["dataset"]["name"] not in DATASETS: raise ValueError("M9 supports only cub and aircraft")
    seed = value["diagnostic"].get("seed")
    if seed != 0 or isinstance(seed, bool): raise ValueError("diagnostic seed must be exactly 0")
    if value["diagnostic"].get("tier") != "diagnostic": raise ValueError("M9 refuses non-diagnostic tiers")
    if value["diagnostic"].get("primary_metric") != "gcd_all_v2":
        raise ValueError("M9 primary metric must be gcd_all_v2")
    margin = value["diagnostic"].get("minimum_delta")
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) or margin < 0:
        raise ValueError("minimum_delta must be non-negative")
    return value


def _git() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout)
    return commit, dirty


def _check(condition: bool, name: str, detail: Any, checks: list[dict]) -> None:
    checks.append({"name": name, "status": "passed" if condition else "failed", "detail": detail})
    if not condition: raise ValueError(f"preflight failed: {name}: {detail}")


def preflight(config_path: str | Path, *, require_cuda: bool = True, inspect_checkpoint: bool = True) -> dict:
    """Fail-closed validation. This reads assets but never starts training."""
    started = time.perf_counter(); config = load_config(config_path); checks: list[dict] = []
    dataset, backbone, selex = config["dataset"], config["backbone"], config["selex"]
    try:
        records = read_manifest(dataset["manifest"], dataset_root=dataset["root"], check_images=True)
        _check(len(records) == int(dataset["samples"]), "manifest_sample_count", len(records), checks)
        observed = sha256_file(dataset["manifest"])
        _check(observed == dataset["manifest_sha256"], "manifest_sha256", observed, checks)
        _check(all(r["dataset"] == dataset["name"] for r in records), "manifest_dataset", dataset["name"], checks)
        split_path = Path(dataset["split_validation"])
        split = json.loads(split_path.read_text(encoding="utf-8"))
        _check(split.get("validation_outcome") == "passed", "split_validation", split.get("validation_outcome"), checks)
        classes = set(split["known_class_ids"]) | set(split["novel_class_ids"])
        _check(all(r["original_class_id"] in classes for r in records), "split_manifest_compatibility", len(classes), checks)
        checkpoint_hash = sha256_file(backbone["checkpoint"])
        _check(checkpoint_hash == backbone["checkpoint_sha256"], "checkpoint_sha256", checkpoint_hash, checks)
        if inspect_checkpoint:
            _, inspection = inspect_official_checkpoint(backbone["checkpoint"], backbone["checkpoint_sha256"],
                                                        source_root=backbone["source_root"])
            _check(inspection.compatible and inspection.strict_checkpoint_load,
                   "strict_checkpoint_inspection", asdict(inspection), checks)
            source_hashes = inspection.source_hashes
        else:
            # Test-only seam: production CLI never disables strict inspection.
            source_hashes = backbone.get("source_hashes", {})
        revision = subprocess.run(["git", "-C", backbone["source_root"], "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
        _check(revision == DINOV2_REVISION, "dinov2_source_revision", revision, checks)
        gate_hash = sha256_file(selex["equivalence_report"])
        _check(gate_hash == selex["equivalence_sha256"], "selex_report_sha256", gate_hash, checks)
        gate = json.loads(Path(selex["equivalence_report"]).read_text(encoding="utf-8"))
        _check(gate.get("status") == "passed", "selex_equivalence", gate.get("status"), checks)
        _check(gate.get("selex_commit") == selex["revision"], "selex_revision", gate.get("selex_commit"), checks)
        gcd = gcd_provenance(implementation_path=Path(__file__).parents[1] / "evaluation/gcd_v2.py")
        _check(gcd["protocol"] == "gcd_v2" and gcd["metric_unit"] == "fraction",
               "gcd_v2_reference", gcd, checks)
        _check((not require_cuda) or (torch.cuda.is_available() and torch.cuda.device_count() == 1),
               "single_cuda", {"available": torch.cuda.is_available(), "count": torch.cuda.device_count()}, checks)
        output = Path(config["output_root"]); output.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(output).free
        _check(free >= int(config["diagnostic"]["minimum_free_bytes"]), "writable_output_space", free, checks)
        probe = output / ".m9_write_probe"; probe.write_text("ok", encoding="utf-8"); probe.unlink()
        commit, dirty = _git()
        report = {"schema_version": SCHEMA, "status": "passed", "training_started": False,
                  "dataset": dataset["name"], "seed": 0, "repository_commit": commit,
                  "repository_dirty": dirty, "checks": checks, "manifest_sha256": observed,
                  "split_validation_sha256": sha256_file(split_path), "checkpoint_sha256": checkpoint_hash,
                  "dinov2_revision": revision, "dinov2_source_hashes": source_hashes,
                  "selex_report_sha256": gate_hash, "resolved_config": config,
                  "gcd_v2_provenance": gcd,
                  "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                  "elapsed_seconds": time.perf_counter() - started}
        _atomic_json(output / dataset["name"] / "preflight.json", report)
        return report
    except Exception as error:
        failure = {"schema_version": SCHEMA, "status": "failed", "training_started": False,
                   "dataset": dataset.get("name"), "checks": checks, "failure_reason": str(error),
                   "elapsed_seconds": time.perf_counter() - started}
        try: _atomic_json(Path(config["output_root"]) / dataset["name"] / "preflight.json", failure)
        except OSError: pass
        raise


def verdict(baseline: float, delta: float, margin: float) -> str:
    difference = delta - baseline
    at_positive = difference > 0 and (difference > margin or math.isclose(difference, margin, rel_tol=1e-12, abs_tol=1e-12))
    if at_positive: return "positive"
    if difference < -margin and not math.isclose(difference, -margin, rel_tol=1e-12, abs_tol=1e-12): return "negative"
    return "neutral"


def run_diagnostic(config_path: str | Path, *, resume: bool = False,
                   executor: Callable[[dict, Path, bool], dict] | None = None) -> dict:
    """Run the production M9 implementation, or an injected fixture executor in tests.

    The production executor lives in ``deltasub.diagnostic.runner`` and is deliberately
    imported only after preflight, preventing model allocation before every gate passes.
    """
    config = load_config(config_path); gate = preflight(config_path)
    output = Path(config["output_root"]) / config["dataset"]["name"]
    if executor is None:
        from .runner import execute_production
        executor = execute_production
    started = time.perf_counter()
    try:
        result = executor(config, output, resume)
        metrics = result["metrics"]
        for method in METHODS:
            if method not in metrics or "gcd_all_v2" not in metrics[method]:
                raise ValueError(f"missing required GCD metric for {method}")
        base, delta = (float(metrics[m]["gcd_all_v2"]) for m in METHODS)
        absolute = delta - base
        final = {"schema_version": SCHEMA, "status": "completed", "dataset": config["dataset"]["name"],
                 "seed": 0, "preflight_sha256": stable_hash(gate), "metrics": metrics,
                 "absolute_delta": absolute, "relative_delta": absolute / abs(base) if base else None,
                 "verdict": verdict(base, delta, float(config["diagnostic"]["minimum_delta"])),
                 "margin": config["diagnostic"]["minimum_delta"], "runtime_seconds": time.perf_counter()-started,
                 **{k: v for k, v in result.items() if k != "metrics"}}
        _atomic_json(output / "diagnostic.json", final); return final
    except Exception as error:
        _atomic_json(output / "diagnostic.json", {"schema_version": SCHEMA, "status": "failed",
                     "verdict": "invalid", "failure_reason": str(error), "dataset": config["dataset"]["name"],
                     "seed": 0, "runtime_seconds": time.perf_counter()-started})
        raise


def summarize(output_root: str | Path) -> dict:
    root = Path(output_root); datasets = {}
    for name in ("cub", "aircraft"):
        path = root / name / "diagnostic.json"
        if not path.is_file(): datasets[name] = {"status": "not_run", "reason": "diagnostic artifact absent"}; continue
        value = json.loads(path.read_text(encoding="utf-8"))
        datasets[name] = {k: value.get(k) for k in ("status", "verdict", "absolute_delta", "relative_delta", "failure_reason") if k in value}
    datasets["cars"] = {"status": "not_run", "reason": "dataset unavailable by user choice"}
    completed = [v.get("verdict") for k, v in datasets.items() if k != "cars" and v.get("status") == "completed"]
    campaign = ("invalid" if len(completed) != 2 or any(v == "invalid" for v in completed) else
                "negative" if "negative" in completed else "positive" if completed and all(v == "positive" for v in completed) else "neutral")
    report = {"schema_version": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
              "status": "authorization_ready" if campaign == "positive" else "diagnostic_only",
              "verdict": campaign, "datasets": datasets, "core_or_full_started": False}
    _atomic_json(root / "campaign.json", report); return report
