"""Executable, hash-bound comparison with the pinned SelEx reference snapshot."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import torch

from ..losses.selex import selex_loss
from ..utils.hashing import sha256_file, stable_hash

SELEX_COMMIT = "569ee7085e779999502bd73ea92240f3d32fc84d"
REFERENCE = Path("third_party/reference/selex_scalar_reference.py")
PRODUCTION = Path("src/deltasub/losses/per_anchor_selex.py")
THRESHOLDS = {"fp32_atol": 2e-6, "fp32_rtol": 2e-6, "bf16_atol": .08, "bf16_rtol": .08}


def _load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("_deltasub_pinned_selex_reference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.scalar_reference


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _cases(device, dtype):
    cases = []
    for batch in (2, 4, 6, 8):
        for labelled_mode, levels in (("mixed", 1), ("all", 2), ("none", 0)):
            generator = torch.Generator(device=device).manual_seed(1000 + batch * 10 + levels)
            features = torch.randn(batch, 2, 16, generator=generator, device=device, dtype=dtype)
            labels = torch.arange(batch, device=device) // 2
            if labelled_mode == "mixed":
                labelled = torch.arange(batch, device=device) % 2 == 0
            else:
                labelled = torch.full((batch,), labelled_mode == "all", device=device, dtype=torch.bool)
            hierarchy = tuple(labels // (2 ** (i + 1)) for i in range(levels))
            confusion = torch.rand(batch * 2, batch * 2, generator=generator, device=device, dtype=dtype)
            confusion = confusion / confusion.sum(1, keepdim=True)
            cases.append((f"b{batch}_{labelled_mode}_l{levels}", features, labels, labelled, hierarchy, confusion))
    return cases


def _run(device, dtype):
    reference = _load_reference(REFERENCE)
    maximum_absolute = maximum_relative = 0.0
    configurations = []
    for name, features, labels, labelled, hierarchy, confusion in _cases(device, dtype):
        actual = selex_loss(features, labels, labelled, hierarchy, confusion)
        # Preserve the pinned snapshot verbatim.  For BF16 validation it executes
        # its unsupported/unstable operations in FP32 using the exact same
        # already-BF16-quantized values that enter the production loss.
        reference_features = features.float() if dtype == torch.bfloat16 else features
        reference_confusion = confusion.float() if dtype == torch.bfloat16 else confusion
        expected = reference(
            reference_features, labels, labelled, hierarchy, reference_confusion
        )
        absolute = float((actual - expected).abs().float())
        relative = absolute / max(float(expected.abs().float()), 1e-12)
        maximum_absolute, maximum_relative = max(maximum_absolute, absolute), max(maximum_relative, relative)
        configurations.append(name)
    return {"max_absolute_error": maximum_absolute, "max_relative_error": maximum_relative,
            "configuration_sha256": stable_hash(configurations)}


def recompute_fp32_equivalence():
    """Re-execute the pinned reference; used by the training gate validator."""
    return _run(torch.device("cpu"), torch.float32)


def verify_equivalence(
    output: str | Path, *, include_cuda: bool = False, include_bf16: bool = False
):
    """Verify CPU FP32 by default, with explicitly requested CUDA modes only."""
    if include_bf16 and not include_cuda:
        raise ValueError("BF16 verification requires include_cuda=True")
    fp32 = _run(torch.device("cpu"), torch.float32)
    cuda = {"status": "not_requested"}
    bf16 = {"status": "not_requested"}
    modes = ["cpu_fp32"]
    if include_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA verification was requested but CUDA is unavailable")
        cuda_value = _run(torch.device("cuda:0"), torch.float32)
        cuda = {"status": "passed", **cuda_value}
        modes.append("cuda_fp32")
    if include_bf16:
        bf16_value = _run(torch.device("cuda:0"), torch.bfloat16)
        bf16 = {"status": "passed", **bf16_value}
        modes.append("cuda_bf16_input_fp32_distance")
    passed = (
        fp32["max_absolute_error"] <= THRESHOLDS["fp32_atol"]
        and fp32["max_relative_error"] <= THRESHOLDS["fp32_rtol"]
        and (cuda["status"] != "passed" or
             (cuda["max_absolute_error"] <= THRESHOLDS["fp32_atol"] and cuda["max_relative_error"] <= THRESHOLDS["fp32_rtol"]))
        and (bf16["status"] != "passed" or
             (bf16["max_absolute_error"] <= THRESHOLDS["bf16_atol"] and bf16["max_relative_error"] <= THRESHOLDS["bf16_rtol"]))
    )
    value = {
        "status": "passed" if passed else "failed", "selex_commit": SELEX_COMMIT,
        "reference_source_path": str(REFERENCE), "reference_source_sha256": sha256_file(REFERENCE),
        "production_source_path": str(PRODUCTION), "production_source_sha256": sha256_file(PRODUCTION),
        "configuration_sha256": fp32["configuration_sha256"],
        "fp32_max_absolute_error": fp32["max_absolute_error"],
        "fp32_max_relative_error": fp32["max_relative_error"],
        "cuda_fp32": cuda, "cuda_bf16": bf16, "thresholds": THRESHOLDS,
        "modes_run": modes,
        "numerical_policy": {
            "distance": "euclidean_torch_cdist_p2",
            "cuda_bf16": "bf16_input_fp32_normalization_distance_logits_and_reductions",
            "loss_output_dtype": "float32",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("SelEx scalar equivalence failed")
    return value
