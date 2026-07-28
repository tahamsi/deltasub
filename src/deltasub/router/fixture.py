from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from .features import RouterFeatureBatch
from .training import train_records
from ..gains.cache import open_cache
from ..gains.fixture import run_fixture_collection
from ..utils.hashing import stable_hash


def fixture_feature_batch(cache, *, dimension: int = 8, seed: int = 41) -> RouterFeatureBatch:
    records = list(cache.iter_records())
    sample_ids = tuple(cache.metadata["sample_ids"])
    generator = torch.Generator().manual_seed(seed)
    parents = torch.randn(len(sample_ids), 256, dimension, generator=generator)
    # Give every observed candidate a deterministic target-correlated local feature.
    by_sample = {sample: index for index, sample in enumerate(sample_ids)}
    for record in records:
        parents[by_sample[record.sample_id], record.candidate_parent_index, 0] = record.gain
    return RouterFeatureBatch(
        sample_ids, cache.metadata["view_layout_identifier"],
        cache.metadata["batch_context_hashes"][0], parents,
        cache.metadata["model_checkpoint_sha256"], cache.metadata["dinov2_source_revision"],
        cache.metadata["dataset_manifest_sha256"], cache.metadata["configuration_sha256"],
        cache.metadata["child_projector_state_sha256"],
    )


def fixture_config(cache, output: Path, *, epochs: int = 2) -> dict:
    return {
        "schema_version": 1, "test_only": True, "gain_cache_path": str(cache.root),
        "expected_cache_id": cache.cache_id, "feature_cache_path": "",
        "splits": {"train_fraction": .67, "validation_fraction": .33,
                   "test_fraction": 0., "seed": 3},
        "router": {"input_dim": 8, "hidden_dim": 16, "depth": 2,
                   "normalization": "layernorm", "dropout": 0.,
                   "coordinate_features": True, "global_context_features": True,
                   "learned_position_embedding": False},
        "loss": {"regression": "huber", "huber_delta": 1.,
                 "regression_weight": 1., "ranking_weight": .5,
                 "ranking_margin": 0., "maximum_pairs_per_anchor": 16,
                 "sign_weight": .1, "sign_epsilon": 0., "gain_clip": None},
        "training": {"physical_batch_size": 3, "gradient_accumulation": 1,
                     "effective_batch_size": 3, "optimizer": "adamw",
                     "learning_rate": .01, "weight_decay": 0., "scheduler": "none",
                     "epochs": epochs, "gradient_clipping": 1., "precision": "fp32",
                     "device": "cpu"},
        "sampling": {"informative_fraction": .34},
        "replay": {"capacity": 4, "ratio": .25, "priority_mode": "stratified_priority",
                   "near_zero_epsilon": 1e-8},
        "checkpoint_directory": str(output / "training"),
        "validation_metric": "spearman", "seed": 23,
    }


def run_router_fixture(output: str | Path, *, resume: bool = False) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    gain = run_fixture_collection(output / "m4", resume=resume)
    cache = open_cache(gain["cache_path"])
    epochs = 2
    checkpoint = output / "training" / "checkpoint_last.pt"
    if resume and checkpoint.is_file():
        from ..utils.checkpointing import load_checkpoint
        epochs = int(load_checkpoint(checkpoint)["epoch"]) + 1
    config = fixture_config(cache, output, epochs=epochs)
    config_path = output / "router_fixture.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    validation = cache.validate()
    result = train_records(config, list(cache.iter_records()), [fixture_feature_batch(cache)],
                           cache, validation, resume=resume)
    result["resumed_global_step"] = result["global_step"] if resume else 0
    result["m4_fixture_diagnostic"] = gain
    (output / "fixture_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
