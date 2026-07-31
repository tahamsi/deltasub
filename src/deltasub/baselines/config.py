from __future__ import annotations
from pathlib import Path
import yaml
from ..utils.hashing import stable_hash

CONFIG_VERSION="m8.common_protocol.v1"
ALLOWED={"schema_version","common_protocol","enabled_adapters","expected_adapter_statuses","source_roots",
"source_revisions","checkpoint_paths","expected_hashes","execution_mode","budget_comparison_unit","target_budget",
"precision","device","batch_size","gradient_accumulation","loss","optimizer","scheduler","output_directory","seed","mode"}
METHODS={"vit_dinov2_selex","subvit_reimplementation","deltasub","msvit_gcd_reimplementation","dart_gcd_port"}
UNITS={"effective_total_tokens","padded_total_tokens","attention_token_pairs","measured_synchronized_latency"}

def load_config(path):
    value=yaml.safe_load(Path(path).read_text())
    if not isinstance(value,dict): raise ValueError("M8 configuration must be a mapping")
    unknown=set(value)-ALLOWED
    if unknown: raise ValueError(f"unknown critical M8 keys: {sorted(unknown)}")
    missing=ALLOWED-set(value)
    if missing: raise ValueError(f"missing critical M8 keys: {sorted(missing)}")
    if value["schema_version"] != CONFIG_VERSION: raise ValueError("unsupported M8 configuration schema")
    ids=value["enabled_adapters"]
    if len(ids)!=len(set(ids)): raise ValueError("duplicate method IDs")
    if set(ids)-METHODS: raise ValueError("unsupported adapter")
    if value["budget_comparison_unit"] not in UNITS: raise ValueError("ambiguous budget unit")
    if value["execution_mode"] not in {"padded","bucketed"}: raise ValueError("unsupported execution mode")
    if value["precision"] not in {"fp32","bf16"} or value["device"] not in {"cpu","cuda:0"}: raise ValueError("unsupported precision/device")
    if value["mode"] not in {"fixture","production"}: raise ValueError("unsupported mode")
    protocol=value["common_protocol"]
    required={"architecture","input_size","parent_tokens","prefix_tokens","register_tokens","selex","dataset_manifest_hash","split_hash","augmentations","head","optimizer","scheduler","precision","physical_batch_size","effective_batch_size","gradient_accumulation","seed","token_budget","execution_mode","checkpoint_hashes","source_revisions"}
    if set(protocol)!=required: raise ValueError(f"common protocol keys mismatch: {sorted(set(protocol)^required)}")
    if protocol["architecture"]!="DINOv2 ViT-B/14" or protocol["input_size"]!=224 or protocol["parent_tokens"]!=256: raise ValueError("M8 protocol architecture mismatch")
    if value["mode"]=="production":
        if any(str(x).startswith("fixture") for x in value["checkpoint_paths"].values()): raise ValueError("fixture component in production")
        if not all(value["checkpoint_paths"].get(x) and value["expected_hashes"].get(x) for x in ids): raise ValueError("production provenance is incomplete")
    value["config_hash"]=stable_hash(value)
    return value
