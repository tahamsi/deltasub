from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import yaml

from ..diagnostic.campaign import preflight as diagnostic_preflight
from ..gains.collector import collect as collect_gains
from ..router.features import load_feature_cache
from ..router.training import train_from_cache
from ..utils.hashing import sha256_file, stable_hash
from .training import METHODS, run_production

SCHEMA = "m9.campaign.v1"
CARS = {"status": "not_run", "reason": "dataset unavailable by user choice"}


def _atomic(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True); fd,name=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream: json.dump(value,stream,indent=2,sort_keys=True,default=str); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def load_campaign(path: str | Path) -> dict:
    value=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required={"schema_version","tier","datasets","methods","seeds","backbone","selex","gcd","training","deltasub","ablations","output_root","unavailable_datasets"}
    if not isinstance(value,dict) or set(value)!=required: raise ValueError(f"campaign keys mismatch: {sorted(set(value or {})^required)}")
    if value["schema_version"]!=SCHEMA or value["tier"] not in {"diagnostic","core","full"}: raise ValueError("unsupported campaign schema/tier")
    if set(value["datasets"])!={"cub","aircraft"} or set(value["methods"])!=METHODS: raise ValueError("campaign requires CUB/Aircraft and baseline/DeltaSub")
    if value["tier"]=="diagnostic" and value["seeds"]!=[0]: raise ValueError("diagnostic seed must be exactly 0")
    if value["tier"]!="diagnostic" and value["seeds"]!=[0,1,2]: raise ValueError("publication seeds must be exactly 0,1,2")
    if value["unavailable_datasets"].get("cars")!=CARS: raise ValueError("Cars status/reason is immutable")
    return value


def resolve(config: dict, dataset: str, method: str, seed: int, ablation: str | None = None) -> dict:
    if dataset not in config["datasets"] or method not in METHODS or seed not in config["seeds"]: raise ValueError("run is outside campaign matrix")
    data=dict(config["datasets"][dataset]); split=json.loads(Path(data["split_validation"]).read_text())
    data["classes"]=len(set(split["known_class_ids"])|set(split["novel_class_ids"]))
    result={"schema_version":"m9.run.v1","tier":config["tier"],"dataset":data,"backbone":dict(config["backbone"]),
            "selex":dict(config["selex"]),"gcd":dict(config["gcd"]),"seed":seed,
            "baseline":dict(config["training"]["baseline"]),"deltasub":dict(config["training"]["deltasub"])}
    result["deltasub"].update(config["deltasub"])
    if ablation is not None:
        if method != "deltasub" or ablation not in config["ablations"]: raise ValueError("unknown or inapplicable ablation")
        result["deltasub"].update(config["ablations"][ablation]); result["ablation"]=ablation
    root=Path(config["output_root"])/config["tier"]/dataset/method/f"seed_{seed}"
    if ablation is not None: root=root/f"ablation_{ablation}"
    result["output_directory"]=str(root)
    result["deltasub"]["router_checkpoint"]=str(root/"stages/router/checkpoint_best.pt")
    return result


def _write_yaml(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(yaml.safe_dump(value,sort_keys=True),encoding="utf-8")


def _ensure_deltasub_stages(run: dict, *, resume: bool):
    out=Path(run["output_directory"]); stages=out/"stages"; d=run["dataset"]; b=run["backbone"]; s=run["selex"]; spec=run["deltasub"]
    candidate_count=int(spec["gain_candidates_per_image"])
    parents=list(range(256)) if candidate_count==256 else [int(i*256/candidate_count) for i in range(candidate_count)]
    gain={"schema_version":1,"test_only":False,"dataset":{"name":d["name"],"manifest":d["manifest"],"manifest_sha256":d["manifest_sha256"],
        "split_report":d["split_validation"],"split_report_sha256":sha256_file(d["split_validation"])},"backbone":b,
        "selex":{"equivalence_gate":s["equivalence_report"],"equivalence_gate_sha256":s["equivalence_sha256"]},
        "subtokens":{"config":"configs/subtokens/haar_direct.yaml","config_sha256":sha256_file("configs/subtokens/haar_direct.yaml")},
        "collection":{"batch_size":int(spec["gain_batch_size"]),"candidate":{"mode":"list","parent_indices":parents,"start":None,"stop":None,"sample_limit":spec["gain_sample_limit"],"batch_limit":None},
        "seed":run["seed"],"precision":"bf16","device":"cuda:0","shard_size":int(spec["gain_shard_size"]),"deterministic_algorithms":"error","counterfactual_chunk_size":int(spec["candidate_microbatch_size"]),"resume_policy":"verify"},
        "output_root":str(stages/"gains")}
    gain_path=stages/"gain_config.yaml"; _write_yaml(gain_path,gain)
    gain_result=collect_gains(gain_path,resume=resume)
    router={"schema_version":1,"test_only":False,"gain_cache_path":gain_result["cache_path"],"expected_cache_id":gain_result["cache_id"],
        "feature_cache_path":gain_result["router_feature_cache"],"splits":{"train_fraction":.8,"validation_fraction":.2,"test_fraction":0.,"seed":run["seed"]},
        "router":spec["router_architecture"],"loss":{"regression":"huber","huber_delta":.1,"regression_weight":1.,"ranking_weight":.5,"ranking_margin":0.,"maximum_pairs_per_anchor":1024,"sign_weight":0.,"sign_epsilon":0.,"gain_clip":None},
        "training":{"physical_batch_size":int(spec["router_batch_size"]),"gradient_accumulation":1,"effective_batch_size":int(spec["router_batch_size"]),"optimizer":"adamw","learning_rate":3e-4,"weight_decay":.01,"scheduler":"cosine","epochs":int(spec["router_epochs"]),"gradient_clipping":1.,"precision":"bf16","device":"cuda:0"},
        "sampling":{"informative_fraction":.5},"replay":{"capacity":4096,"ratio":.25,"priority_mode":"stratified_priority","near_zero_epsilon":1e-6},
        "checkpoint_directory":str(stages/"router"),"validation_metric":"spearman","seed":run["seed"]}
    router_path=stages/"router_config.yaml"; _write_yaml(router_path,router)
    train_from_cache(router_path,load_feature_cache(gain_result["router_feature_cache"]),resume=resume)


def preflight(path: str | Path) -> dict:
    config=load_campaign(path); reports={}
    for dataset in config["datasets"]:
        resolved=resolve(config,dataset,"baseline",config["seeds"][0]); temporary=Path(config["output_root"])/"preflight"/f"{dataset}.yaml"
        diag={"schema_version":"m9.diagnostic.v1","dataset":{k:resolved["dataset"][k] for k in ("name","root","manifest","manifest_sha256","samples","split_validation")},
              "backbone":resolved["backbone"],"selex":resolved["selex"],"diagnostic":{"tier":"diagnostic","seed":0,"primary_metric":"gcd_all_v2","minimum_delta":0.,"minimum_free_bytes":config["training"]["baseline"]["minimum_free_bytes"],"precision":"bf16_input_fp32_loss_and_metrics","baseline":{},"deltasub":{}},"output_root":str(Path(config["output_root"])/"preflight")}
        _write_yaml(temporary,diag); reports[dataset]=diagnostic_preflight(temporary)
    result={"schema_version":SCHEMA,"status":"passed","training_started":False,"datasets":reports,"cars":CARS}; _atomic(Path(config["output_root"])/"preflight.json",result); return result


def _authorize_publication(tier: str, *, confirm_full: bool) -> None:
    verdict_path=Path("DIAGNOSTIC_VERDICT.md")
    text=verdict_path.read_text(encoding="utf-8").lower() if verdict_path.is_file() else ""
    positive="status: **positive**" in text or "status: positive" in text
    neutral="status: **neutral**" in text or "status: neutral" in text
    if tier=="core" and not (positive or neutral):
        raise RuntimeError("core tier blocked: DIAGNOSTIC_VERDICT.md is neither positive nor neutral")
    if tier=="full" and (not positive or not confirm_full):
        raise RuntimeError("full tier requires a positive DIAGNOSTIC_VERDICT.md and --confirm-full")


def run(path,dataset,method,seed,*,resume=False,confirm_full=False,ablation=None):
    config=load_campaign(path); resolved=resolve(config,dataset,method,seed,ablation); out=Path(resolved["output_directory"])
    _authorize_publication(config["tier"],confirm_full=confirm_full)
    _write_yaml(out/"resolved_config.yaml",resolved)
    try:
        if method=="deltasub": _ensure_deltasub_stages(resolved,resume=resume)
        return run_production(resolved,method,out,resume=resume)
    except Exception as error:
        _atomic(out/"failure.json",{"status":"failed","dataset":dataset,"method":method,"seed":seed,
                "failure_reason":str(error),"timestamp":datetime.now(timezone.utc).isoformat()})
        raise


def status(path):
    config=load_campaign(path); runs=[]
    for dataset in config["datasets"]:
      for method in sorted(METHODS):
       for seed in config["seeds"]:
        out=Path(resolve(config,dataset,method,seed)["output_directory"]); artifact=out/"result.json"
        value=json.loads(artifact.read_text()) if artifact.is_file() else {"status":"not_run"}
        runs.append({"dataset":dataset,"method":method,"seed":seed,"status":value.get("status"),"path":str(out)})
    result={"schema_version":SCHEMA,"tier":config["tier"],"runs":runs,"cars":CARS}; _atomic(Path(config["output_root"])/config["tier"]/"status.json",result); return result


def aggregate(path):
    config=load_campaign(path); rows=[]
    for dataset in config["datasets"]:
      for method in sorted(METHODS):
        values=[]
        for seed in config["seeds"]:
            artifact=Path(resolve(config,dataset,method,seed)["output_directory"])/"result.json"
            if artifact.is_file() and json.loads(artifact.read_text()).get("status")=="completed": values.append(json.loads(artifact.read_text()))
        row={"dataset":dataset,"method":method,"completed_seeds":len(values),"required_seeds":len(config["seeds"]),"status":"completed" if len(values)==len(config["seeds"]) else "incomplete"}
        for metric in ("all","old","new"):
            data=[v["metrics"][metric] for v in values]; row[f"gcd_{metric}_v2_mean"]=sum(data)/len(data) if data else None
            row[f"gcd_{metric}_v2_std"]=(sum((x-row[f"gcd_{metric}_v2_mean"])**2 for x in data)/len(data))**.5 if data else None
        rows.append(row)
    root=Path(config["output_root"])/config["tier"]; result={"schema_version":SCHEMA,"generated_at":datetime.now(timezone.utc).isoformat(),"rows":rows,"cars":CARS}; _atomic(root/"aggregate.json",result)
    with (root/"aggregate.csv").open("w",newline="",encoding="utf-8") as stream: writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    columns=("dataset","method","completed_seeds","required_seeds","status","gcd_all_v2_mean","gcd_old_v2_mean","gcd_new_v2_mean")
    lines=["| "+" | ".join(columns)+" |","| "+" | ".join("---" for _ in columns)+" |"]
    lines.extend("| "+" | ".join(str(row.get(key)) for key in columns)+" |" for row in rows)
    (root/"publication_table.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result
