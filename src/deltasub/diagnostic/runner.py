"""Real M9 diagnostic executor backed by the production experiment path."""
from __future__ import annotations

import json
from pathlib import Path

from ..experiment.campaign import _ensure_deltasub_stages
from ..experiment.training import run_production


def _resolved(config: dict, output: Path) -> dict:
    split=json.loads(Path(config["dataset"]["split_validation"]).read_text(encoding="utf-8"))
    dataset=dict(config["dataset"]); dataset["classes"]=len(set(split["known_class_ids"])|set(split["novel_class_ids"]))
    common={"learning_rate":1e-3,"weight_decay":.05,"num_workers":4,"evaluation_batch_size":64,
            "temperature":1.,"supervised_weight":.35,"classification_weight":1.,"selex_weight":1.,"gradient_clipping":1.}
    baseline={**common,**config["diagnostic"]["baseline"]}
    delta={**common,**config["diagnostic"]["deltasub"]}
    baseline.pop("frozen_backbone",None)
    delta["epochs"]=int(delta.pop("adaptive_epochs"))
    delta["physical_batch_size"]=int(delta.get("physical_batch_size",8)); delta["gradient_accumulation"]=int(delta.get("gradient_accumulation",4))
    delta.setdefault("gain_batch_size",2); delta.setdefault("gain_shard_size",128); delta.setdefault("candidate_microbatch_size",1)
    delta.setdefault("router_batch_size",64); delta.setdefault("weight_decay",.01)
    delta.setdefault("router_architecture",{"input_dim":768,"hidden_dim":256,"depth":2,"normalization":"layernorm","dropout":0.,"coordinate_features":True,"global_context_features":True,"learned_position_embedding":False})
    delta["router_checkpoint"]=str(output/"deltasub"/"stages/router/checkpoint_best.pt")
    return {"schema_version":"m9.run.v1","tier":"diagnostic","dataset":dataset,"backbone":config["backbone"],
            "selex":config["selex"],"gcd":{"protocol":"gcd_v2","unit":"fraction"},"seed":0,
            "baseline":baseline,"deltasub":delta}


def execute_production(config: dict, output: Path, resume: bool) -> dict:
    root=output/"seed_0"; resolved=_resolved(config,root)
    baseline=run_production(resolved,"baseline",root/"baseline",resume=resume)
    delta_config={**resolved,"output_directory":str(root/"deltasub")}
    _ensure_deltasub_stages(delta_config,resume=resume)
    delta=run_production(delta_config,"deltasub",root/"deltasub",resume=resume)
    def metrics(value):
        return {"gcd_all_v2":value["metrics"]["all"],"gcd_old_v2":value["metrics"]["old"],"gcd_new_v2":value["metrics"]["new"],"unit":"fraction"}
    return {"metrics":{"vit_dinov2_selex":metrics(baseline),"deltasub":metrics(delta)},
            "run_directories":{"vit_dinov2_selex":str(root/"baseline"),"deltasub":str(root/"deltasub")}}
