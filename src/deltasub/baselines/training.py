from __future__ import annotations
import json, platform
from pathlib import Path
import torch, yaml
from .fixture import synthetic_input
from .registry import build_registry
from ..utils.checkpointing import atomic_torch_save, load_checkpoint
from ..utils.hashing import stable_hash

def run_fixture_training(output,method_id="vit_dinov2_selex",*,resume=False,seed=0):
    out=Path(output); out.mkdir(parents=True,exist_ok=True); adapter=build_registry()[method_id]
    inputs=synthetic_input(seed=seed); d=inputs.parent_tokens.shape[-1]
    with torch.random.fork_rng(devices=[]): torch.manual_seed(seed); head=torch.nn.Linear(d,5)
    optimizer=torch.optim.SGD(head.parameters(),lr=.01); checkpoint=out/"checkpoint.pt"; start=0
    if resume:
        state=load_checkpoint(checkpoint); head.load_state_dict(state["head"]); optimizer.load_state_dict(state["optimizer"]); start=state["step"]
    initial=torch.cat([p.detach().flatten() for p in head.parameters()]).clone(); adapter_frozen=None
    metrics=[]
    if resume and (out/"metrics.jsonl").is_file():
        metrics=[json.loads(line) for line in (out/"metrics.jsonl").read_text().splitlines() if line]
    for step in range(start,2):
        optimizer.zero_grad(); result=adapter.run(inputs,head); labels=torch.arange(3)%5
        if adapter_frozen is None: adapter_frozen=dict(result.frozen_state_hashes)
        elif adapter_frozen != result.frozen_state_hashes: raise RuntimeError("adapter frozen state changed")
        loss=torch.nn.functional.cross_entropy(result.logits,labels); loss.backward(); optimizer.step()
        metrics.append({"step":step+1,"loss":float(loss.detach()),"synthetic":True})
        atomic_torch_save({"step":step+1,"head":head.state_dict(),"optimizer":optimizer.state_dict(),"seed":seed},checkpoint)
    with torch.no_grad(): result=adapter.run(inputs,head)
    if adapter_frozen is None: adapter_frozen=dict(result.frozen_state_hashes)
    frozen_equal=adapter_frozen==result.frozen_state_hashes
    final=torch.cat([p.detach().flatten() for p in head.parameters()]); change=float(torch.linalg.vector_norm(final-initial))
    (out/"metrics.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in metrics))
    artifacts={"environment.json":{"python":platform.python_version(),"torch":str(torch.__version__)},
      "resolved_config.yaml":{"schema_version":"m8.training.v1","method_id":method_id,"seed":seed,"fixture":True},
      "adapter_status.json":{"status":adapter.status.value,"reason":adapter.evidence.reason},
      "token_accounting.json":{"samples":[x.__dict__ for x in result.accounting],"token_counts_are_not_flops":True},
      "provenance.json":{"fixture":True,"reportable":False,"frozen_state_hashes":result.frozen_state_hashes}}
    for name,value in artifacts.items():
        path=out/name; path.write_text(yaml.safe_dump(value) if name.endswith("yaml") else json.dumps(value,indent=2,sort_keys=True))
    return {"label":"SYNTHETIC DIAGNOSTIC NON-REPORTABLE","method_id":method_id,"steps":2,"resumed_from_step":start,
            "parameter_change_norm":change,"frozen_state_equality":frozen_equal,"checkpoint":str(checkpoint),"artifact_hash":stable_hash(artifacts)}
