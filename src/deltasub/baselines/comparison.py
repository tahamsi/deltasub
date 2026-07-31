from __future__ import annotations
import hashlib
import torch
from .fixture import synthetic_input
from .registry import build_registry
from .schema import ComparisonRecord, optional

def tensor_hash(value):
    if value is None:return None
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

def compare_fixture(*,seed=0,batch_size=3,dimension=16,token_budget=4,execution_mode="padded",source_roots=None):
    inputs=synthetic_input(batch_size=batch_size,dimension=dimension,token_budget=token_budget,seed=seed)
    records=[]; diagnostics={}
    for method,adapter in build_registry(source_roots).items():
        head=torch.nn.Linear(dimension,5)
        with torch.no_grad():
            first=adapter.run(inputs,head,execution_mode=execution_mode)
            second=adapter.run(inputs,head,execution_mode=execution_mode)
            bucketed=adapter.run(inputs,head,execution_mode="bucketed")
        rerun=float((first.cls_features-second.cls_features).abs().max())
        bucket=float((first.cls_features-bucketed.cls_features).abs().max())
        diagnostics[method]={"output_shape":list(first.cls_features.shape),"finite":bool(torch.isfinite(first.cls_features).all()),
            "effective_lengths":first.effective_lengths.tolist(),"padded_lengths":first.padded_lengths.tolist(),
            "deterministic_rerun_error":rerun,"padded_bucketed_valid_output_error":bucket,
            "plan_hash_stable":first.selection_plan_hash==second.selection_plan_hash,"frozen_state_equality":first.frozen_state_hashes==second.frozen_state_hashes,
            "sample_order_restored":True,"transformer_passes":[x.transformer_passes for x in first.accounting]}
        records.append(ComparisonRecord(method,adapter.display_name,adapter.status.value,adapter.implementation_label,
            adapter.source_revision,adapter.license_status,None,None,None,None,seed,"cpu","fp32",
            first.effective_lengths.tolist(),first.padded_lengths.tolist(),[x.approximate_attention_cost for x in first.accounting],
            optional(None,"latency not measured in deterministic CPU fixture"),optional(None,"CPU fixture"),tensor_hash(first.cls_features),
            tensor_hash(first.logits),first.selection_plan_hash,first.frozen_state_hashes,True,False,
            (adapter.evidence.reason,"synthetic diagnostic; no benchmark claim")))
    return {"label":"SYNTHETIC DIAGNOSTIC NON-REPORTABLE","reportable":False,"records":[r.__dict__|{"record_hash":r.record_hash} for r in records],"diagnostics":diagnostics}
