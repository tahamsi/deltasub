"""Clean-room mixed-scale fixture; no paper-specific equations are asserted."""
from __future__ import annotations
import torch
from ..base import AdapterInput, BaselineAdapter
from ._common import evidence, fixture_block, output_from
from ...utils.hashing import stable_hash

class MSViTAdapter(BaselineAdapter):
    method_id="msvit_gcd_reimplementation"; display_name="MSViT-GCD-Reimplementation"
    implementation_label="clean-room common-protocol mixed-scale fixture"
    source_revision="483c2bc76cb215a489594be6009f06ad144c675f"
    license_status="batch-shaping utility: Qualcomm BSD-3-Clause-like; no patent license granted; implementation clean-room"
    def __init__(self,source_root=None): super().__init__(evidence("fixture mixed-scale coverage validated; upstream exposes no complete end-to-end MSViT model",source=bool(source_root),revision=bool(source_root)))
    def _regions(self,k):
        # Adaptation decision: replace deterministic 2x2 parent blocks by their mean.
        merges=max(0,min(64,256-k)); chosen=tuple(range(merges))
        covered=set(); regions=[]
        for q in chosen:
            r,c=divmod(q,8); cells=(2*r*16+2*c,2*r*16+2*c+1,(2*r+1)*16+2*c,(2*r+1)*16+2*c+1)
            covered.update(cells); regions.append(cells)
        regions.extend((j,) for j in range(256) if j not in covered)
        return tuple(regions)
    def prepare_tokens(self,inputs):
        inputs.validate(); self.require_available(inputs.mode)
        regions=self._regions(inputs.token_budget); rows=[]
        for b in range(len(inputs.sample_ids)):
            spatial=torch.stack([inputs.parent_tokens[b,list(r)].mean(0) for r in regions])
            rows.append(torch.cat((inputs.prefix_tokens[b],spatial)))
        tokens=torch.stack(rows); valid=torch.ones(tokens.shape[:2],dtype=torch.bool,device=tokens.device)
        lengths=torch.full((tokens.shape[0],),tokens.shape[1],dtype=torch.long,device=tokens.device)
        return tokens,valid,lengths,regions
    def execute(self,prepared,*,execution_mode):
        tokens,valid,lengths,_=prepared
        from ._common import execute_tokens
        return execute_tokens(tokens,valid,lengths,fixture_block(tokens.shape[-1],0,tokens.device),execution_mode)
    def run(self,inputs,head=None,*,execution_mode="padded"):
        tokens,valid,lengths,regions=self.prepare_tokens(inputs); model=fixture_block(tokens.shape[-1],inputs.seed,tokens.device)
        region_meta=[{"region_id":i,"parent_indices":list(r),"area_parent_units":len(r),"scale":"2x2" if len(r)==4 else "1x1"} for i,r in enumerate(regions)]
        records=[{"regions":region_meta,"complete_coverage":sorted(j for r in regions for j in r)==list(range(256))} for _ in inputs.sample_ids]
        plan=stable_hash({"method":self.method_id,"regions":regions,"adaptation":"deterministic top-left 2x2 merges"})
        retained=len(regions); removed=256-retained
        return output_from(self,inputs,tokens,valid,lengths,[{"mixed_scale_tokens":retained}]*len(records),records,
            "complete-coverage mixed 1x1/2x2 parent-region tokens",[0]*len(records),[removed]*len(records),model,head,execution_mode,plan,retained=retained,overhead=256)
