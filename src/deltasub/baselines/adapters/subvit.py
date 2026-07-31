from __future__ import annotations
import torch
from ..base import AdapterInput, BaselineAdapter
from ._common import evidence, fixture_block, output_from
from ...diagnostics.subvit.ats import assemble_ats
from ...diagnostics.subvit.distillation import SubViTRouter, SubViTRouterConfig
from ...models.backbones.dinov2 import DINOV2_REVISION
from ...utils.hashing import stable_hash

class SubViTAdapter(BaselineAdapter):
    method_id="subvit_reimplementation"; display_name="SubViT-Reimplementation"
    implementation_label="clean-room paper-described SubViT diagnostic reimplementation"
    source_revision=None; license_status="clean-room project code; paper only, no verified official source license"
    required_checkpoint_provenance=("model_sha256","subvit_router_sha256","teacher_sha256")
    def __init__(self, source_root=None): super().__init__(evidence("fixture validated; no official SubViT source and real router/teacher checkpoints absent",source=bool(source_root),revision=bool(source_root)))
    def prepare_tokens(self,inputs):
        inputs.validate(); self.require_available(inputs.mode)
        d=inputs.parent_tokens.shape[-1]
        if inputs.child_tokens is None or inputs.child_tokens.shape != (len(inputs.sample_ids),256,4,d):
            raise ValueError("SubViT f=2 requires exactly four direct spatial children")
        router=SubViTRouter(SubViTRouterConfig(d,max(8,d),test_only=True),seed=inputs.seed).to(inputs.parent_tokens.device)
        scores=router(inputs.parent_tokens)
        child_pos=inputs.parent_positions[:,:,None,:].expand_as(inputs.child_tokens)
        return assemble_ats(inputs.prefix_tokens,inputs.parent_tokens,inputs.child_tokens,
            inputs.prefix_positions,inputs.parent_positions,child_pos,scores,inputs.token_budget,factor=2), router
    def execute(self,prepared,*,execution_mode):
        seq,_=prepared
        from ._common import execute_tokens
        return execute_tokens(seq.tokens,seq.valid_mask,seq.effective_lengths,fixture_block(seq.tokens.shape[-1],0,seq.tokens.device),execution_mode)
    def run(self,inputs,head=None,*,execution_mode="padded"):
        seq,router=self.prepare_tokens(inputs); model=fixture_block(seq.tokens.shape[-1],inputs.seed,seq.tokens.device)
        records=[{"selected_parent_indices":list(x),"children_per_parent":4,"teacher_forwards_at_inference":0} for x in seq.selected_indices]
        meta=[{"parents":256,"direct_spatial_children":4*len(x),"child_order":"row-major TL,TR,BL,BR"} for x in seq.selected_indices]
        plan=stable_hash({"method":self.method_id,"selected":seq.selected_indices,"factor":2})
        result=output_from(self,inputs,seq.tokens,seq.valid_mask,seq.effective_lengths,meta,records,
            "256 retained parents plus four direct f=2 children per selected parent",
            [4*len(x) for x in seq.selected_indices],[0]*len(records),model,head,execution_mode,plan,overhead=256*d if (d:=inputs.parent_tokens.shape[-1]) else 0)
        from ._common import state_hash
        result.frozen_state_hashes["subvit_router"]=state_hash(router)
        return result
