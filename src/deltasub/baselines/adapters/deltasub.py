from __future__ import annotations
import torch
from ..base import AdapterInput, BaselineAdapter
from ._common import evidence, fixture_block, output_from
from ...adaptive.selection import deterministic_select
from ...adaptive.assembly import assemble_adaptive
from ...models.backbones.dinov2 import DINOV2_REVISION

class DeltaSubAdapter(BaselineAdapter):
    method_id="deltasub"; display_name="DeltaSub"; implementation_label="native DeltaSub M3-M6 adapter"
    source_revision=DINOV2_REVISION; license_status="project code; DINOv2 Apache-2.0 integration"
    required_checkpoint_provenance=("model_sha256","router_sha256")
    def __init__(self, source_root=None): super().__init__(evidence("fixture validated; production requires validated M2 and M5 checkpoints",source=bool(source_root),revision=bool(source_root)))
    def prepare_tokens(self, inputs):
        inputs.validate(); self.require_available(inputs.mode)
        if inputs.child_tokens is None or inputs.child_tokens.shape != inputs.parent_tokens.shape[:2]+(4,inputs.parent_tokens.shape[-1]):
            raise ValueError("DeltaSub requires four direct children to derive three Haar details")
        from ...models.subtokens.haar import HaarDetails
        details=HaarDetails()(inputs.child_tokens)
        scores=inputs.parent_tokens.float().square().mean(-1)
        k=torch.full((len(inputs.sample_ids),),inputs.token_budget,dtype=torch.long,device=scores.device)
        selection=deterministic_select(scores,k,inputs.valid_parent_mask)
        detail_pos=inputs.parent_positions[:,:,None,:].expand_as(details)
        seq=assemble_adaptive(inputs.prefix_tokens,inputs.parent_tokens,details,inputs.prefix_positions,
                              inputs.parent_positions,detail_pos,selection,inputs.valid_parent_mask)
        return seq
    def execute(self, prepared, *, execution_mode):
        from ._common import execute_tokens
        model=fixture_block(prepared.tokens.shape[-1],0,prepared.tokens.device)
        return execute_tokens(prepared.tokens,prepared.valid_mask,prepared.effective_lengths,model,execution_mode)
    def run(self,inputs,head=None,*,execution_mode="padded"):
        seq=self.prepare_tokens(inputs); model=fixture_block(seq.tokens.shape[-1],inputs.seed,seq.tokens.device)
        meta=[]; records=[]; added=[]
        for indices in deterministic_select(inputs.parent_tokens.float().square().mean(-1),[inputs.token_budget]*len(inputs.sample_ids)).selected_indices:
            meta.append({"parents":256,"haar_details":3*len(indices),"detail_modes":["horizontal","vertical","diagonal"]})
            records.append({"selected_parent_indices":list(indices),"tie_policy":"descending score then ascending index"}); added.append(3*len(indices))
        return output_from(self,inputs,seq.tokens,seq.valid_mask,seq.effective_lengths,meta,records,
            "256 retained parents plus three Haar details per selected parent",added,[0]*len(added),model,head,execution_mode,seq.selection_plan_hash,overhead=256*inputs.parent_tokens.shape[-1])
