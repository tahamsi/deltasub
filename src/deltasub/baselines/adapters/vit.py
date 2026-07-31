from __future__ import annotations
import torch
from ..base import AdapterInput, BaselineAdapter
from ._common import evidence, fixture_block, output_from
from ...models.backbones.dinov2 import DINOV2_REVISION
from ...utils.hashing import stable_hash

class ViTAdapter(BaselineAdapter):
    method_id="vit_dinov2_selex"; display_name="ViT / DINOv2 + SelEx"
    implementation_label="official DINOv2 architecture common-protocol adapter"
    source_revision=DINOV2_REVISION; license_status="Apache-2.0 (DINOv2); MIT (SelEx)"
    required_checkpoint_provenance=("model_sha256",); required_source_provenance=("revision","source_hashes")
    def __init__(self, source_root=None):
        super().__init__(evidence("fixture validated; production requires a complete real DINOv2 checkpoint and protocol provenance",
                                  source=bool(source_root), revision=bool(source_root)))
    def prepare_tokens(self, inputs):
        inputs.validate(); self.require_available(inputs.mode)
        if inputs.images is not None: raise ValueError("fixture adapter accepts strict pre-transformer state only")
        tokens=torch.cat((inputs.prefix_tokens, inputs.parent_tokens),1)
        valid=torch.ones(tokens.shape[:2],dtype=torch.bool,device=tokens.device)
        lengths=torch.full((tokens.shape[0],),tokens.shape[1],dtype=torch.long,device=tokens.device)
        return tokens,valid,lengths
    def execute(self, prepared, *, execution_mode):
        tokens,valid,lengths=prepared; model=fixture_block(tokens.shape[-1],0,tokens.device)
        from ._common import execute_tokens
        return execute_tokens(tokens,valid,lengths,model,execution_mode)
    def run(self, inputs, head=None, *, execution_mode="padded"):
        tokens,valid,lengths=self.prepare_tokens(inputs); model=fixture_block(tokens.shape[-1],inputs.seed,tokens.device)
        plan=stable_hash({"method":self.method_id,"parents":list(range(256)),"prefix":inputs.prefix_tokens.shape[1]})
        meta=[{"prefix_tokens":inputs.prefix_tokens.shape[1],"parent_tokens":256,"added_tokens":0} for _ in inputs.sample_ids]
        records=[{"parents":list(range(256)),"adaptive":False} for _ in inputs.sample_ids]
        return output_from(self,inputs,tokens,valid,lengths,meta,records,"all 256 row-major DINOv2 parent tokens",[0]*len(meta),[0]*len(meta),model,head,execution_mode,plan)
