"""Synthetic DART-shaped interface; production fails closed without pinned source."""
from __future__ import annotations
import torch
from ..base import AdapterInput, BaselineAdapter
from ._common import evidence, fixture_block, output_from
from ...utils.hashing import stable_hash

class DARTAdapter(BaselineAdapter):
    method_id="dart_gcd_port"; display_name="DART-GCD-Port"
    implementation_label="synthetic tokenizer interface fixture; not a verified faithful port"
    source_revision="df9e34ded12f3d7813b5004eccf922a67fb8491f"
    license_status="root Apache-2.0; README says MIT (recorded discrepancy)"
    required_source_provenance=("revision","adapted_symbols","source_file_hashes")
    def __init__(self,source_root=None): super().__init__(evidence("synthetic region-allocation interface only; exact pinned DART behavior/source hashes not locally verified",source=False,revision=False))
    def _regions(self,budget):
        # Fixture-only contiguous row bands, deliberately not top-K patch selection.
        bands=max(1,min(16,budget)); bounds=[round(i*16/bands) for i in range(bands+1)]
        return tuple(tuple(range(bounds[i]*16,bounds[i+1]*16)) for i in range(bands))
    def prepare_tokens(self,inputs):
        inputs.validate(); self.require_available(inputs.mode)
        regions=self._regions(inputs.token_budget); rows=[]
        for b in range(len(inputs.sample_ids)):
            generated=torch.stack([inputs.parent_tokens[b,list(r)].mean(0) for r in regions])
            rows.append(torch.cat((inputs.prefix_tokens[b],generated)))
        tokens=torch.stack(rows); valid=torch.ones(tokens.shape[:2],dtype=torch.bool,device=tokens.device)
        lengths=torch.full((tokens.shape[0],),tokens.shape[1],dtype=torch.long,device=tokens.device)
        return tokens,valid,lengths,regions
    def execute(self,prepared,*,execution_mode):
        tokens,valid,lengths,_=prepared
        from ._common import execute_tokens
        return execute_tokens(tokens,valid,lengths,fixture_block(tokens.shape[-1],0,tokens.device),execution_mode)
    def run(self,inputs,head=None,*,execution_mode="padded"):
        tokens,valid,lengths,regions=self.prepare_tokens(inputs); model=fixture_block(tokens.shape[-1],inputs.seed,tokens.device)
        rm=[{"region_id":i,"parent_indices":list(r),"area_parent_units":len(r),"kind":"synthetic_row_band"} for i,r in enumerate(regions)]
        records=[{"regions":rm,"fixture_mechanism":"synthetic, not claimed as exact DART"} for _ in inputs.sample_ids]
        plan=stable_hash({"method":self.method_id,"regions":regions,"synthetic":True})
        retained=len(regions)
        return output_from(self,inputs,tokens,valid,lengths,[{"generated_region_tokens":retained}]*len(records),records,
            "fixture quantile-like generated regions with explicit provenance",[retained]*len(records),[256]*len(records),model,head,execution_mode,plan,retained=0,overhead=256)
