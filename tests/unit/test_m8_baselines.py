import math
import pytest
import torch
from deltasub.baselines.base import AdapterStatus, BaselineAdapter, StatusEvidence
from deltasub.baselines.config import load_config
from deltasub.baselines.fixture import synthetic_input
from deltasub.baselines.registry import build_registry
from deltasub.baselines.schema import ComparisonRecord, optional

def test_registry_contract_and_evidence_status():
    registry=build_registry(); assert len(registry)==5==len(set(registry))
    assert all(isinstance(x,BaselineAdapter) and x.status==AdapterStatus.FIXTURE_ONLY for x in registry.values())
    assert StatusEvidence(False,False,False,False,True,True,True,False,"license").status==AdapterStatus.BLOCKED

def test_config_and_undefined_schema():
    c=load_config("configs/smoke/m8_baselines.yaml"); assert c["common_protocol"]["parent_tokens"]==256
    assert optional(None,"not measured")=={"value":None,"reason":"not measured"}
    with pytest.raises(ValueError): optional(None)
    with pytest.raises(ValueError): optional(math.nan,"bad")

def test_vit_exact_semantics_and_no_fallback():
    a=build_registry()["vit_dinov2_selex"]; x=synthetic_input(token_budget=3); y=a.run(x)
    assert y.effective_lengths.tolist()==[257]*3
    assert all(z.retained_parent_count==256 and z.added_token_count==z.removed_token_count==0 for z in y.accounting)
    x.mode="production"
    with pytest.raises(RuntimeError,match="fixture_only"): a.run(x)

def test_method_specific_token_semantics():
    x=synthetic_input(token_budget=3); r=build_registry()
    d=r["deltasub"].run(x); s=r["subvit_reimplementation"].run(x)
    assert d.effective_lengths.tolist()==[266]*3 and all(a.added_token_count==9 for a in d.accounting)
    assert s.effective_lengths.tolist()==[269]*3 and all(a.added_token_count==12 for a in s.accounting)
    assert "Haar" in d.accounting[0].token_semantics and "direct" in s.accounting[0].token_semantics

def test_msvit_and_dart_distinct_complete_regions():
    x=synthetic_input(token_budget=4); r=build_registry()
    m=r["msvit_gcd_reimplementation"].run(x); d=r["dart_gcd_port"].run(x)
    assert m.plan_records[0]["complete_coverage"]
    assert d.plan_records[0]["regions"][0]["kind"]=="synthetic_row_band"
    assert m.selection_plan_hash != d.selection_plan_hash

def test_input_rejections():
    x=synthetic_input(); x.valid_parent_mask[:,0]=False
    with pytest.raises(ValueError): build_registry()["vit_dinov2_selex"].run(x)
