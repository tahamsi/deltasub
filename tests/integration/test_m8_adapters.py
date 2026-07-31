import torch
from deltasub.baselines.comparison import compare_fixture
from deltasub.baselines.fixture import synthetic_input
from deltasub.baselines.registry import build_registry

def test_identical_batch_determinism_and_bucketed_equivalence():
    result=compare_fixture(seed=9,batch_size=2,dimension=8,token_budget=2)
    assert not result["reportable"]
    for value in result["diagnostics"].values():
        assert value["finite"] and value["deterministic_rerun_error"]==0
        assert value["padded_bucketed_valid_output_error"]==0
        assert value["plan_hash_stable"] and value["frozen_state_equality"]

def test_adapter_isolation_and_sample_order():
    x=synthetic_input(batch_size=2,dimension=8); before=x.parent_tokens.clone()
    for adapter in build_registry().values(): adapter.run(x)
    assert torch.equal(before,x.parent_tokens)
