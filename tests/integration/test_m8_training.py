import json
from deltasub.baselines.training import run_fixture_training

def test_training_updates_artifacts_and_exact_resume(tmp_path):
    first=run_fixture_training(tmp_path,seed=4)
    assert first["parameter_change_norm"]>0 and first["frozen_state_equality"]
    resumed=run_fixture_training(tmp_path,resume=True,seed=4)
    assert resumed["resumed_from_step"]==2 and resumed["parameter_change_norm"]==0
    for name in ("metrics.jsonl","environment.json","resolved_config.yaml","adapter_status.json","token_accounting.json","provenance.json","checkpoint.pt"):
        assert (tmp_path/name).is_file()
    assert json.loads((tmp_path/"provenance.json").read_text())["reportable"] is False
