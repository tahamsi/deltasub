import json
from pathlib import Path

import pytest
import yaml

from deltasub.diagnostic.campaign import load_config, summarize, verdict


def test_seed_and_metric_fail_closed(tmp_path):
    value = yaml.safe_load(Path("configs/diagnostic/cub_seed0.yaml").read_text())
    value["diagnostic"]["seed"] = 1
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="exactly 0"): load_config(path)
    value["diagnostic"]["seed"] = 0; value["diagnostic"]["primary_metric"] = "accuracy"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="gcd_all_v2"): load_config(path)


def test_verdict_margin_policy():
    assert verdict(.5, .51, .01) == "positive"
    assert verdict(.5, .49, .01) == "neutral"
    assert verdict(.5, .489, .01) == "negative"
    assert verdict(.5, .5, 0) == "neutral"


def test_campaign_cars_and_missing_results(tmp_path):
    report = summarize(tmp_path)
    assert report["verdict"] == "invalid"
    assert report["datasets"]["cars"] == {
        "status": "not_run", "reason": "dataset unavailable by user choice"
    }
    assert report["core_or_full_started"] is False
    assert json.loads((tmp_path / "campaign.json").read_text())["datasets"]["cars"]["status"] == "not_run"
