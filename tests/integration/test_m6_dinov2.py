from __future__ import annotations

import os
import pytest
import torch

from deltasub.models.backbones.dinov2 import construct_official_vitb14
from deltasub.adaptive.selection import deterministic_select
from deltasub.router.model import GainRouter, RouterConfig


@pytest.mark.integration
@pytest.mark.parametrize("register_tokens", [0, 4])
def test_pinned_official_dinov2_pretransformer_selection_no_mutation(register_tokens):
    root = os.environ.get("DINOV2_SOURCE_ROOT")
    if not root:
        pytest.skip("DINOV2_SOURCE_ROOT not supplied")
    model, _ = construct_official_vitb14(root, register_tokens=register_tokens)
    model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        parents = model.patch_embed(torch.zeros(1, 3, 224, 224))
        scores = GainRouter(RouterConfig(), seed=6)(parents)
        selected = deterministic_select(scores, [3])
    assert selected.selected_indices[0] == tuple(selected.canonical_rank_order[0][:3])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
