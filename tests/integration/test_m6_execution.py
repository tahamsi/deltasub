from __future__ import annotations

import pytest
import torch

from deltasub.adaptive.assembly import assemble_adaptive
from deltasub.adaptive.execution import MaskedSelfAttentionBlock, execute_adaptive
from deltasub.adaptive.metrics import maximum_valid_output_error
from deltasub.adaptive.selection import deterministic_select


def make_sequence(counts=(0, 1, 8, 256), prefix=5, device="cpu", dtype=torch.float32):
    torch.manual_seed(61)
    b, d = len(counts), 8
    parents = torch.randn(b, 256, d, device=device, dtype=dtype)
    details = torch.randn(b, 256, 3, d, device=device, dtype=dtype)
    prefix_tokens = torch.randn(b, prefix, d, device=device, dtype=dtype)
    pp = torch.randn_like(parents)
    dp = pp[:, :, None].expand(-1, -1, 3, -1)
    selection = deterministic_select(torch.zeros(b, 256, device=device, dtype=dtype),
                                     list(counts))
    return assemble_adaptive(prefix_tokens, parents, details, prefix_tokens.clone(), pp, dp,
                             selection)


def test_padded_bucketed_exact_length_and_boundaries_restore_order_masks():
    sequence = make_sequence()
    model = MaskedSelfAttentionBlock(8, 2).eval()
    padded = execute_adaptive(model, sequence, mode="padded")
    exact = execute_adaptive(model, sequence, mode="bucketed")
    boundary = execute_adaptive(model, sequence, mode="bucketed",
                                bucket_boundaries=(300, 500, 1100))
    assert maximum_valid_output_error(padded.outputs, exact.outputs, sequence.valid_mask) < 2e-6
    assert maximum_valid_output_error(padded.outputs, boundary.outputs, sequence.valid_mask) < 2e-6
    assert exact.effective_tokens == int(sequence.effective_lengths.sum())
    assert exact.padding_tokens == 0
    assert padded.padding_tokens > 0
    assert sum(map(len, exact.bucket_composition.values())) == 4


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_m6_cuda_padded_bucketed(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    prop = torch.cuda.get_device_properties(0)
    assert "A100" in prop.name and prop.total_memory >= 75 * 1024**3
    sequence = make_sequence((0, 3, 16), device="cuda:0", dtype=dtype)
    model = MaskedSelfAttentionBlock(8, 2).to("cuda:0", dtype=dtype).eval()
    padded = execute_adaptive(model, sequence, mode="padded")
    bucketed = execute_adaptive(model, sequence, mode="bucketed")
    tolerance = 2e-5 if dtype == torch.float32 else .05
    assert maximum_valid_output_error(padded.outputs, bucketed.outputs,
                                      sequence.valid_mask) <= tolerance
