from __future__ import annotations
import hashlib
from contextlib import nullcontext
import torch
from ...adaptive.execution import MaskedSelfAttentionBlock, execute_adaptive
from ...adaptive.assembly import AdaptiveSequence
from ..base import AdapterInput, AdapterOutput
from ..accounting import account
from ...utils.hashing import stable_hash

def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def fixture_block(d: int, seed: int, device) -> MaskedSelfAttentionBlock:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MaskedSelfAttentionBlock(d, heads=2 if d % 2 == 0 else 1).to(device)

def precision_context(tokens: torch.Tensor):
    """Use the production CUDA BF16 policy while retaining FP32 model weights."""
    if tokens.is_cuda and tokens.dtype == torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

def execute_tokens(tokens, valid, lengths, model, mode):
    dummy = AdaptiveSequence(tokens, torch.zeros_like(tokens), torch.zeros_like(valid, dtype=torch.int8),
        torch.full_like(valid, -1, dtype=torch.long), torch.full_like(valid, -1, dtype=torch.int8),
        valid, lengths, "")
    with precision_context(tokens):
        result = execute_adaptive(model, dummy, mode=mode)
    padded = torch.tensor([result.outputs.shape[1]] * tokens.shape[0], device=tokens.device)
    return result.outputs, padded

def output_from(adapter, inputs: AdapterInput, tokens, valid, lengths, metadata, records,
                semantics, added, removed, model, head, execution_mode, plan_hash, retained=256, overhead=0):
    before = state_hash(model)
    transformed, padded = execute_tokens(tokens, valid, lengths, model, execution_mode)
    after = state_hash(model)
    if before != after: raise RuntimeError("transformer state mutated during adapter inference")
    cls = transformed[:, 0]
    with precision_context(cls):
        logits = None if head is None else head(cls)
    prefix = inputs.prefix_tokens.shape[1]
    accounting = tuple(account(prefix=prefix, retained=retained, added=int(added[i]), removed=int(removed[i]),
        padded=int(padded[i]), semantics=semantics, overhead=overhead) for i in range(len(inputs.sample_ids)))
    return AdapterOutput(cls, transformed[:, 1:prefix], transformed[:, prefix:prefix+retained] if retained else None,
        logits, lengths, padded, tuple(metadata), tuple(records), accounting, plan_hash,
        {"transformer": before}, {**adapter.diagnostic_metadata(), "execution_mode": execution_mode}).finalize()

def evidence(reason, *, source=False, revision=False, checkpoint=False):
    from ..base import StatusEvidence
    return StatusEvidence(source, revision, True, checkpoint, True, True, True, False, reason)
