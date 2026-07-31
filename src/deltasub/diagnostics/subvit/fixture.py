from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import torch

from .ats import assemble_ats, interpolate_child_positions
from .degradation import evaluate_head_degradation
from .metrics import compare_maps
from .training import train_router_fixture
from ...utils.hashing import stable_hash


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_fixture(output: str | Path, *, resume: bool = False) -> dict:
    torch.manual_seed(17)
    b, d, p, h, k = 3, 8, 3, 4, 4
    prefix = torch.randn(b, p, d); parents = torch.randn(b, 256, d)
    children = parents[:, :, None] + torch.arange(4).view(1, 1, 4, 1) / 10
    prefix_pos = torch.randn(b, p, d); parent_pos = torch.randn(b, 256, d)
    child_pos = interpolate_child_positions(parent_pos)
    attention = torch.randn(b, h, 256)
    ats = assemble_ats(prefix, parents, children, prefix_pos, parent_pos, child_pos,
                       attention[:, 0], torch.tensor([0, 1, 256]))
    frozen_teacher = torch.nn.Linear(d, d, bias=False)
    for parameter in frozen_teacher.parameters():
        parameter.requires_grad_(False)
    teacher_before = stable_hash({k: v.tolist() for k, v in frozen_teacher.state_dict().items()})

    def teacher(tokens, valid):
        weights = valid.float(); pooled = (tokens * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True)
        return frozen_teacher(pooled)
    degradation = evaluate_head_degradation(parents, attention, k, teacher, chunk_size=2)
    chosen = torch.stack([attention[i, degradation.selected_heads[i]] for i in range(b)])
    training = train_router_fixture(parents, chosen, Path(output) / "training", resume=resume, k=k)
    teacher_after = stable_hash({k: v.tolist() for k, v in frozen_teacher.state_dict().items()})
    report = {
        "label": "SYNTHETIC DIAGNOSTIC NON-REPORTABLE", "synthetic_only": True,
        "official_subvit_implementation": False, "paper_reproduction_claimed": False,
        "ats_effective_lengths": ats.effective_lengths.tolist(),
        "selected_heads": list(degradation.selected_heads),
        "feature_degradation_distances": degradation.distances,
        "per_head_selection_frequency": {str(q): list(degradation.selected_heads).count(q) / b for q in range(h)},
        "token_semantics": {"factor": 2, "subvit_direct_children_per_selected_parent": 4,
                            "deltasub_haar_details_per_selected_parent": 3,
                            "mechanisms_equivalent": False},
        "mean_attention_comparison": compare_maps(chosen[0], attention[0].mean(0), k),
        "teacher_hash_before": teacher_before, "teacher_hash_after": teacher_after,
        "teacher_equal": teacher_before == teacher_after, "training": training,
        "m4_m6_production_decisions_modified": False,
    }
    hashable = json.loads(json.dumps(report))
    hashable["training"]["checkpoint"] = "checkpoint_last.pt"
    report["deterministic_hash"] = stable_hash(hashable)
    _atomic_json(report, Path(output) / "diagnostic_comparison.json")
    return report
