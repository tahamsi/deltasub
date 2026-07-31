from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pytest
import torch
from torch import nn

from deltasub.gains.counterfactual import BatchContext, CounterfactualGainEvaluator
from deltasub.models.backbones.dinov2 import (
    DINOv2Adapter, construct_official_vitb14, inspect_official_checkpoint,
)
from deltasub.models.subtokens.positions import ParentAwareDetailPositions
from deltasub.utils.hashing import sha256_file


def _source() -> Path:
    value = os.environ.get("DINOV2_SOURCE_ROOT")
    if not value:
        pytest.skip("DINOV2_SOURCE_ROOT pinned checkout not supplied")
    return Path(value)


@pytest.mark.integration
def test_m4_official_generated_state_strict_m3_path():
    model, _ = construct_official_vitb14(_source(), register_tokens=4)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "complete.pt"
        torch.save(model.state_dict(), checkpoint)
        loaded, report = inspect_official_checkpoint(
            checkpoint, sha256_file(checkpoint), source_root=_source()
        )
        adapter = DINOv2Adapter(loaded, checkpoint, report)
        child = adapter.build_child_projector(trainable=False)
        image = torch.zeros(1, 3, 224, 224)
        parent = adapter.pre_transformer_parent_embeddings(image)
        assert parent.shape == (1, 256, 768)
        assert adapter.prefix_token_count == 5
        assert child.initialization_sha256


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_m4_official_paired_counterfactual_cuda(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    props = torch.cuda.get_device_properties(0)
    if "A100" not in props.name or props.total_memory < 75 * 1024**3:
        pytest.skip("one NVIDIA A100 80 GB is required")
    model, _ = construct_official_vitb14(_source())
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "complete.pt"
        torch.save(model.state_dict(), checkpoint)
        loaded, report = inspect_official_checkpoint(
            checkpoint, sha256_file(checkpoint), source_root=_source()
        )
        adapter = DINOv2Adapter(loaded, checkpoint, report).to("cuda:0", dtype=dtype).eval()
        adapter.requires_grad_(False)
        child = adapter.build_child_projector(trainable=False)
        positions = ParentAwareDetailPositions().to("cuda:0", dtype=dtype).eval()
        positions.requires_grad_(False)
        head = nn.Identity()

        def encode(sequence):
            value = sequence.unsqueeze(0)
            for block in adapter.model.blocks:
                value = block(value)
            return adapter.model.norm(value)[0, 0]

        evaluator = CounterfactualGainEvaluator(adapter, child, positions, head, encode)
        images = torch.zeros(2, 2, 3, 224, 224, device="cuda:0", dtype=dtype)
        labels = torch.tensor([0, 0], device="cuda:0")
        labelled = torch.tensor([True, False], device="cuda:0")
        hierarchy = (torch.tensor([0, 0], device="cuda:0"),)
        confusion = torch.full((4, 4), .25, device="cuda:0", dtype=dtype)
        confidence = torch.ones(2, device="cuda:0", dtype=dtype)
        context = BatchContext.build(
            images=images, sample_ids=("a", "b"), view_ids=("v0", "v1"),
            augmentation_seeds=(0, 1), augmentation_parameters={"fixed": True},
            labels=labels, labelled=labelled, hierarchy_labels=hierarchy,
            confusion_factor=confusion,
            pseudo_label_confidence=confidence, model=adapter, child_projector=child,
            position_module=positions, head=head, configuration_hash="0" * 64, precision=str(dtype),
            device="cuda:0", source_git_commit="integration",
        )
        before = context.model_state_hash
        result = evaluator.evaluate(
            images=images, labels=labels, labelled=labelled,
            hierarchy_labels=hierarchy, confusion_factor=confusion,
            pseudo_label_confidence=confidence, context=context, anchor=0, parent=0,
        )
        assert torch.isfinite(torch.tensor(result.gain))
        assert result.counterfactual_effective_token_count == result.base_effective_token_count + 3
        assert BatchContext.build(
            images=images, sample_ids=("a", "b"), view_ids=("v0", "v1"),
            augmentation_seeds=(0, 1), augmentation_parameters={"fixed": True},
            labels=labels, labelled=labelled, hierarchy_labels=hierarchy,
            confusion_factor=confusion,
            pseudo_label_confidence=confidence, model=adapter, child_projector=child,
            position_module=positions, head=head, configuration_hash="0" * 64, precision=str(dtype),
            device="cuda:0", source_git_commit="integration",
        ).model_state_hash == before
