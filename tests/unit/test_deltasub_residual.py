from __future__ import annotations

import torch

from deltasub.models.deltasub_residual import (
    DeltaSubResidualConfig,
    LocalGridEncoder,
    bound_token_correction,
    norm_controlled_correction,
    remove_semantic_subspace,
)


def test_local_encoder_produces_parent_grid() -> None:
    encoder = LocalGridEncoder(
        feature_dim=32,
        local_channels=16,
    )

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = encoder(images)

    assert output.shape == (
        2,
        256,
        32,
    )
    assert torch.isfinite(output).all()


def test_semantic_subspace_is_removed() -> None:
    torch.manual_seed(7)

    residual = torch.randn(
        3,
        5,
        16,
    )
    global_direction = torch.randn(
        3,
        16,
    )
    local_direction = torch.randn(
        3,
        5,
        16,
    )

    output = remove_semantic_subspace(
        residual,
        (
            global_direction,
            local_direction,
        ),
    )

    global_normalized = torch.nn.functional.normalize(
        global_direction,
        dim=-1,
    )[:, None, :]

    overlap = (
        output.float()
        * global_normalized.float()
    ).sum(dim=-1)

    assert overlap.abs().max() < 1e-4


def test_token_correction_is_bounded() -> None:
    correction = torch.randn(
        2,
        256,
        32,
    ) * 100
    parents = torch.randn(
        2,
        256,
        32,
    )

    bounded, ratio = bound_token_correction(
        correction,
        parents,
        maximum_ratio=0.1,
    )

    assert bounded.shape == correction.shape
    assert ratio.shape == (
        2,
        256,
    )
    assert float(ratio.max()) <= 0.10001


def test_projection_requires_prediction_residual() -> None:
    config = DeltaSubResidualConfig(
        feature_dim=32,
        insertion_block=10,
        trainable_blocks=2,
        local_channels=16,
        predictor_hidden_dim=16,
        transport_rank=8,
        use_prediction_residual=False,
        use_semantic_projection=True,
    )

    try:
        config.validate()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "invalid configuration was accepted"
        )


def test_norm_controlled_correction_has_requested_ratio() -> None:
    torch.manual_seed(13)

    direction = torch.randn(
        2,
        256,
        32,
    )
    parents = torch.randn(
        2,
        256,
        32,
    )

    correction, ratio = norm_controlled_correction(
        direction,
        parents,
        maximum_ratio=0.1,
        scale=torch.tensor(0.05),
    )

    measured = (
        correction.float().norm(dim=-1)
        / parents.float().norm(dim=-1)
    )

    assert torch.allclose(
        ratio,
        torch.full_like(ratio, 0.005),
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        measured,
        ratio,
        atol=1e-5,
        rtol=1e-4,
    )
