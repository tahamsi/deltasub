from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from ..utils.hashing import stable_hash


@dataclass(frozen=True)
class RouterConfig:
    input_dim: int = 768
    hidden_dim: int = 256
    depth: int = 2
    normalization: str = "layernorm"
    dropout: float = 0.0
    coordinate_features: bool = True
    global_context_features: bool = True
    learned_position_embedding: bool = False
    test_only: bool = False

    def validate(self) -> None:
        if self.input_dim != 768 and not self.test_only:
            raise ValueError("production router requires 768-dimensional DINOv2 parents")
        if self.input_dim <= 0 or self.hidden_dim <= 0 or self.depth < 1:
            raise ValueError("invalid router dimensions")
        if self.normalization not in {"layernorm", "none"}:
            raise ValueError("normalization must be layernorm or none")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class GainRouter(nn.Module):
    """O(256D) shared per-parent predictor; it never selects or inserts tokens."""

    def __init__(self, config: RouterConfig, *, seed: int = 0):
        super().__init__()
        config.validate()
        self.config = config
        width = config.input_dim
        width += config.input_dim if config.global_context_features else 0
        width += 2 if config.coordinate_features else 0
        self.position = (
            nn.Embedding(256, config.input_dim) if config.learned_position_embedding else None
        )
        if self.position is not None:
            width += config.input_dim
        layers: list[nn.Module] = []
        for index in range(config.depth):
            incoming = width if index == 0 else config.hidden_dim
            layers.append(nn.Linear(incoming, config.hidden_dim))
            if config.normalization == "layernorm":
                layers.append(nn.LayerNorm(config.hidden_dim))
            layers.extend((nn.GELU(), nn.Dropout(config.dropout)))
        layers.append(nn.Linear(config.hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        # fork_rng prevents construction from perturbing caller RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02)

    @staticmethod
    def coordinates(device=None, dtype=None) -> torch.Tensor:
        indices = torch.arange(256, device=device)
        row, column = indices.div(16, rounding_mode="floor"), indices.remainder(16)
        return torch.stack((row, column), -1).to(dtype=dtype or torch.float32).div(15).mul(2).sub(1)

    def construct_features(
        self, parents: torch.Tensor, validity_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if parents.ndim != 3 or parents.shape[1:] != (256, self.config.input_dim):
            raise ValueError(f"parents must have shape [B, 256, {self.config.input_dim}]")
        if not parents.is_floating_point():
            raise TypeError("parent embeddings must be floating point")
        batch = parents.shape[0]
        if validity_mask is None:
            validity_mask = torch.ones((batch, 256), dtype=torch.bool, device=parents.device)
        if validity_mask.shape != (batch, 256) or validity_mask.dtype != torch.bool:
            raise ValueError("validity_mask must be bool [B, 256]")
        features = [parents]
        if self.config.global_context_features:
            denominator = validity_mask.sum(1, keepdim=True).clamp_min(1).to(parents.dtype)
            pooled = (parents * validity_mask.unsqueeze(-1)).sum(1) / denominator
            features.append(pooled[:, None].expand(-1, 256, -1))
        if self.config.coordinate_features:
            features.append(self.coordinates(parents.device, parents.dtype)[None].expand(batch, -1, -1))
        if self.position is not None:
            features.append(self.position(torch.arange(256, device=parents.device))[None].expand(batch, -1, -1))
        return torch.cat(features, -1)

    def forward(self, parents: torch.Tensor, validity_mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.construct_features(parents, validity_mask)
        scores = self.network(features).squeeze(-1)
        if scores.shape != (parents.shape[0], 256):
            raise RuntimeError("router output contract failed")
        return scores

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def multiply_add_estimate(self, batch_size: int = 1) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total = 0
        for module in self.network:
            if isinstance(module, nn.Linear):
                total += batch_size * 256 * module.in_features * module.out_features
        return total

    @property
    def configuration_hash(self) -> str:
        return stable_hash(asdict(self.config))
