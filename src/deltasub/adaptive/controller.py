from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import torch

from .budget import BudgetSpec
from ..utils.hashing import stable_hash


@dataclass(frozen=True)
class ControllerConfig:
    mode: str
    budget: BudgetSpec
    threshold: float = 0.0
    initial_lambda: float = 0.0
    dual_lr: float = 0.0
    lambda_min: float = 0.0
    lambda_max: float = 1e6
    update_interval: int = 1

    def __post_init__(self):
        if self.mode not in {"fixed_k", "threshold_with_bounds", "dual_threshold"}:
            raise ValueError("unsupported controller mode")
        values = (self.threshold, self.initial_lambda, self.dual_lr,
                  self.lambda_min, self.lambda_max)
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("controller values must be finite")
        if self.dual_lr < 0 or self.update_interval < 1:
            raise ValueError("dual_lr must be nonnegative and update_interval positive")
        if not self.lambda_min <= self.initial_lambda <= self.lambda_max:
            raise ValueError("invalid lambda bounds")

    @property
    def configuration_hash(self) -> str:
        return stable_hash(asdict(self))


class BudgetController:
    """Positive violation raises lambda, hence raises the selection threshold."""
    SCHEMA_VERSION = 1

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.lambda_value = float(config.initial_lambda)
        self.update_count = 0
        self.accumulated_realized_usage = 0.0
        self.accumulated_sample_count = 0
        self.history: list[dict] = []
        self.bound_hit_count = 0

    def choose_k(self, scores: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if scores.shape[-1:] != (256,) or not torch.isfinite(scores).all():
            raise ValueError("finite [B, 256] scores required")
        available = (valid_mask.sum(1) if valid_mask is not None else
                     torch.full((scores.shape[0],), 256, device=scores.device))
        lo, hi = self.config.budget.min_k, self.config.budget.max_k
        if bool((available < lo).any()):
            raise ValueError("valid-parent count is below minimum K")
        hi_values = torch.minimum(torch.full_like(available, hi), available)
        if self.config.mode == "fixed_k":
            result = torch.full_like(available, int(self.config.budget.target_selected_parents))
        else:
            threshold = (self.config.threshold if self.config.mode == "threshold_with_bounds"
                         else self.lambda_value)
            active = scores > threshold
            if valid_mask is not None:
                active &= valid_mask
            result = active.sum(1)
        return torch.maximum(torch.minimum(result.to(torch.int64), hi_values), torch.full_like(available, lo))

    def accumulate(self, k: torch.Tensor, *, training: bool, optimizer_step_succeeded: bool) -> bool:
        if not training or not optimizer_step_succeeded or self.config.mode != "dual_threshold":
            return False
        for value in k.detach().cpu().tolist():
            self.accumulated_realized_usage += self.config.budget.usage(int(value))
            self.accumulated_sample_count += 1
        if self.accumulated_sample_count < self.config.update_interval:
            return False
        realized = self.accumulated_realized_usage / self.accumulated_sample_count
        violation = realized - self.config.budget.target
        previous = self.lambda_value
        self.lambda_value = min(
            self.config.lambda_max,
            max(self.config.lambda_min, previous + self.config.dual_lr * violation),
        )
        if self.lambda_value in (self.config.lambda_min, self.config.lambda_max):
            self.bound_hit_count += 1
        self.update_count += 1
        self.history.append({"update": self.update_count, "previous_lambda": previous,
                             "realized_usage": realized, "target_usage": self.config.budget.target,
                             "violation": violation, "lambda": self.lambda_value})
        self.accumulated_realized_usage = 0.0
        self.accumulated_sample_count = 0
        return True

    def state_dict(self) -> dict:
        value = {
            "schema_version": self.SCHEMA_VERSION, "lambda_value": self.lambda_value,
            "update_count": self.update_count, "target_budget": self.config.budget.target,
            "budget_unit": self.config.budget.unit.value, "learning_rate": self.config.dual_lr,
            "minimum_lambda": self.config.lambda_min, "maximum_lambda": self.config.lambda_max,
            "averaging_interval": self.config.update_interval,
            "accumulated_realized_usage": self.accumulated_realized_usage,
            "accumulated_sample_count": self.accumulated_sample_count,
            "configuration_hash": self.config.configuration_hash,
            "history": self.history, "bound_hit_count": self.bound_hit_count,
        }
        value["state_hash"] = stable_hash(value)
        return value

    def load_state_dict(self, value: dict) -> None:
        expected = value.get("state_hash")
        if stable_hash({k: v for k, v in value.items() if k != "state_hash"}) != expected:
            raise ValueError("controller state hash mismatch")
        if value.get("schema_version") != self.SCHEMA_VERSION or value.get(
                "configuration_hash") != self.config.configuration_hash:
            raise ValueError("incompatible controller state")
        self.lambda_value = float(value["lambda_value"])
        self.update_count = int(value["update_count"])
        self.accumulated_realized_usage = float(value["accumulated_realized_usage"])
        self.accumulated_sample_count = int(value["accumulated_sample_count"])
        self.history = list(value["history"])
        self.bound_hit_count = int(value["bound_hit_count"])
