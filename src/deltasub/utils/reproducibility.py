from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RNGState:
    python: object
    numpy: tuple
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def capture_rng_state() -> RNGState:
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng_state(state: RNGState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda)


@contextlib.contextmanager
def preserve_rng_state():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)
