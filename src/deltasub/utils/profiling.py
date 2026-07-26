from __future__ import annotations

import time
from contextlib import contextmanager

import torch


@contextmanager
def elapsed_timer(result: dict, key: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result[key] = time.perf_counter() - start


def peak_memory_bytes(device: torch.device) -> int:
    return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
