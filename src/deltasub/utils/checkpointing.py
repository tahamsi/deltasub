from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(value: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path) -> Any:
    return torch.load(Path(path), map_location="cpu", weights_only=False)
