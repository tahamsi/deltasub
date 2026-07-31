from __future__ import annotations
from .base import TokenAccounting

def account(*, prefix: int, retained: int, added: int, removed: int, padded: int,
            semantics: str, passes: int = 1, overhead: int | None = 0) -> TokenAccounting:
    effective = prefix + retained + added
    if padded < effective: raise ValueError("padded tokens cannot be below effective tokens")
    return TokenAccounting(256, retained, added, removed, effective, padded, semantics,
                           passes, overhead, effective * effective)
