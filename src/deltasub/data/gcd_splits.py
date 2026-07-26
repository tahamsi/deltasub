from __future__ import annotations


def deterministic_class_split(class_names: list[str], known_fraction: float = 0.5) -> dict:
    if not 0 < known_fraction < 1:
        raise ValueError("known_fraction must lie strictly between zero and one")
    ordered = sorted(class_names)
    cut = round(len(ordered) * known_fraction)
    return {"known": ordered[:cut], "novel": ordered[cut:]}
