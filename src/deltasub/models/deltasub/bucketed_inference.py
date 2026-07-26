from __future__ import annotations

from collections import defaultdict

import torch


def bucketed_forward(model, images: torch.Tensor, selections: list[torch.Tensor]) -> torch.Tensor:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, selected in enumerate(selections):
        groups[len(selected)].append(index)
    outputs = [None] * len(images)
    for _, indices in sorted(groups.items()):
        batch = images[indices]
        chosen = [selections[i] for i in indices]
        values = model(batch, chosen)
        for local, original in enumerate(indices):
            outputs[original] = values[local]
    return torch.stack(outputs)


def token_counts(selections: list[torch.Tensor], parent_count: int = 256) -> tuple[int, int]:
    effective = sum(parent_count + 3 * len(item) for item in selections)
    padded_length = max(parent_count + 3 * len(item) for item in selections)
    return effective, padded_length * len(selections)
