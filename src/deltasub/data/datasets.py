from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticGCDDataset(Dataset):
    def __init__(self, size: int = 32, classes: int = 4, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, 3, 224, 224, generator=generator)
        self.labels = torch.arange(size) % classes
        self.labelled = (torch.arange(size) % 2) == 0

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "labelled": self.labelled[index],
            "sample_id": f"synthetic-{index:05d}",
        }
