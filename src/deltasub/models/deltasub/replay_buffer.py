from __future__ import annotations

from collections import defaultdict, deque


class StratifiedReplayBuffer:
    def __init__(self, capacity_per_stratum: int = 128) -> None:
        self.capacity = capacity_per_stratum
        self.data = defaultdict(lambda: deque(maxlen=self.capacity))

    @staticmethod
    def stratum(record: dict) -> str:
        if record.get("low_confidence", False):
            return "low_confidence"
        if record.get("disagreement", 0.0) > 0.5:
            return "high_disagreement"
        gain = record["normalized_gain"]
        if gain > 0.01:
            return "positive"
        if gain < -0.01:
            return "negative"
        return "near_zero"

    def add(self, record: dict) -> None:
        self.data[self.stratum(record)].append(record)

    def counts(self) -> dict[str, int]:
        return {key: len(value) for key, value in self.data.items()}
