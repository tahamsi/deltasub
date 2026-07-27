from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatePlan:
    parents: tuple[int, ...]
    sample_limit: int | None = None
    batch_limit: int | None = None

    def __post_init__(self):
        if not self.parents:
            raise ValueError("candidate set must not be empty")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("duplicate candidate indices")
        if any(not isinstance(value, int) or not 0 <= value <= 255 for value in self.parents):
            raise ValueError("candidate indices must be in [0, 255]")
        if tuple(sorted(self.parents)) != self.parents:
            raise ValueError("candidate indices must be ascending")
        if self.sample_limit is not None and self.sample_limit <= 0:
            raise ValueError("sample limit must be positive")
        if self.batch_limit is not None and self.batch_limit <= 0:
            raise ValueError("batch limit must be positive")

    @classmethod
    def from_config(cls, value: dict) -> "CandidatePlan":
        mode = value.get("mode")
        if mode == "all":
            parents = tuple(range(256))
        elif mode == "list":
            raw = value.get("parent_indices", [])
            if len(set(raw)) != len(raw):
                raise ValueError("duplicate candidate indices")
            parents = tuple(sorted(raw))
        elif mode == "range":
            start, stop = value.get("start"), value.get("stop")
            if not isinstance(start, int) or not isinstance(stop, int) or start >= stop:
                raise ValueError("candidate range requires start < stop")
            parents = tuple(range(start, stop))
        else:
            raise ValueError("candidate mode must be all, list, or range")
        return cls(parents, value.get("sample_limit"), value.get("batch_limit"))

    def iter_keys(self, batches):
        seen = 0
        for batch_index, sample_ids in enumerate(batches):
            if self.batch_limit is not None and batch_index >= self.batch_limit:
                return
            for anchor, _ in enumerate(sample_ids):
                if self.sample_limit is not None and seen >= self.sample_limit:
                    return
                for parent in self.parents:
                    yield batch_index, anchor, parent
                seen += 1
