from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .sampling import target_sign
from .schema import RouterExample
from ..utils.hashing import stable_hash


@dataclass
class ReplayEntry:
    gain_record_key: str
    sample_id: str
    target_sign: str
    labelled: bool
    parent_region: str
    priority: float
    insertion_order: int


class DeterministicReplayBuffer:
    def __init__(self, capacity: int, gain_cache_id: str, configuration_hash: str,
                 *, mode: str = "stratified_priority", near_zero_epsilon: float = 1e-8):
        if capacity <= 0 or mode not in {"fifo", "priority", "stratified_priority"}:
            raise ValueError("invalid replay configuration")
        self.capacity, self.gain_cache_id = capacity, gain_cache_id
        self.configuration_hash, self.mode = configuration_hash, mode
        self.near_zero_epsilon = near_zero_epsilon
        self.entries: dict[str, ReplayEntry] = {}
        self.next_order = 0

    def add(self, example: RouterExample, priority: float = 0.) -> None:
        if example.split_assignment != "train":
            raise ValueError("replay forbids validation/test records")
        if example.gain_cache_id != self.gain_cache_id:
            raise ValueError("replay gain-cache mismatch")
        if example.gain_record_key in self.entries:
            raise ValueError("duplicate replay record key")
        region = f"{example.parent_row // 4},{example.parent_column // 4}"
        self.entries[example.gain_record_key] = ReplayEntry(
            example.gain_record_key, example.sample_id,
            target_sign(example.target_gain, self.near_zero_epsilon), example.labelled,
            region, float(priority), self.next_order,
        )
        self.next_order += 1
        if len(self.entries) > self.capacity:
            self._evict()

    def _evict(self) -> None:
        values = list(self.entries.values())
        if self.mode == "fifo":
            victim = min(values, key=lambda x: (x.insertion_order, x.gain_record_key))
        else:
            counts = {sign: sum(x.target_sign == sign for x in values)
                      for sign in ("positive", "negative", "near_zero")}
            over = max(counts, key=lambda sign: (counts[sign], sign))
            pool = [x for x in values if self.mode != "stratified_priority" or x.target_sign == over]
            victim = min(pool, key=lambda x: (x.priority, x.insertion_order, x.gain_record_key))
        del self.entries[victim.gain_record_key]

    def update_priority(self, key: str, value: float) -> None:
        if key not in self.entries:
            raise KeyError(key)
        if value < 0:
            raise ValueError("priority must be nonnegative")
        self.entries[key].priority = float(value)

    @property
    def checksum(self) -> str:
        return stable_hash(self.state_dict(include_checksum=False))

    def state_dict(self, *, include_checksum: bool = True) -> dict:
        value = {
            "schema_version": 1, "capacity": self.capacity,
            "gain_cache_id": self.gain_cache_id,
            "configuration_hash": self.configuration_hash, "mode": self.mode,
            "near_zero_epsilon": self.near_zero_epsilon, "next_order": self.next_order,
            "entries": [asdict(x) for x in sorted(self.entries.values(), key=lambda y: y.insertion_order)],
        }
        if include_checksum:
            value["checksum"] = stable_hash(value)
        return value

    def load_state_dict(self, value: dict) -> None:
        checksum = value.get("checksum")
        core = {k: v for k, v in value.items() if k != "checksum"}
        if checksum != stable_hash(core):
            raise ValueError("replay checksum mismatch")
        for field in ("capacity", "gain_cache_id", "configuration_hash", "mode", "near_zero_epsilon"):
            if core[field] != getattr(self, field):
                raise ValueError(f"replay {field} mismatch")
        entries = [ReplayEntry(**x) for x in core["entries"]]
        if len({x.gain_record_key for x in entries}) != len(entries) or len(entries) > self.capacity:
            raise ValueError("invalid replay contents")
        self.entries = {x.gain_record_key: x for x in entries}
        self.next_order = core["next_order"]

    @property
    def memory_bytes_estimate(self) -> int:
        return len(json.dumps(self.state_dict(), sort_keys=True).encode())
