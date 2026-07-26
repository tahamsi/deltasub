from __future__ import annotations

from abc import ABC, abstractmethod


class BaselineAdapter(ABC):
    @abstractmethod
    def build_model(self, config): ...

    @abstractmethod
    def forward(self, images): ...

    @abstractmethod
    def train_step(self, batch): ...

    @abstractmethod
    def evaluate(self, batch): ...

    @abstractmethod
    def effective_token_count(self, batch): ...

    @abstractmethod
    def padded_token_count(self, batch): ...

    @abstractmethod
    def selection_metadata(self): ...

    @abstractmethod
    def router_or_tokenizer_latency(self): ...

    @abstractmethod
    def load_official_weights_if_compatible(self): ...

    @abstractmethod
    def provenance(self): ...

    @abstractmethod
    def limitations(self): ...

    @abstractmethod
    def integration_status(self): ...


class UnavailableBaseline(BaselineAdapter):
    def __init__(self, name: str, reason: str) -> None:
        self.name, self.reason = name, reason

    def _unavailable(self, *args, **kwargs):
        raise RuntimeError(f"{self.name} is unavailable: {self.reason}")

    build_model = forward = train_step = evaluate = _unavailable

    def effective_token_count(self, batch): return None
    def padded_token_count(self, batch): return None
    def selection_metadata(self): return {}
    def router_or_tokenizer_latency(self): return None
    def load_official_weights_if_compatible(self): return False
    def provenance(self): return {"name": self.name, "status": "unavailable"}
    def limitations(self): return [self.reason]
    def integration_status(self): return "unavailable"
