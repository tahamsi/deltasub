"""M4 deterministic counterfactual gain collection."""

from .counterfactual import BatchContext, CounterfactualGainEvaluator, GainEvaluation
from .schema import GainRecord, GAIN_SCHEMA_VERSION

__all__ = [
    "BatchContext", "CounterfactualGainEvaluator", "GainEvaluation",
    "GainRecord", "GAIN_SCHEMA_VERSION",
]
