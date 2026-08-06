"""RegretGCD: class-held-out learning to arbitrate GCD experts."""

from .core import (
    PrototypeState,
    RouterFeatures,
    RouterFit,
    apply_mapping,
    build_prototypes,
    build_router_features,
    feature_indices,
    fit_regret_router,
    fixed_alignment_paired_bootstrap,
    hmean,
    hungarian_mapping,
    known_centroid_scores,
    normalize_rows,
    random_matched_switch,
    regret_labels,
    route_predictions,
)

__all__ = [
    "PrototypeState",
    "RouterFeatures",
    "RouterFit",
    "apply_mapping",
    "build_prototypes",
    "build_router_features",
    "feature_indices",
    "fit_regret_router",
    "fixed_alignment_paired_bootstrap",
    "hmean",
    "hungarian_mapping",
    "known_centroid_scores",
    "normalize_rows",
    "random_matched_switch",
    "regret_labels",
    "route_predictions",
]
