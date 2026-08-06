from __future__ import annotations

import unittest

import numpy as np

from deltasub.regretgcd.core import (
    build_prototypes,
    feature_indices,
    fit_regret_router,
    fixed_alignment_paired_bootstrap,
    hungarian_mapping,
    prototype_view_scores,
    route_predictions,
)


class RegretGCDCoreTests(unittest.TestCase):
    def test_leave_one_out_prototype_score_is_finite(self):
        mean = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            dtype=np.float64,
        )
        labels = np.asarray([4, 4, 8, 8], dtype=np.int64)
        state = build_prototypes(mean, labels)
        views = np.stack((mean, mean), axis=1)
        scores = prototype_view_scores(
            views,
            state,
            own_cluster=state.inverse,
            temperature=0.1,
        )
        self.assertEqual(scores.shape, (4, 2, 2))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.array_equal(state.labels, np.asarray([4, 8])))

    def test_class_holdout_router_learns_unique_winners(self):
        rng = np.random.default_rng(7)
        classes = np.repeat(np.arange(20), 12)
        samples = np.asarray([f"s-{index}" for index in range(len(classes))])
        signal = rng.normal(size=len(classes))
        features = np.column_stack((signal, rng.normal(size=len(classes))))
        target = classes.copy()
        parametric = target.copy()
        prototype = target.copy()
        prototype[signal < 0] = (prototype[signal < 0] + 1) % 20
        parametric[signal >= 0] = (parametric[signal >= 0] + 1) % 20
        mapping = hungarian_mapping(
            np.concatenate((parametric, prototype)),
            np.concatenate((target, target)),
        )
        groups = {"all": ("signal", "noise")}
        indices = feature_indices(("signal", "noise"), groups=groups)
        fit, oof = fit_regret_router(
            features=features,
            feature_names=("signal", "noise"),
            target=target,
            class_ids=classes,
            sample_ids=samples,
            parametric_prediction=parametric,
            prototype_prediction=prototype,
            mapping=mapping,
            feature_index=indices,
            folds=5,
            seed=3,
            fold_kind="class",
        )
        routed = route_predictions(parametric, prototype, oof, fit.threshold)
        self.assertGreater(np.mean(routed == target), 0.90)
        self.assertIsNotNone(fit.oof_auc)
        self.assertGreater(fit.oof_auc, 0.90)

    def test_fixed_alignment_bootstrap_detects_clear_gain(self):
        target = np.tile(np.arange(10), 20)
        old = target < 5
        baseline = target.copy()
        baseline[::3] = (baseline[::3] + 1) % 10
        candidate = target.copy()
        interval = fixed_alignment_paired_bootstrap(
            target=target,
            old=old,
            baseline_prediction=baseline,
            candidate_prediction=candidate,
            draws=500,
            seed=11,
        )
        self.assertGreater(interval["point"], 0.20)
        self.assertGreater(interval["lower_95"], 0.0)


if __name__ == "__main__":
    unittest.main()
