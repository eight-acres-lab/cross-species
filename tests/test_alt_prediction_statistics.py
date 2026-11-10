"""Regression checks for label-dependent permutation inference and tied scores."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import spearmanr

SPEC = importlib.util.spec_from_file_location(
    "alt_prediction", Path(__file__).resolve().parents[1] / "scripts/run_alt_prediction.py")
alt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alt)


class PredictionStatisticsTest(unittest.TestCase):
    def test_tied_auc_is_half_and_order_invariant(self):
        scores = np.ones(12)
        labels = np.arange(12) < 6
        for seed in range(5):
            order = np.random.default_rng(seed).permutation(12)
            self.assertEqual(alt.auc(scores[order], labels[order])[0], 0.5)

    def test_tied_spearman_matches_scipy(self):
        x = np.repeat([1., 2., 3., 4.], 3)
        y = np.array([2, 1, 4, 3, 3, 2, 7, 5, 6, 9, 8, 9.])
        self.assertAlmostEqual(alt.spearman(x, y), spearmanr(x, y).statistic)

    def test_label_dependent_fit_is_repeated_under_the_null(self):
        # Deliberate target leakage fits every permuted target perfectly too. Its
        # pipeline-level p must therefore be 1, rather than a small fixed-score p.
        y = np.arange(12, dtype=float)
        labels = y >= 6
        report, calls = {}, []

        def target_copy(permuted):
            calls.append(permuted.copy())
            return permuted.copy()

        with contextlib.redirect_stdout(io.StringIO()):
            alt.evaluate('leaking predictor', y.copy(), y, labels,
                         np.random.default_rng(7), 29, report,
                         predict_from_labels=target_copy)
        self.assertEqual(len(calls), 29)
        self.assertTrue(any(not np.array_equal(v, y) for v in calls))
        self.assertEqual(report['leaking predictor']['p_auc'], 1.)
        self.assertEqual(report['leaking predictor']['p_spearman'], 1.)

    def test_fixed_baseline_monte_carlo_p_is_nonzero(self):
        y = np.arange(12, dtype=float)
        report = {}
        with contextlib.redirect_stdout(io.StringIO()):
            alt.evaluate('independent score', y.copy(), y, y >= 6,
                         np.random.default_rng(7), 29, report)
        self.assertGreaterEqual(report['independent score']['p_auc'], 1 / 30)
        self.assertEqual(report['independent score']['permutation_method'],
                         'fixed_label_independent_scores')


if __name__ == '__main__':
    unittest.main()
