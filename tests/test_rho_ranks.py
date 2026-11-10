"""A taxonomy category must have one rank regardless of species-pair row order."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "rho", Path(__file__).resolve().parents[1] / "scripts/run_rho_reliability_and_phylogeny.py")
rho = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rho)


class RhoRanksTest(unittest.TestCase):
    def test_average_ties(self):
        np.testing.assert_allclose(rho.centered_ranks([3, 1, 1, 5]), [0.5, -1, -1, 1.5])

    def test_partial_is_invariant_to_pair_order(self):
        x = np.array([0.6, 0.2, 0.5, 0.3, 0.8, 0.1, 0.7, 0.4])
        y = np.array([0.4, 0.5, 0.1, 0.2, 0.7, 0.6, 0.8, 0.3])
        tax = np.array([2, 2, 3, 3, 4, 4, 5, 5])
        expected = rho.partial_spearman(x, y, tax)
        for seed in range(12):
            order = np.random.default_rng(seed).permutation(len(x))
            self.assertAlmostEqual(rho.partial_spearman(x[order], y[order], tax[order]),
                                   expected, places=13)

    def test_partial_matches_rank_residual_formula(self):
        # Explicit, independently specified ranks (taxonomy has repeated categories).
        x = np.array([1, 3, 2, 6, 4, 5], dtype=float)
        y = np.array([2, 1, 4, 3, 6, 5], dtype=float)
        z = np.array([2, 2, 4, 4, 4, 5], dtype=float)
        rz = np.array([1.5, 1.5, 4, 4, 4, 6])
        rxy, rxz, ryz = np.corrcoef([x, y, rz])[np.triu_indices(3, 1)]
        expected = (rxy-rxz*ryz) / np.sqrt((1-rxz**2)*(1-ryz**2))
        self.assertAlmostEqual(rho.partial_spearman(x, y, z), expected, places=13)

    def test_constant_taxonomy_leaves_spearman_unchanged(self):
        x, y = [1, 1, 2, 3, 4], [2, 1, 1, 3, 4]
        self.assertAlmostEqual(rho.partial_spearman(x, y, np.ones(5)), rho.spearman(x, y))

    def test_constant_response_has_undefined_correlation(self):
        self.assertTrue(np.isnan(rho.spearman(np.ones(5), np.arange(5))))
        self.assertTrue(np.isnan(rho.partial_spearman(np.ones(5), np.arange(5), [2, 2, 4, 4, 5])))


if __name__ == "__main__":
    unittest.main()
