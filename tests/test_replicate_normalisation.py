"""Guard against interpreting the four-pair normalisation as an unconditional bound."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_replicate_normalisation import independent_profiles, mask_audit, score


class ReplicateNormalisationTest(unittest.TestCase):
    def test_equal_precision_recovers_known_latent_agreement(self):
        values = independent_profiles([.64, .64, .81, .81])
        self.assertAlmostEqual(score(values)["normalised_score"], 1.0, places=12)

    def test_independent_errors_can_exceed_one_with_unequal_precision(self):
        values = independent_profiles([.81, .09, .81, .09])
        signal = np.tile(np.repeat([-1., 1.], 16), 16)
        errors = values - signal[:, None]
        covariance = np.cov(np.column_stack([signal, errors]), rowvar=False, ddof=0)
        np.testing.assert_allclose(covariance - np.diag(np.diag(covariance)), 0, atol=1e-14)
        self.assertAlmostEqual(score(values)["normalised_score"], 4 / 3, places=12)
        selected = score(independent_profiles([.99, .49, .99, .49]))
        self.assertTrue(.60 <= selected["denominator"] < .78)
        self.assertGreater(selected["normalised_score"], 1)

    def test_pairwise_masks_are_not_a_common_gene_axis(self):
        result = mask_audit()
        self.assertEqual(result["pairwise_gene_counts"], [480] * 6)
        self.assertEqual(result["common_gene_count"], 448)


if __name__ == "__main__":
    unittest.main()
