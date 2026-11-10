import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_tggates_knn_validation import retrieve


class RetrievalMaskTest(unittest.TestCase):
    def test_mask_survives_requested_k_larger_than_eligible_pool(self):
        values = np.array([[1., 2.], [3., 4.], [100., -100.]])
        scores = np.array([[.8, .7, -1e9]])
        pred = retrieve(scores, values, 20, 'mean', .1)
        np.testing.assert_allclose(pred, [[2., 3.]])

    def test_no_eligible_requires_explicit_fallback(self):
        with self.assertRaisesRegex(ValueError, 'no eligible'):
            retrieve(np.array([[-1e9, -1e9]]), np.ones((2, 3)), 2, 'mean', .1)


if __name__ == '__main__':
    unittest.main()
