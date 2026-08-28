import unittest
import numpy as np

from mojopqc_sca.evaluation.guessing_entropy import compute_ge_curve, compute_key_rank


class GuessingEntropyTests(unittest.TestCase):
    def test_rank_is_one_for_best_candidate(self):
        self.assertEqual(compute_key_rank(np.array([-3.0, -1.0, -2.0]), 1), 1)

    def test_curve_has_final_point(self):
        scores = np.log(np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]))
        curve = compute_ge_curve(scores, np.array([1, 2]), step=1)
        self.assertEqual(sorted(curve), [1, 2])
        self.assertGreaterEqual(curve[2], 1.0)
