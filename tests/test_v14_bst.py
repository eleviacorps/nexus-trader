from __future__ import annotations

import unittest

import numpy as np

from src.v14.bst import batch_branch_survival, branch_survival_score


class TestV14BST(unittest.TestCase):
    def test_zero_atr_returns_deterministic_score(self) -> None:
        score = branch_survival_score(np.asarray([1.0, 2.0, 3.0]), current_atr=0.0)
        self.assertEqual(score, 1.0)

    def test_short_path_returns_neutral(self) -> None:
        score = branch_survival_score(np.asarray([1.0]), current_atr=1.0)
        self.assertEqual(score, 0.5)

    def test_batch_shape(self) -> None:
        scores = batch_branch_survival([np.asarray([1.0, 2.0]), np.asarray([2.0, 1.0])], current_atr=1.0, n_perturbations=5)
        self.assertEqual(scores.shape, (2,))


if __name__ == "__main__":
    unittest.main()
