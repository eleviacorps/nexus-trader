from __future__ import annotations

import unittest

import numpy as np

from src.v21.raam import RetrievalAugmentedAnalogMemory


class V21RAAMTests(unittest.TestCase):
    def test_retrieval_returns_weighted_neighbors(self) -> None:
        memory = RetrievalAugmentedAnalogMemory(embedding_dim=4, n_neighbors=2)
        embeddings = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
        outcomes = [
            {"direction_15m": -1.0, "vol_next": 0.1, "regime_match": 0.0, "max_drawdown": 0.04},
            {"direction_15m": 1.0, "vol_next": 0.2, "regime_match": 1.0, "max_drawdown": 0.02},
            {"direction_15m": 1.0, "vol_next": 0.5, "regime_match": 0.0, "max_drawdown": 0.10},
        ]
        memory.build(embeddings, outcomes)
        retrieval = memory.retrieve(np.asarray([0.9, 0.9, 1.1, 1.0], dtype=np.float32), k=2)
        self.assertEqual(len(retrieval.indices), 2)
        self.assertAlmostEqual(sum(retrieval.weights), 1.0, places=4)

    def test_analog_prior_aggregates_outcomes(self) -> None:
        memory = RetrievalAugmentedAnalogMemory(embedding_dim=2, n_neighbors=2)
        memory.build(
            np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            [
                {"direction_15m": -1.0, "vol_next": 0.1, "regime_match": 0.0, "max_drawdown": 0.03},
                {"direction_15m": 1.0, "vol_next": 0.2, "regime_match": 1.0, "max_drawdown": 0.05},
            ],
        )
        prior = memory.get_analog_prior(np.asarray([1.0, 1.0], dtype=np.float32))
        self.assertIn("analog_direction_15m", prior)
        self.assertIn("analog_vol_next", prior)
        self.assertIn("analog_regime_match_rate", prior)
        self.assertIn("analog_max_drawdown", prior)


if __name__ == "__main__":
    unittest.main()
