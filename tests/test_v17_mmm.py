import unittest

import numpy as np
import pandas as pd

from src.v17.mmm import MultifractalMarketMemory


class V17MMMTests(unittest.TestCase):
    def test_compute_all_returns_expected_keys(self) -> None:
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0001, 0.001, 240)
        volatility = np.abs(rng.normal(0.0005, 0.0002, 240))
        mmm = MultifractalMarketMemory(window=120)
        result = mmm.compute_all(returns, volatility)
        self.assertIn("hurst_overall", result)
        self.assertIn("market_memory_regime", result)

    def test_rolling_features_appends_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "return_1": np.linspace(-0.002, 0.002, 80),
                "atr_pct": np.linspace(0.0005, 0.0015, 80),
            }
        )
        enriched = MultifractalMarketMemory(window=32).rolling_features(frame)
        self.assertIn("hurst_asymmetry", enriched.columns)
        self.assertEqual(len(enriched), len(frame))


if __name__ == "__main__":
    unittest.main()
