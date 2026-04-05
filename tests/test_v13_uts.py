import unittest

import numpy as np

from src.v13.uts import UTSThresholdSelector, derive_contradiction_type, unified_trade_score


class UTSTests(unittest.TestCase):
    def test_unified_trade_score_range(self) -> None:
        score = unified_trade_score(0.7, 0.62, 0.55, 0.60, 'short_term_contrary', 0.2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_threshold_selector(self) -> None:
        selector = UTSThresholdSelector()
        scores = np.linspace(0.1, 0.9, 100, dtype=np.float32)
        regimes = ['trending_up'] * 100
        outcomes = np.where(scores > 0.5, 1.0, -1.0).astype(np.float32)
        selector.fit(scores, regimes, outcomes)
        self.assertTrue(selector.should_trade(0.9, 'trending_up'))
        self.assertFalse(selector.should_trade(0.1, 'trending_up'))

    def test_contradiction_type(self) -> None:
        self.assertEqual(derive_contradiction_type(1, 1, 'trending_up', 0.8), 'full_agreement_bull')


if __name__ == '__main__':
    unittest.main()
