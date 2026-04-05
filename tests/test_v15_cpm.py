import unittest

import pandas as pd

from src.v15.cpm import ConditionalPredictabilityMapper


class V15CPMTests(unittest.TestCase):
    def test_score_row_rewards_strong_baseline_alignment(self) -> None:
        mapper = ConditionalPredictabilityMapper()
        row = {
            'return_1': 0.01,
            'rsi_14': 25.0,
            'ema_cross': 1.0,
            'macd_hist': 0.05,
            'bb_pct': 0.02,
            'volume_ratio': 2.0,
        }
        score = mapper.score_row(row)
        self.assertGreaterEqual(score['predictability'], 0.99)
        self.assertEqual(score['n_active'], 6)

    def test_label_archive_adds_expected_columns(self) -> None:
        mapper = ConditionalPredictabilityMapper()
        frame = pd.DataFrame(
            [
                {'return_1': 0.01, 'rsi_14': 25.0, 'ema_cross': 1.0, 'macd_hist': 0.05, 'bb_pct': 0.02, 'volume_ratio': 2.0},
                {'return_1': 0.02, 'rsi_14': 75.0, 'ema_cross': -1.0, 'macd_hist': -0.05, 'bb_pct': 0.98, 'volume_ratio': 2.1},
                {'return_1': -0.01, 'rsi_14': 50.0, 'ema_cross': 0.0, 'macd_hist': 0.0, 'bb_pct': 0.50, 'volume_ratio': 1.0},
            ]
        )
        labeled = mapper.label_archive(frame)
        self.assertIn('cpm_predictability', labeled.columns)
        self.assertIn('cpm_agreement', labeled.columns)
        self.assertEqual(len(labeled), 3)


if __name__ == '__main__':
    unittest.main()
