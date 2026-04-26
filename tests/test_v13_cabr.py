import unittest

import numpy as np
import pandas as pd

from src.v13.cabr import build_cabr_pairs, derive_cabr_feature_columns, load_v13_candidate_frames


class CABRTests(unittest.TestCase):
    def test_build_cabr_pairs_within_regime(self) -> None:
        frame = pd.DataFrame(
            {
                'sample_id': [0, 1, 2, 3, 4, 5, 6, 7],
                'timestamp': pd.date_range('2024-01-01', periods=8, freq='h', tz='UTC'),
                'setl_target_net_unit_pnl': [1.0, -1.0, 0.5, 0.8, -0.5, 1.2, -1.2, 0.4],
                'regime_class': ['trending_up', 'trending_up', 'trending_up', 'trending_up', 'ranging', 'ranging', 'ranging', 'ranging'],
                'bcfe_return_1': [0.1, 0.2, 0.3, 0.25, 0.1, 0.0, -0.1, 0.05],
                'branch_entropy': [0.2, 0.1, 0.3, 0.15, 0.4, 0.5, 0.6, 0.45],
                'context_regime_confidence': [0.8, 0.8, 0.8, 0.78, 0.6, 0.6, 0.6, 0.58],
                'context_rsi_14': [55, 56, 54, 57, 48, 49, 47, 46],
                'context_macd_hist': [0.1, 0.2, 0.1, 0.15, -0.1, -0.2, -0.1, -0.05],
                'context_bb_pct': [0.7, 0.6, 0.8, 0.75, 0.3, 0.2, 0.4, 0.35],
                'context_atr_percentile_30d': [0.5, 0.6, 0.7, 0.65, 0.4, 0.4, 0.4, 0.45],
                'context_days_since_regime_change': [1, 1, 1, 1, 0, 0, 0, 0],
                'context_emotional_momentum': [0.1, 0.1, 0.1, 0.12, -0.1, -0.1, -0.1, -0.08],
                'context_emotional_fragility': [0.3] * 8,
                'context_emotional_conviction': [0.6] * 8,
                'context_narrative_age': [2] * 8,
                'context_regime_trending_up': [1, 1, 1, 1, 0, 0, 0, 0],
                'context_regime_trending_down': [0] * 8,
                'context_regime_ranging': [0, 0, 0, 0, 1, 1, 1, 1],
                'context_regime_breakout': [0] * 8,
                'context_regime_panic_shock': [0] * 8,
                'context_regime_low_volatility': [0] * 8,
            }
        )
        pair_payload = build_cabr_pairs(
            frame,
            branch_feature_names=('bcfe_return_1', 'branch_entropy'),
            context_feature_names=(
                'context_regime_confidence',
                'context_rsi_14',
                'context_macd_hist',
                'context_bb_pct',
                'context_atr_percentile_30d',
                'context_days_since_regime_change',
                'context_emotional_momentum',
                'context_emotional_fragility',
                'context_emotional_conviction',
                'context_narrative_age',
                'context_regime_trending_up',
                'context_regime_trending_down',
                'context_regime_ranging',
                'context_regime_breakout',
                'context_regime_panic_shock',
                'context_regime_low_volatility',
            ),
        )
        self.assertGreater(len(pair_payload['pairs']), 0)
        self.assertEqual(pair_payload['branch_features'].shape[1], 2)


if __name__ == '__main__':
    unittest.main()

