import unittest

import numpy as np

from src.v6.branch_features import BRANCH_FEATURE_NAMES, compute_branch_feature_dict
from src.v6.branch_selector import BranchSelectorModel, rank_branches_with_selector
from src.v6.historical_retrieval import HistoricalRetrievalResult
from src.v6.regime_detection import REGIME_LABELS, detect_regime
from src.v6.volatility_constraints import build_volatility_envelopes


class V6SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current_row = {
            "close": 2500.0,
            "atr_14": 8.0,
            "atr_pct": 0.0025,
            "ema_cross": 0.4,
            "macd_hist": 0.35,
            "quant_trend_score": 0.42,
            "quant_route_prob_up": 0.58,
            "quant_route_prob_down": 0.18,
            "quant_route_prob_range": 0.14,
            "quant_route_prob_chop": 0.10,
            "quant_regime_strength": 0.64,
            "quant_transition_risk": 0.21,
            "quant_tail_risk": 0.17,
            "quant_vol_forecast": 0.0030,
            "quant_kalman_fair_value": 2497.5,
            "news_bias": 0.25,
            "crowd_bias": 0.18,
            "macro_bias": 0.22,
            "llm_market_bias": 0.24,
            "news_shock": 0.08,
            "displacement": 0.16,
            "volume_ratio": 1.2,
            "dist_to_high": 0.8,
            "dist_to_low": 1.2,
            "crowd_extreme": 0.55,
            "minutes_since_news": 18.0,
        }
        self.branches = [
            {
                "path_id": 1,
                "probability": 0.30,
                "branch_fitness": 0.72,
                "minority_guardrail": 0.12,
                "analog_confidence": 0.55,
                "predicted_prices": [2502.0, 2505.5, 2508.0],
            },
            {
                "path_id": 2,
                "probability": 0.18,
                "branch_fitness": 0.48,
                "minority_guardrail": 0.08,
                "analog_confidence": 0.31,
                "predicted_prices": [2499.0, 2496.0, 2492.5],
            },
            {
                "path_id": 3,
                "probability": 0.14,
                "branch_fitness": 0.20,
                "minority_guardrail": 0.06,
                "analog_confidence": 0.22,
                "predicted_prices": [2525.0, 2550.0, 2585.0],
            },
        ]

    def test_regime_detection_returns_normalized_probabilities(self):
        regime = detect_regime(self.current_row)
        self.assertIn(regime.dominant_regime, REGIME_LABELS)
        self.assertAlmostEqual(sum(regime.probabilities.values()), 1.0, places=4)
        self.assertGreaterEqual(regime.dominant_confidence, 0.0)

    def test_volatility_envelope_builds_expected_horizons(self):
        regime = detect_regime(self.current_row)
        envelopes = build_volatility_envelopes(self.current_row["close"], self.current_row, regime)
        self.assertEqual(set(envelopes.keys()), {5, 15, 30})
        self.assertGreater(envelopes[15].expected_move_abs, 0.0)
        self.assertLess(envelopes[15].lower_bound, envelopes[15].upper_bound)

    def test_branch_features_include_all_expected_columns(self):
        regime = detect_regime(self.current_row)
        envelopes = build_volatility_envelopes(self.current_row["close"], self.current_row, regime)
        retrieval = HistoricalRetrievalResult(support=16, similarity=0.61, directional_prior=0.34, hold_prior_15m=0.22, hold_prior_30m=0.28)
        feature_dict = compute_branch_feature_dict(self.branches[0], self.current_row, regime, envelopes[15], retrieval)
        self.assertEqual(set(feature_dict.keys()), set(BRANCH_FEATURE_NAMES))
        self.assertTrue(np.isfinite(np.asarray(list(feature_dict.values()), dtype=np.float32)).all())

    def test_selector_prefers_more_plausible_branch(self):
        regime = detect_regime(self.current_row)
        envelopes = build_volatility_envelopes(self.current_row["close"], self.current_row, regime)
        retrieval = HistoricalRetrievalResult(support=16, similarity=0.61, directional_prior=0.34, hold_prior_15m=0.22, hold_prior_30m=0.28)
        result = rank_branches_with_selector(self.branches, self.current_row, regime, envelopes, retrieval, selector=BranchSelectorModel())
        self.assertEqual(len(result.scores), len(self.branches))
        self.assertEqual(result.selected_branch_id, 1)
        self.assertGreater(result.selected_score, 0.0)
        self.assertEqual(len(result.rationale), len(self.branches))


if __name__ == "__main__":
    unittest.main()
