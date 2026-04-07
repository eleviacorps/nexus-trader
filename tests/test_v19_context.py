from __future__ import annotations

import unittest

import pandas as pd

from src.v19.context_sampler import SimulationContextSampler, build_context_from_row, feature_map_from_context


class V19ContextTests(unittest.TestCase):
    def test_build_context_from_row_contains_required_sections(self) -> None:
        context = build_context_from_row(
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "anchor_price": 2050.0,
                "predicted_price_5m": 2051.0,
                "predicted_price_10m": 2052.0,
                "predicted_price_15m": 2054.0,
                "model_confidence_prob_15m": 0.7,
                "generator_probability": 0.65,
                "analog_similarity": 0.62,
                "branch_confidence": 0.63,
                "quant_route_confidence": 0.66,
                "dominant_regime": "trending_up",
                "hurst_overall": 0.58,
                "hurst_positive": 0.61,
                "hurst_negative": 0.49,
                "hurst_asymmetry": 0.12,
                "quant_trend_score": 0.2,
                "quant_kalman_fair_value": 2048.0,
                "retail_impact": 0.2,
                "institutional_impact": 0.15,
                "analog_disagreement": 0.1,
                "crowd_extreme": 0.2,
                "macro_bias": 0.1,
                "news_bias": 0.05,
                "crowd_bias": 0.02,
                "consensus_score": 0.4,
            }
        )
        self.assertIn("market", context)
        self.assertIn("simulation", context)
        self.assertIn("technical_analysis", context)
        self.assertIn("mfg", context)

    def test_feature_map_adds_categorical_flags(self) -> None:
        feature_map = feature_map_from_context(
            {
                "market": {"current_price": 2050.0},
                "simulation": {"direction": "BUY", "detected_regime": "trend_up", "sqt_label": "HOT"},
                "technical_analysis": {"structure": "bullish", "location": "discount"},
                "bot_swarm": {"aggregate": {"signal": "bullish"}},
                "sqt": {"label": "HOT"},
            }
        )
        self.assertEqual(feature_map["cat.direction.buy"], 1.0)
        self.assertEqual(feature_map["cat.regime.trend_up"], 1.0)
        self.assertEqual(feature_map["cat.sqt.hot"], 1.0)

    def test_sampler_works_with_custom_frame(self) -> None:
        frame = pd.DataFrame(
            [
                {"timestamp": "2024-01-01T00:00:00+00:00", "anchor_price": 2000.0, "predicted_price_15m": 2003.0, "dominant_regime": "trending_up"},
                {"timestamp": "2024-01-01T00:15:00+00:00", "anchor_price": 2001.0, "predicted_price_15m": 1998.0, "dominant_regime": "trending_down"},
                {"timestamp": "2024-01-01T00:30:00+00:00", "anchor_price": 2002.0, "predicted_price_15m": 2004.0, "dominant_regime": "trending_up"},
                {"timestamp": "2024-01-01T00:45:00+00:00", "anchor_price": 2003.0, "predicted_price_15m": 2000.0, "dominant_regime": "trending_down"},
            ]
        )
        sampler = SimulationContextSampler(source_frame=frame, seed=7)
        contexts = sampler.sample_contexts(n_samples=4, balance_regimes=True)
        self.assertEqual(len(contexts), 4)
        self.assertTrue(all("simulation" in item for item in contexts))


if __name__ == "__main__":
    unittest.main()
