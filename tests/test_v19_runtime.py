from __future__ import annotations

import unittest

from src.v19.runtime import build_live_v19_candidate_frame, build_v19_runtime_state, infer_sqt_label


class V19RuntimeTests(unittest.TestCase):
    def test_infer_sqt_label_uses_expected_buckets(self):
        self.assertEqual(infer_sqt_label(0.30, 0.70), "COLD")
        self.assertEqual(infer_sqt_label(0.60, 0.60), "GOOD")
        self.assertEqual(infer_sqt_label(0.80, 0.80), "HOT")

    def test_build_live_candidate_frame_creates_rows(self):
        payload = {
            "symbol": "XAUUSD",
            "market": {"current_price": 2320.0},
            "simulation": {
                "cpm_score": 0.61,
                "consensus_score": 0.64,
                "overall_confidence": 0.66,
                "analog_confidence": 0.58,
                "cone_width_pips": 10.0,
                "contradiction_type": "mixed",
                "detected_regime": "balanced_range",
                "hurst_overall": 0.56,
                "hurst_positive": 0.58,
                "hurst_negative": 0.53,
                "hurst_asymmetry": 0.05,
                "crowd_bias": 0.12,
                "crowd_extreme": 0.22,
                "mean_probability": 0.57,
            },
            "technical_analysis": {
                "quant_regime_strength": 0.63,
                "quant_transition_risk": 0.28,
                "quant_vol_realism": 0.74,
                "quant_fair_value_z": -0.32,
                "rsi_14": 55.0,
            },
            "current_row": {
                "quant_route_confidence": 0.59,
                "atr_14": 3.4,
                "ema_cross": 0.11,
                "wltc_testosterone_retail": 0.42,
                "wltc_testosterone_noise": 0.19,
            },
            "mfg": {"disagreement": 0.17, "consensus_drift": 0.0011},
            "wltc": {
                "retail": {"testosterone_index": 0.42},
                "institutional": {"testosterone_index": 0.31},
                "noise": {"testosterone_index": 0.19},
            },
            "branches": [
                {"path_id": 1, "probability": 0.61, "selector_score": 0.66, "predicted_prices": [2321.4, 2323.2, 2325.0], "branch_label": "consensus_path"},
                {"path_id": 2, "probability": 0.39, "selector_score": 0.34, "predicted_prices": [2319.4, 2317.8, 2316.0], "branch_label": "minority_path"},
            ],
            "paper_trading": {"open_positions": [], "summary": {"unrealized_pnl": 0.0}},
            "mode": "frequency",
        }
        frame = build_live_v19_candidate_frame(payload)
        self.assertEqual(len(frame), 2)
        self.assertIn("decision_direction", frame.columns)

    def test_build_v19_runtime_state_returns_branch_runtime(self):
        payload = {
            "symbol": "XAUUSD",
            "generated_at": "2026-04-07T00:00:00+00:00",
            "market": {"current_price": 2320.0},
            "simulation": {
                "cpm_score": 0.61,
                "consensus_score": 0.64,
                "overall_confidence": 0.66,
                "analog_confidence": 0.58,
                "cone_width_pips": 10.0,
                "contradiction_type": "mixed",
                "detected_regime": "balanced_range",
                "hurst_overall": 0.56,
                "hurst_positive": 0.58,
                "hurst_negative": 0.53,
                "hurst_asymmetry": 0.05,
                "crowd_bias": 0.12,
                "crowd_extreme": 0.22,
                "mean_probability": 0.57,
            },
            "technical_analysis": {
                "quant_regime_strength": 0.63,
                "quant_transition_risk": 0.28,
                "quant_vol_realism": 0.74,
                "quant_fair_value_z": -0.32,
                "rsi_14": 55.0,
            },
            "current_row": {
                "quant_route_confidence": 0.59,
                "atr_14": 3.4,
                "ema_cross": 0.11,
                "wltc_testosterone_retail": 0.42,
                "wltc_testosterone_noise": 0.19,
            },
            "mfg": {"disagreement": 0.17, "consensus_drift": 0.0011},
            "wltc": {
                "retail": {"testosterone_index": 0.42},
                "institutional": {"testosterone_index": 0.31},
                "noise": {"testosterone_index": 0.19},
            },
            "branches": [
                {"path_id": 1, "probability": 0.61, "selector_score": 0.66, "predicted_prices": [2321.4, 2323.2, 2325.0], "branch_label": "consensus_path"},
                {"path_id": 2, "probability": 0.39, "selector_score": 0.34, "predicted_prices": [2319.4, 2317.8, 2316.0], "branch_label": "minority_path"},
            ],
            "paper_trading": {"open_positions": [], "summary": {"unrealized_pnl": 0.0}},
            "mode": "frequency",
        }
        local_judge = {
            "content": {
                "final_call": "SKIP",
                "confidence": "LOW",
            }
        }
        runtime = build_v19_runtime_state(payload, local_judge=local_judge)
        self.assertTrue(runtime["available"])
        self.assertIn(runtime["runtime_call"], {"BUY", "SELL", "SKIP", "HOLD"})
        self.assertIn("branch_scores", runtime)


if __name__ == "__main__":
    unittest.main()
