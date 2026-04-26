import unittest

from src.v16.csl import build_v16_simulation_result
from src.v16.sqt import SimulationQualityTracker


class V16CSLTests(unittest.TestCase):
    def test_builds_15m_simulation_payload(self) -> None:
        tracker = SimulationQualityTracker()
        tracker.record("BUY", "BUY", "high")
        payload = {
            "symbol": "XAUUSD",
            "generated_at": "2026-04-06T00:00:00Z",
            "market": {
                "current_price": 2300.0,
                "candles": [{"timestamp": "2026-04-06T00:00:00Z"}],
            },
            "simulation": {
                "mean_probability": 0.62,
                "overall_confidence": 0.61,
                "selector_top_score": 0.71,
            },
            "current_row": {
                "atr_14": 3.0,
                "return_1": 0.01,
                "rsi_14": 28.0,
                "ema_cross": 1.0,
                "macd_hist": 0.04,
                "bb_pct": 0.05,
                "volume_ratio": 1.8,
            },
            "cone": [
                {"horizon": 1, "timestamp": "2026-04-06T00:05:00Z", "center_price": 2300.5, "lower_price": 2299.8, "upper_price": 2301.2},
                {"horizon": 2, "timestamp": "2026-04-06T00:10:00Z", "center_price": 2301.1, "lower_price": 2300.2, "upper_price": 2302.0},
                {"horizon": 3, "timestamp": "2026-04-06T00:15:00Z", "center_price": 2301.8, "lower_price": 2300.7, "upper_price": 2302.8},
            ],
            "branches": [
                {"path_id": 1, "selector_score": 0.75, "probability": 0.72, "predicted_prices": [2300.5, 2301.1, 2301.8], "dominant_persona": "institutional"},
                {"path_id": 2, "selector_score": 0.43, "probability": 0.28, "predicted_prices": [2299.7, 2299.2, 2298.8], "dominant_persona": "retail"},
            ],
            "highlighted_branches": {"minority_branch_id": 2},
            "paper_trading": {"balance": 1000.0, "equity": 1000.0},
            "eci": {"cone_width_modifier": 0.0},
        }
        model_prediction = {"bullish_probability": 0.66, "horizon_probabilities": {"15m": 0.67}}
        result = build_v16_simulation_result(payload, model_prediction, mode="frequency", sqt=tracker, eci_context=payload["eci"])
        simulation = result["simulation"]
        self.assertEqual(simulation["direction"], "BUY")
        self.assertEqual(simulation["primary_horizon_minutes"], 15)
        self.assertEqual(len(simulation["consensus_path"]), 4)
        self.assertGreater(simulation["cabr_score"], 0.5)
        self.assertIn("horizon_table", result["final_forecast"])


if __name__ == "__main__":
    unittest.main()
