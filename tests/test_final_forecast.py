from __future__ import annotations

import unittest

from src.service.live_data import _build_final_forecast


class FinalForecastTests(unittest.TestCase):
    def test_build_final_forecast_returns_horizon_table(self):
        payload = {
            "market": {"current_price": 3000.0},
            "current_row": {"atr_14": 12.0},
            "cone": [
                {"horizon": 1, "timestamp": "2026-04-01T00:05:00+00:00", "center_price": 3001.0},
                {"horizon": 2, "timestamp": "2026-04-01T00:10:00+00:00", "center_price": 3002.0},
                {"horizon": 3, "timestamp": "2026-04-01T00:15:00+00:00", "center_price": 3003.0},
                {"horizon": 6, "timestamp": "2026-04-01T00:30:00+00:00", "center_price": 3006.0},
            ],
        }
        bot_swarm = {
            "aggregate": {
                "horizon_predictions": [
                    {"minutes": 5, "target_price": 3001.2},
                    {"minutes": 10, "target_price": 3002.4},
                    {"minutes": 15, "target_price": 3003.8},
                    {"minutes": 30, "target_price": 3007.5},
                ]
            }
        }
        llm_content = {"institutional_bias": 0.2, "whale_bias": 0.1, "retail_bias": 0.0}
        model_prediction = {"bullish_probability": 0.6}
        result = _build_final_forecast(payload, bot_swarm, llm_content, model_prediction)
        self.assertEqual([row["minutes"] for row in result["horizon_table"]], [5, 10, 15, 30])
        self.assertEqual(len(result["points"]), 4)


if __name__ == "__main__":
    unittest.main()
