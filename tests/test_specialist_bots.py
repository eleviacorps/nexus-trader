from __future__ import annotations

import unittest

from src.service.specialist_bots import run_specialist_bots


class SpecialistBotsTests(unittest.TestCase):
    def test_run_specialist_bots_returns_ten_bots_and_horizons(self):
        payload = run_specialist_bots(
            symbol="XAUUSD",
            current_row={
                "close": 3000.0,
                "atr_14": 12.0,
                "ema_cross": 0.4,
                "rsi_14": 58.0,
                "rsi_7": 61.0,
                "macd_hist": 0.8,
                "bb_pct": 0.72,
                "macro_bias": 0.25,
                "macro_shock": 0.32,
                "news_bias": 0.18,
                "crowd_bias": -0.15,
                "crowd_extreme": 0.45,
            },
            technical_analysis={
                "equilibrium": 2998.0,
                "nearest_support": {"price": 2991.0},
                "nearest_resistance": {"price": 3008.0},
                "order_blocks": [{"type": "bullish_order_block", "high": 3002.0, "low": 2997.0, "strength": 0.7}],
                "fair_value_gaps": [{"type": "bullish_fvg", "high": 3004.0, "low": 3001.0, "size": 3.0}],
            },
            feeds={
                "news": [{"sentiment": 0.2}],
                "public_discussions": [{"sentiment": -0.1}],
            },
            llm_content={
                "institutional_bias": 0.2,
                "whale_bias": 0.1,
                "retail_bias": -0.05,
            },
        )
        self.assertEqual(len(payload["bots"]), 10)
        self.assertEqual([item["minutes"] for item in payload["aggregate"]["horizon_predictions"]], [5, 10, 15, 30])
        self.assertIn("graph", payload)
        self.assertTrue(payload["graph"]["nodes"])


if __name__ == "__main__":
    unittest.main()
