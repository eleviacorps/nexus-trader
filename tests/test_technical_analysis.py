from __future__ import annotations

import unittest

import pandas as pd

from src.service.live_data import build_technical_analysis, engineer_price_features


class TechnicalAnalysisTests(unittest.TestCase):
    def test_build_technical_analysis_returns_expected_sections(self):
        candles = pd.DataFrame(
            {
                "open": [100, 101, 100.5, 99.8, 101.2, 100.9, 102.5, 101.7, 103.8, 103.2],
                "high": [101, 101.5, 101.0, 101.4, 101.6, 103.0, 103.2, 104.0, 104.1, 104.4],
                "low": [99.5, 100.2, 99.9, 99.4, 100.6, 100.8, 101.6, 101.3, 102.8, 102.9],
                "close": [100.8, 100.4, 100.1, 101.1, 100.9, 102.7, 101.8, 103.7, 103.0, 104.2],
                "volume": [1] * 10,
            },
            index=pd.date_range("2026-04-01 00:00:00+00:00", periods=10, freq="5min"),
        )
        enriched = engineer_price_features(candles)
        analysis = build_technical_analysis(enriched)
        self.assertIn("order_blocks", analysis)
        self.assertIn("fair_value_gaps", analysis)
        self.assertIn("nearest_support", analysis)
        self.assertIn("nearest_resistance", analysis)
        self.assertIn("session", analysis)


if __name__ == "__main__":
    unittest.main()
