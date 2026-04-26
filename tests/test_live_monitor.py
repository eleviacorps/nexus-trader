from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.service.live_data import SIMULATION_SCHEMA_VERSION, build_history_entry, build_simulation_comparison


class LiveMonitorTests(unittest.TestCase):
    def test_build_history_entry_captures_anchor_and_center_path(self):
        payload = {
            "symbol": "XAUUSD",
            "generated_at": "2026-03-31T07:00:00+00:00",
            "market": {
                "current_price": 3050.0,
                "candles": [
                    {"timestamp": "2026-03-31T06:50:00+00:00", "open": 3048.0, "high": 3050.0, "low": 3047.0, "close": 3049.0, "volume": 1.0},
                    {"timestamp": "2026-03-31T06:55:00+00:00", "open": 3049.0, "high": 3051.0, "low": 3048.0, "close": 3050.0, "volume": 1.0},
                ],
            },
            "simulation": {
                "scenario_bias": "bullish",
                "overall_confidence": 0.73,
                "consensus_score": 0.81,
                "uncertainty_width": 0.12,
                "dominant_driver": "crowd_buying",
            },
            "cone": [
                {"timestamp": "2026-03-31T07:00:00+00:00", "center_price": 3051.0, "lower_price": 3049.0, "upper_price": 3053.0},
                {"timestamp": "2026-03-31T07:05:00+00:00", "center_price": 3052.0, "lower_price": 3048.0, "upper_price": 3054.0},
            ],
        }

        entry = build_history_entry(payload, {"signal": "bullish", "bullish_probability": 0.61})
        self.assertEqual(entry["anchor_timestamp"], "2026-03-31T06:55:00+00:00")
        self.assertEqual(len(entry["center_path"]), 2)
        self.assertEqual(entry["scenario_bias"], "bullish")

    def test_build_simulation_comparison_returns_hit_rate_for_recorded_entry(self):
        root = Path("tests/.tmp/live_monitor")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            history_path = root / "live_sim_history.json"
            history_entry = {
                "symbol": "XAUUSD",
                "generated_at": "2026-03-31T07:00:00+00:00",
                "simulation_version": SIMULATION_SCHEMA_VERSION,
                "anchor_timestamp": "2026-03-31T06:55:00+00:00",
                "anchor_price": 3050.0,
                "market_source": "unknown",
                "scenario_bias": "bullish",
                "overall_confidence": 0.75,
                "consensus_score": 0.80,
                "uncertainty_width": 0.10,
                "dominant_driver": "crowd_buying",
                "center_path": [
                    {"timestamp": "2026-03-31T07:00:00+00:00", "price": 3051.0},
                    {"timestamp": "2026-03-31T07:05:00+00:00", "price": 3052.0},
                ],
                "cone": [
                    {"timestamp": "2026-03-31T07:00:00+00:00", "center_price": 3051.0, "lower_price": 3049.0, "upper_price": 3053.0},
                    {"timestamp": "2026-03-31T07:05:00+00:00", "center_price": 3052.0, "lower_price": 3050.0, "upper_price": 3054.0},
                ],
                "model_prediction": {"signal": "bullish"},
            }
            history_path.write_text(json.dumps([history_entry]), encoding="utf-8")

            candles = pd.DataFrame(
                {
                    "open": [3048.0, 3049.0, 3051.0, 3052.0],
                    "high": [3050.0, 3051.0, 3052.5, 3054.0],
                    "low": [3047.0, 3048.0, 3050.5, 3051.5],
                    "close": [3049.0, 3050.0, 3051.5, 3053.0],
                    "volume": [1.0, 1.0, 1.0, 1.0],
                },
                index=pd.to_datetime(
                    [
                        "2026-03-31T06:50:00+00:00",
                        "2026-03-31T06:55:00+00:00",
                        "2026-03-31T07:00:00+00:00",
                        "2026-03-31T07:05:00+00:00",
                    ],
                    utc=True,
                ),
            )

            with patch("src.service.live_data.LIVE_SIMULATION_HISTORY_PATH", history_path):
                comparison = build_simulation_comparison("XAUUSD", candles)

            self.assertIn("active_prediction", comparison)
            self.assertEqual(comparison["active_prediction"]["matched_points"], 2)
            self.assertGreaterEqual(comparison["active_prediction"]["hit_rate"], 0.5)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
