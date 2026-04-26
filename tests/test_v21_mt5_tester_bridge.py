from __future__ import annotations

import unittest

import pandas as pd

from src.v21.mt5_tester_bridge import build_v21_mt5_signal_rows, summarize_v21_mt5_signal_rows


class V21MT5TesterBridgeTests(unittest.TestCase):
    def test_build_signal_rows_exports_buy_and_sell_actions(self) -> None:
        index = pd.date_range("2024-01-01 00:00:00+00:00", periods=8, freq="15min", tz="UTC")
        raw = pd.DataFrame(
            {
                "open": [2000.0, 2001.0, 2002.0, 2003.0, 2004.0, 2005.0, 2006.0, 2007.0],
                "high": [2001.0, 2002.0, 2003.0, 2004.0, 2005.0, 2006.0, 2007.0, 2008.0],
                "low": [1999.0, 2000.0, 2001.0, 2002.0, 2003.0, 2004.0, 2005.0, 2006.0],
                "close": [2000.5, 2001.5, 2002.5, 2003.5, 2004.5, 2003.5, 2002.5, 2001.5],
                "volume": [10, 10, 10, 10, 10, 10, 10, 10],
            },
            index=index,
        )

        def fake_runtime(payload: dict, mode: str = "precision") -> dict:
            current_price = float(payload["market"]["current_price"])
            direction = "BUY" if current_price < 2004.6 else "SELL"
            return {
                "available": True,
                "decision_direction": direction,
                "raw_stance": direction,
                "confidence_tier": "high",
                "sqt_label": "GOOD",
                "cabr_score": 0.71,
                "cpm_score": 0.66,
                "lepl_features": {
                    "suggested_lot": 0.12,
                    "conformal_confidence": 0.74,
                    "kelly_fraction": 0.03,
                },
                "v21_dangerous_branch_count": 1,
                "selected_branch_label": "test_branch",
                "execution_reason": f"runtime_{mode}",
            }

        def fake_judge(payload: dict, runtime: dict) -> dict:
            current_price = float(payload["market"]["current_price"])
            if runtime["decision_direction"] == "BUY":
                stop_loss = current_price - 1.2
                take_profit = current_price + 2.4
            else:
                stop_loss = current_price + 1.2
                take_profit = current_price - 2.4
            return {
                "content": {
                    "final_summary": "tester bridge summary",
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
            }

        rows = build_v21_mt5_signal_rows(
            raw,
            start="2024-01-01 00:00:00+00:00",
            end="2024-01-01 02:00:00+00:00",
            lookback_bars=4,
            runtime_builder=fake_runtime,
            judge_builder=fake_judge,
        )

        self.assertGreaterEqual(len(rows), 2)
        self.assertIn("execution_time_utc", rows.columns)
        self.assertIn("lot", rows.columns)
        self.assertIn("stop_loss", rows.columns)
        self.assertIn("take_profit", rows.columns)
        self.assertEqual(float(rows.iloc[0]["lot"]), 0.12)
        self.assertTrue(set(rows["action"].unique().tolist()).issubset({"BUY", "SELL"}))

    def test_summary_counts_actions(self) -> None:
        rows = pd.DataFrame(
            [
                {"action": "BUY", "lot": 0.10, "cabr_score": 0.70, "cpm_score": 0.60, "execution_time_utc": "2024-01-01T00:15:00+00:00"},
                {"action": "SELL", "lot": 0.20, "cabr_score": 0.80, "cpm_score": 0.50, "execution_time_utc": "2024-01-01T00:30:00+00:00"},
            ]
        )
        summary = summarize_v21_mt5_signal_rows(rows)
        self.assertEqual(summary["signals"], 2)
        self.assertEqual(summary["buy_signals"], 1)
        self.assertEqual(summary["sell_signals"], 1)
        self.assertAlmostEqual(summary["avg_lot"], 0.15, places=6)


if __name__ == "__main__":
    unittest.main()
