from __future__ import annotations

import unittest

import pandas as pd

from src.v20.sjd_v20 import latest_sjd_decision, rule_based_sjd_labels


class V20SJDTests(unittest.TestCase):
    def test_rule_based_labels_emit_expected_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "future_return_15m": [0.01, -0.01, 0.0],
                "hmm_state": [1, 5, 0],
                "hurst_overall": [0.6, 0.45, 0.52],
                "mfg_disagreement": [0.2, 0.8, 0.4],
                "atr_pct": [0.001, 0.0012, 0.0011],
            }
        )
        labels = rule_based_sjd_labels(frame)
        self.assertEqual(list(labels.columns), ["stance", "confidence", "tp_offset", "sl_offset", "kelly"])
        self.assertEqual(labels.iloc[0]["stance"], "BUY")
        self.assertEqual(labels.iloc[1]["stance"], "HOLD")

    def test_latest_decision_returns_trade_or_skip(self) -> None:
        decision = latest_sjd_decision(
            {
                "direction_signal": 0.2,
                "hmm_state": 1,
                "hurst_overall": 0.6,
                "mfg_disagreement": 0.2,
                "atr_pct": 0.0015,
            }
        )
        self.assertEqual(decision.final_call, "BUY")
        self.assertGreater(decision.tp_offset, 0.0)


if __name__ == "__main__":
    unittest.main()
