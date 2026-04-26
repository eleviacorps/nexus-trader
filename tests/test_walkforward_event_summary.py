import unittest

from src.evaluation.walkforward import summarize_event_backtests


class EventBacktestSummaryTests(unittest.TestCase):
    def test_summarize_event_backtests_aggregates_fold_reports(self):
        reports = [
            {
                "trade_count": 10,
                "hold_count": 20,
                "participation_rate": 0.25,
                "win_rate": 0.6,
                "loss_rate": 0.4,
                "long_win_rate": 0.6,
                "short_win_rate": 0.0,
                "avg_unit_pnl": 0.1,
                "gross_avg_unit_pnl": 0.12,
                "cumulative_unit_pnl": 1.0,
                "gross_cumulative_unit_pnl": 1.2,
                "max_drawdown_units": 0.4,
                "decision_threshold": 0.6,
                "confidence_floor": 0.1,
                "gate_threshold": 0.2,
                "hold_threshold": 0.55,
                "fee_model": "FixedBpsFeeModel",
                "slippage_model": "VolatilityScaledSlippageModel",
                "capital_backtests": {
                    "usd_10": {"final_capital": 11.0, "max_drawdown_pct": 10.0},
                    "usd_1000": {"final_capital": 1100.0, "max_drawdown_pct": 10.0},
                    "usd_10_fixed_risk": {"final_capital": 14.0, "max_drawdown_pct": 8.0},
                    "usd_1000_fixed_risk": {"final_capital": 1400.0, "max_drawdown_pct": 8.0},
                },
            },
            {
                "trade_count": 30,
                "hold_count": 10,
                "participation_rate": 0.5,
                "win_rate": 0.7,
                "loss_rate": 0.3,
                "long_win_rate": 0.7,
                "short_win_rate": 0.0,
                "avg_unit_pnl": 0.2,
                "gross_avg_unit_pnl": 0.23,
                "cumulative_unit_pnl": 3.0,
                "gross_cumulative_unit_pnl": 3.4,
                "max_drawdown_units": 0.9,
                "decision_threshold": 0.6,
                "confidence_floor": 0.1,
                "gate_threshold": 0.2,
                "hold_threshold": 0.55,
                "fee_model": "FixedBpsFeeModel",
                "slippage_model": "VolatilityScaledSlippageModel",
                "capital_backtests": {
                    "usd_10": {"final_capital": 12.0, "max_drawdown_pct": 12.0},
                    "usd_1000": {"final_capital": 1200.0, "max_drawdown_pct": 12.0},
                    "usd_10_fixed_risk": {"final_capital": 15.0, "max_drawdown_pct": 9.0},
                    "usd_1000_fixed_risk": {"final_capital": 1500.0, "max_drawdown_pct": 9.0},
                },
            },
        ]

        summary = summarize_event_backtests(reports)
        self.assertEqual(summary["trade_count"], 40)
        self.assertEqual(summary["hold_count"], 30)
        self.assertGreater(summary["win_rate"], 0.65)
        self.assertIn("capital_backtests", summary)


if __name__ == "__main__":
    unittest.main()
