import unittest

import numpy as np

from src.backtest.engine import directional_backtest
from src.backtest.fees import FixedBpsFeeModel
from src.backtest.slippage import FixedBpsSlippageModel, VolatilityScaledSlippageModel


class BacktestEngineTests(unittest.TestCase):
    def test_directional_backtest_supports_cost_models(self):
        targets = np.asarray([1, 1, 0, 0], dtype=np.float32)
        probabilities = np.asarray([0.9, 0.85, 0.1, 0.2], dtype=np.float32)
        base = directional_backtest(targets, probabilities, decision_threshold=0.6, confidence_floor=0.2)
        costed = directional_backtest(
            targets,
            probabilities,
            decision_threshold=0.6,
            confidence_floor=0.2,
            fee_model=FixedBpsFeeModel(entry_bps=5, exit_bps=5),
            slippage_model=FixedBpsSlippageModel(bps=10),
        )
        self.assertEqual(base["trade_count"], costed["trade_count"])
        self.assertLess(costed["avg_unit_pnl"], base["avg_unit_pnl"])
        self.assertLess(costed["cumulative_unit_pnl"], base["cumulative_unit_pnl"])

    def test_directional_backtest_can_return_trade_records(self):
        targets = np.asarray([1, 0, 1], dtype=np.float32)
        probabilities = np.asarray([0.9, 0.2, 0.8], dtype=np.float32)
        report = directional_backtest(targets, probabilities, decision_threshold=0.6, confidence_floor=0.2, return_trades=True)
        self.assertIn("trades", report)
        self.assertEqual(len(report["trades"]), 3)
        self.assertIn("fee_penalty", report["trades"][0])

    def test_volatility_scaled_slippage_penalizes_high_volatility(self):
        targets = np.asarray([1, 1], dtype=np.float32)
        probabilities = np.asarray([0.9, 0.9], dtype=np.float32)
        report = directional_backtest(
            targets,
            probabilities,
            decision_threshold=0.6,
            confidence_floor=0.2,
            slippage_model=VolatilityScaledSlippageModel(base_bps=2, volatility_weight=10),
            volatility_scale=np.asarray([0.1, 2.0], dtype=np.float32),
            return_trades=True,
        )
        trades = report["trades"]
        self.assertLess(trades[1]["net_unit_pnl"], trades[0]["net_unit_pnl"])


if __name__ == "__main__":
    unittest.main()
