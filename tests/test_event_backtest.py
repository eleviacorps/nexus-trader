import unittest

import numpy as np

from src.backtest.event_engine import event_driven_directional_backtest
from src.backtest.fees import FixedBpsFeeModel
from src.backtest.slippage import FixedBpsSlippageModel


class EventDrivenBacktestTests(unittest.TestCase):
    def test_event_driven_backtest_produces_orders_fills_and_trades(self):
        targets = np.asarray([1, 1, 0], dtype=np.float32)
        probabilities = np.asarray([0.9, 0.85, 0.2], dtype=np.float32)
        open_prices = np.asarray([100.0, 101.0, 102.0, 101.0], dtype=np.float32)
        high_prices = np.asarray([101.0, 102.0, 103.0, 102.0], dtype=np.float32)
        low_prices = np.asarray([99.0, 100.0, 101.0, 100.0], dtype=np.float32)
        close_prices = np.asarray([100.5, 101.5, 101.2, 100.6], dtype=np.float32)
        report = event_driven_directional_backtest(
            targets,
            probabilities,
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            close_prices=close_prices,
            decision_threshold=0.6,
            confidence_floor=0.2,
            hold_bars=1,
        )
        self.assertEqual(report["execution_mode"], "event_driven")
        self.assertEqual(len(report["orders"]), 3)
        self.assertEqual(len(report["fills"]), 6)
        self.assertEqual(len(report["trades"]), 3)

    def test_event_driven_backtest_applies_cost_models(self):
        targets = np.asarray([1, 1], dtype=np.float32)
        probabilities = np.asarray([0.9, 0.9], dtype=np.float32)
        open_prices = np.asarray([100.0, 100.0, 101.0], dtype=np.float32)
        high_prices = np.asarray([101.0, 101.0, 102.0], dtype=np.float32)
        low_prices = np.asarray([99.0, 99.0, 100.0], dtype=np.float32)
        close_prices = np.asarray([100.0, 101.0, 102.0], dtype=np.float32)
        plain = event_driven_directional_backtest(
            targets,
            probabilities,
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            close_prices=close_prices,
            decision_threshold=0.6,
            confidence_floor=0.2,
            hold_bars=1,
        )
        costed = event_driven_directional_backtest(
            targets,
            probabilities,
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            close_prices=close_prices,
            decision_threshold=0.6,
            confidence_floor=0.2,
            hold_bars=1,
            fee_model=FixedBpsFeeModel(entry_bps=5, exit_bps=5),
            slippage_model=FixedBpsSlippageModel(bps=10),
        )
        self.assertLess(costed["avg_unit_pnl"], plain["avg_unit_pnl"])


if __name__ == "__main__":
    unittest.main()
