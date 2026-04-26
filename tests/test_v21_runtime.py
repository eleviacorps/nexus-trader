from __future__ import annotations

import unittest

from src.v21.runtime_v21 import V21Runtime


class V21RuntimeTests(unittest.TestCase):
    def test_research_mode_trades_any_non_hold(self) -> None:
        runtime = V21Runtime(mode="research")
        should_trade, failed = runtime.should_trade(
            sjd_output={"stance": "BUY", "disagree_prob": 0.99},
            conformal_confidence=0.1,
            dangerous_branch_count=9,
            meta_label_prob=0.1,
        )
        self.assertTrue(should_trade)
        self.assertEqual(failed, [])

    def test_production_mode_blocks_failed_gates(self) -> None:
        runtime = V21Runtime(mode="production")
        should_trade, failed = runtime.should_trade(
            sjd_output={"stance": "BUY", "disagree_prob": 0.7},
            conformal_confidence=0.54,
            dangerous_branch_count=3,
            meta_label_prob=0.39,
        )
        self.assertFalse(should_trade)
        self.assertEqual(set(failed), {"conformal", "dangerous", "disagreement", "meta_label"})

    def test_lot_size_is_capped(self) -> None:
        runtime = V21Runtime(mode="production")
        lot = runtime.get_size(kelly_fraction=0.25, account_balance=10000.0, price=3000.0)
        self.assertLessEqual(lot, 0.2)
        self.assertGreaterEqual(lot, 0.0)


if __name__ == "__main__":
    unittest.main()
