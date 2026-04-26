import unittest

from src.v13.daps import daps_lot_size, maximum_lot_for_leverage


class DAPSTests(unittest.TestCase):
    def test_regime_and_win_rate_affect_lot(self) -> None:
        trend_lot = daps_lot_size(1000, 1200, 0.7, 'trending_up', 0.8)
        range_lot = daps_lot_size(1000, 1200, 0.4, 'ranging', 0.5)
        self.assertGreaterEqual(trend_lot, range_lot)
        self.assertGreaterEqual(trend_lot, 0.05)

    def test_leverage_cap_can_zero_out_under_minimum_lot(self) -> None:
        capped = daps_lot_size(
            1000,
            1000,
            0.7,
            'trending_up',
            0.8,
            max_account_leverage=1.0,
            price_per_ounce=3000.0,
            contract_size_oz=100.0,
        )
        self.assertEqual(capped, 0.0)

    def test_maximum_lot_for_leverage_scales_with_equity(self) -> None:
        lower = maximum_lot_for_leverage(
            current_equity=1000.0,
            max_account_leverage=200.0,
            price_per_ounce=2000.0,
            contract_size_oz=100.0,
        )
        higher = maximum_lot_for_leverage(
            current_equity=1100.0,
            max_account_leverage=200.0,
            price_per_ounce=2000.0,
            contract_size_oz=100.0,
        )
        self.assertIsNotNone(lower)
        self.assertIsNotNone(higher)
        self.assertGreater(higher, lower)


if __name__ == '__main__':
    unittest.main()
