import unittest

from src.v13.daps import daps_lot_size


class DAPSTests(unittest.TestCase):
    def test_regime_and_win_rate_affect_lot(self) -> None:
        trend_lot = daps_lot_size(1000, 1200, 0.7, 'trending_up', 0.8)
        range_lot = daps_lot_size(1000, 1200, 0.4, 'ranging', 0.5)
        self.assertGreaterEqual(trend_lot, range_lot)
        self.assertGreaterEqual(trend_lot, 0.01)


if __name__ == '__main__':
    unittest.main()
