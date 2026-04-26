import unittest

from src.v13.mbeg import minority_guard


class MBEGTests(unittest.TestCase):
    def test_veto_and_reduce(self) -> None:
        allow, multiplier = minority_guard('BUY', 'SELL', 0.9, 0.5)
        self.assertFalse(allow)
        self.assertEqual(multiplier, 0.0)
        allow2, multiplier2 = minority_guard('BUY', 'SELL', 0.4, 0.4)
        self.assertTrue(allow2)
        self.assertLess(multiplier2, 1.0)


if __name__ == '__main__':
    unittest.main()
