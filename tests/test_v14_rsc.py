from __future__ import annotations

import unittest

from src.v14.rsc import RegimeStratifiedCalibrator


class TestV14RSC(unittest.TestCase):
    def test_unknown_regime_falls_back(self) -> None:
        calibrator = RegimeStratifiedCalibrator()
        value = calibrator.calibrate(0.6, "unknown")
        self.assertGreaterEqual(value, 0.10)
        self.assertLessEqual(value, 0.90)

    def test_record_and_summary(self) -> None:
        calibrator = RegimeStratifiedCalibrator()
        for _ in range(5):
            calibrator.record_outcome(0.6, "ranging", True)
        summary = calibrator.summary()
        self.assertIn("counts_per_regime", summary)
        self.assertIn("ranging", summary["counts_per_regime"])


if __name__ == "__main__":
    unittest.main()
