from __future__ import annotations

import unittest

import numpy as np

from src.v20.conformal_cone import ConformalCone


class V20ConformalTests(unittest.TestCase):
    def test_cone_calibration_and_prediction_are_finite(self) -> None:
        predicted = [[100.0, 101.0, 102.0] for _ in range(24)]
        realized = [[100.0, 100.8, 101.9] for _ in range(24)]
        regimes = [idx % 6 for idx in range(24)]
        cone = ConformalCone(alpha=0.15)
        cone.calibrate(predicted, realized, regimes)
        upper, lower, confidence = cone.predict(np.asarray([100.0, 101.0, 102.0]), current_regime=1)
        self.assertEqual(len(upper), 3)
        self.assertEqual(len(lower), 3)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertTrue(np.all(upper >= lower))


if __name__ == "__main__":
    unittest.main()
