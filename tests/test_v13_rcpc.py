import unittest

from src.v13.rcpc import RegimeConditionalPriorCalibrator, TRANSITION_THRESHOLD


class RCPCTests(unittest.TestCase):
    def test_prior_then_learned_transition(self) -> None:
        calibrator = RegimeConditionalPriorCalibrator()
        prior_value = calibrator.calibrate(0.7, 'trending_up')
        self.assertGreater(prior_value, 0.5)
        for index in range(TRANSITION_THRESHOLD):
            calibrator.record_outcome(0.6 + 0.005 * (index % 5), index % 2 == 0)
        self.assertTrue(calibrator.uses_learned_calibration)
        self.assertIsNotNone(calibrator.calibration_error())


if __name__ == '__main__':
    unittest.main()
