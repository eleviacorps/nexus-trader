import unittest

import numpy as np

from src.v15.pce import PredictabilityConditionedExecution


class V15PCETests(unittest.TestCase):
    def test_predictable_window_passes_when_thresholds_are_met(self) -> None:
        pce = PredictabilityConditionedExecution(cpm_threshold=0.6, min_agreement=0.5, min_regime_stability_bars=3)
        allowed, reason = pce.is_predictable_window(0.7, 0.6, 4, 55.0)
        self.assertTrue(allowed)
        self.assertEqual(reason, 'predictable_window')

    def test_avoid_window_forces_adjusted_score_to_zero(self) -> None:
        pce = PredictabilityConditionedExecution()
        self.assertEqual(pce.adjusted_cpm_score(0.8, eci_boost=0.15, avoid_window=True), 0.0)

    def test_threshold_tuning_matches_target_participation(self) -> None:
        pce = PredictabilityConditionedExecution()
        scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        threshold = pce.tune_threshold_for_participation(scores, target_rate=0.4)
        self.assertGreaterEqual(threshold, 0.6)
        self.assertLessEqual(threshold, 0.9)


if __name__ == '__main__':
    unittest.main()
