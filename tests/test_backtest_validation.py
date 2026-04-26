import unittest

import numpy as np

from src.backtest.validation import (
    analyze_feature_target_correlations,
    analyze_recursive_window_consistency,
    analyze_timestamp_monotonicity,
)


class BacktestValidationTests(unittest.TestCase):
    def test_timestamp_monotonicity_detects_duplicates(self):
        timestamps = np.asarray(["2026-01-01T00:00", "2026-01-01T00:01", "2026-01-01T00:01"], dtype="datetime64[m]")
        report = analyze_timestamp_monotonicity(timestamps)
        self.assertFalse(report["strictly_increasing"])
        self.assertEqual(report["duplicate_count"], 1)

    def test_recursive_window_consistency_reports_zero_mismatch_for_shifted_windows(self):
        tensor = np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )
        report = analyze_recursive_window_consistency(tensor)
        self.assertTrue(report["consistent"])
        self.assertEqual(report["max_abs_mismatch"], 0.0)

    def test_feature_target_correlations_flags_extreme_leakage_like_signal(self):
        features = np.asarray(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        )
        targets = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        report = analyze_feature_target_correlations(features, targets, feature_names=["a", "b"], suspicious_threshold=0.5, critical_threshold=0.9)
        self.assertGreaterEqual(len(report["critical_correlations"]), 1)


if __name__ == "__main__":
    unittest.main()
