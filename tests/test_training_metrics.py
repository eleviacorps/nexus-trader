import unittest

import numpy as np

from src.training.train_tft import build_calibration_report, collect_binary_metrics, find_optimal_threshold


class TrainingMetricTests(unittest.TestCase):
    def test_collect_binary_metrics_includes_threshold(self):
        targets = np.array([0, 1, 1, 0], dtype=np.float32)
        probabilities = np.array([0.2, 0.9, 0.6, 0.3], dtype=np.float32)
        metrics = collect_binary_metrics(targets, probabilities, threshold=0.65)
        self.assertAlmostEqual(metrics["threshold"], 0.65, places=6)

    def test_find_optimal_threshold_returns_valid_range(self):
        targets = np.array([0, 0, 1, 1], dtype=np.float32)
        probabilities = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
        result = find_optimal_threshold(targets, probabilities)
        self.assertGreaterEqual(result["threshold"], 0.05)
        self.assertLessEqual(result["threshold"], 0.95)

    def test_build_calibration_report_produces_finite_values(self):
        targets = np.array([0, 1, 1, 0, 1], dtype=np.float32)
        probabilities = np.array([0.1, 0.8, 0.7, 0.4, 0.9], dtype=np.float32)
        report = build_calibration_report(targets, probabilities, bins=5)
        self.assertGreaterEqual(report["ece"], 0.0)
        self.assertGreaterEqual(report["max_calibration_gap"], 0.0)
        self.assertTrue(report["bins"])


if __name__ == "__main__":
    unittest.main()
