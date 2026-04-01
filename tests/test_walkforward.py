import unittest

import numpy as np

from src.evaluation.walkforward import (
    apply_bucket_calibration,
    capital_backtest_from_unit_pnl,
    confidence_from_probabilities,
    directional_backtest,
    fixed_risk_capital_backtest_from_unit_pnl,
    optimize_backtest_thresholds,
)
from src.training.train_tft import apply_precision_gate, train_precision_gate


class WalkforwardUtilityTests(unittest.TestCase):
    def test_bucket_calibration_maps_to_observed_rates(self):
        probabilities = np.asarray([0.12, 0.18, 0.61, 0.89], dtype=np.float32)
        report = {
            "bins": [
                {"left": 0.0, "right": 0.2, "observed_rate": 0.25},
                {"left": 0.6, "right": 1.0, "observed_rate": 0.8},
            ]
        }
        calibrated = apply_bucket_calibration(probabilities, report)
        self.assertAlmostEqual(float(calibrated[0]), 0.25, places=6)
        self.assertAlmostEqual(float(calibrated[2]), 0.8, places=6)

    def test_directional_backtest_handles_hold_logic(self):
        targets = np.asarray([1, 1, 0, 0, 1], dtype=np.float32)
        probabilities = np.asarray([0.9, 0.55, 0.2, 0.49, 0.51], dtype=np.float32)
        report = directional_backtest(targets, probabilities, decision_threshold=0.6, confidence_floor=0.2)
        self.assertEqual(report["trade_count"], 2)
        self.assertEqual(report["hold_count"], 3)
        self.assertGreaterEqual(report["win_rate"], 0.0)
        self.assertLessEqual(report["win_rate"], 1.0)
        self.assertIn("usd_10", report["capital_backtests"])
        self.assertIn("usd_1000", report["capital_backtests"])
        self.assertIn("usd_10_fixed_risk", report["capital_backtests"])
        self.assertIn("usd_1000_fixed_risk", report["capital_backtests"])

    def test_optimize_backtest_thresholds_returns_finite_configuration(self):
        targets = np.asarray([1, 1, 1, 0, 0, 0, 1, 0], dtype=np.float32)
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1, 0.3, 0.6, 0.4], dtype=np.float32)
        result = optimize_backtest_thresholds(targets, probabilities)
        self.assertGreaterEqual(result["decision_threshold"], 0.5)
        self.assertGreaterEqual(result["confidence_floor"], 0.01)
        self.assertTrue(np.isfinite(result["score"]))
        confidence = confidence_from_probabilities(probabilities)
        self.assertTrue(np.all(confidence >= 0.0))
        self.assertTrue(np.all(confidence <= 1.0))

    def test_capital_backtest_compounds_from_unit_pnl(self):
        report = capital_backtest_from_unit_pnl(np.asarray([1, -1, 0, 1], dtype=np.float32), initial_capital=1000.0, risk_fraction=0.02)
        self.assertEqual(report["trade_count"], 3)
        self.assertGreater(report["final_capital"], 0.0)
        self.assertIn("return_pct", report)

    def test_fixed_risk_capital_backtest_scales_linearly(self):
        report = fixed_risk_capital_backtest_from_unit_pnl(np.asarray([1, -1, 0, 1], dtype=np.float32), initial_capital=1000.0, risk_fraction=0.02)
        self.assertEqual(report["trade_count"], 3)
        self.assertAlmostEqual(report["risk_amount"], 20.0, places=6)
        self.assertGreater(report["final_capital"], 0.0)

    def test_precision_gate_trains_and_scores(self):
        probabilities = np.asarray(
            [
                [0.9, 0.8, 0.7, 0.65],
                [0.85, 0.78, 0.75, 0.7],
                [0.55, 0.52, 0.51, 0.5],
                [0.2, 0.25, 0.3, 0.35],
                [0.15, 0.2, 0.18, 0.22],
                [0.45, 0.48, 0.47, 0.46],
            ],
            dtype=np.float32,
        )
        targets = np.asarray(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        gate = train_precision_gate(probabilities, targets, threshold=0.5, epochs=50, lr=0.1)
        scores = apply_precision_gate(probabilities, gate)
        self.assertEqual(scores.shape, (6,))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))


if __name__ == "__main__":
    unittest.main()
