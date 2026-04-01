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
from src.pipeline.fusion import GATE_CONTEXT_COLUMNS
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
        self.assertFalse(report["overflowed"])

    def test_capital_backtest_handles_extreme_growth_without_nan(self):
        pnl = np.ones(10000, dtype=np.float32)
        report = capital_backtest_from_unit_pnl(pnl, initial_capital=1000.0, risk_fraction=0.10)
        self.assertTrue(report["overflowed"])
        self.assertIsNone(report["final_capital"])
        self.assertIsNone(report["net_profit"])
        self.assertIsNone(report["return_pct"])
        self.assertGreater(report["log10_final_capital"], 0.0)

    def test_fixed_risk_capital_backtest_scales_linearly(self):
        report = fixed_risk_capital_backtest_from_unit_pnl(np.asarray([1, -1, 0, 1], dtype=np.float32), initial_capital=1000.0, risk_fraction=0.02)
        self.assertEqual(report["trade_count"], 3)
        self.assertAlmostEqual(report["risk_amount"], 20.0, places=6)
        self.assertGreater(report["final_capital"], 0.0)

    def test_precision_gate_trains_and_scores(self):
        direction_probabilities = np.asarray(
            [
                [0.90, 0.84, 0.79, 0.76],
                [0.86, 0.80, 0.77, 0.73],
                [0.62, 0.59, 0.57, 0.54],
                [0.18, 0.23, 0.28, 0.31],
                [0.14, 0.18, 0.22, 0.26],
                [0.48, 0.50, 0.49, 0.51],
            ],
            dtype=np.float32,
        )
        hold_probabilities = np.asarray(
            [
                [0.10, 0.12, 0.14, 0.16],
                [0.12, 0.14, 0.15, 0.18],
                [0.22, 0.24, 0.28, 0.32],
                [0.18, 0.20, 0.22, 0.24],
                [0.20, 0.22, 0.24, 0.26],
                [0.62, 0.60, 0.61, 0.63],
            ],
            dtype=np.float32,
        )
        confidence_probabilities = np.asarray(
            [
                [0.82, 0.80, 0.78, 0.76],
                [0.79, 0.77, 0.75, 0.73],
                [0.58, 0.56, 0.54, 0.52],
                [0.68, 0.66, 0.64, 0.62],
                [0.71, 0.69, 0.67, 0.65],
                [0.28, 0.26, 0.24, 0.22],
            ],
            dtype=np.float32,
        )
        probabilities = np.asarray(
            [direction_probabilities, hold_probabilities, confidence_probabilities],
            dtype=np.float32,
        ).reshape(3, 6, 4).transpose(1, 0, 2).reshape(6, 12)
        direction_targets = np.asarray(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        hold_targets = np.asarray(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        confidence_targets = np.asarray(
            [
                [0.90, 0.88, 0.86, 0.84],
                [0.88, 0.86, 0.84, 0.82],
                [0.60, 0.58, 0.56, 0.54],
                [0.74, 0.72, 0.70, 0.68],
                [0.76, 0.74, 0.72, 0.70],
                [0.22, 0.20, 0.18, 0.16],
            ],
            dtype=np.float32,
        )
        targets = np.asarray(
            [direction_targets, hold_targets, confidence_targets],
            dtype=np.float32,
        ).reshape(3, 6, 4).transpose(1, 0, 2).reshape(6, 12)
        safe_context = np.asarray([0.30, 0.25, 0.40, 1.0, 0.60, 0.55, 0.20, 0.20, 0.15, 0.80, 0.20, 0.15, 0.50, 0.70, 0.25, 0.30, 0.10], dtype=np.float32)
        risky_context = np.asarray([0.60, 0.55, 0.35, 0.0, 0.10, 0.05, 0.92, 0.95, 0.90, 0.05, 1.40, 0.98, 0.10, 0.05, -0.02, 0.05, 0.98], dtype=np.float32)
        context_features = np.vstack([safe_context, safe_context, safe_context, risky_context, risky_context, risky_context]).astype(np.float32)
        self.assertEqual(context_features.shape[1], len(GATE_CONTEXT_COLUMNS))
        gate = train_precision_gate(probabilities, targets, context_features=context_features, threshold=0.5, epochs=50, lr=0.1)
        scores = apply_precision_gate(probabilities, gate, context_features=context_features)
        self.assertEqual(scores.shape, (6,))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))
        self.assertEqual(gate["context_feature_names"], list(GATE_CONTEXT_COLUMNS))


if __name__ == "__main__":
    unittest.main()


