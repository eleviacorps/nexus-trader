import unittest

import numpy as np
import pandas as pd

from src.pipeline.fusion import (
    build_gate_context_matrix,
    build_fused_feature_matrix,
    build_sequence_tensor,
    build_trade_target_artifacts,
    merge_market_dynamics_features,
    normalize_binary_targets,
)


class FusionPipelineTests(unittest.TestCase):
    def test_build_fused_feature_matrix_shape(self):
        price = np.zeros((3, 36), dtype=np.float32)
        news = np.ones((3, 32), dtype=np.float32)
        crowd = np.full((3, 32), 2.0, dtype=np.float32)
        fused = build_fused_feature_matrix(price, news, crowd)
        self.assertEqual(fused.shape, (3, 100))

    def test_normalize_binary_targets(self):
        targets = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        normalized = normalize_binary_targets(targets)
        np.testing.assert_array_equal(normalized, np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

    def test_build_sequence_tensor_shape(self):
        features = np.arange(5 * 100, dtype=np.float32).reshape(5, 100)
        targets = np.array([0, 1, 0, 1, 1], dtype=np.float32)
        tensor, seq_targets = build_sequence_tensor(features, targets, sequence_len=3)
        self.assertEqual(tensor.shape, (3, 3, 100))
        np.testing.assert_array_equal(seq_targets, np.array([0, 1, 1], dtype=np.float32))

    def test_build_trade_target_artifacts_outputs_multi_horizon_summary(self):
        frame = pd.DataFrame(
            {
                "close": np.array([100.0, 100.4, 100.8, 100.3, 100.9, 101.5, 101.1, 101.8], dtype=np.float32),
                "atr_pct": np.full(8, 0.002, dtype=np.float32),
            }
        )
        artifacts = build_trade_target_artifacts(frame, horizons=(1, 2, 3), primary_horizon=1, min_abs_return=1e-4)
        self.assertEqual(artifacts.primary_targets.shape, (8,))
        self.assertEqual(artifacts.primary_hold_mask.shape, (8,))
        self.assertEqual(artifacts.sample_weights.shape, (8,))
        self.assertIn(2, artifacts.horizon_returns)
        self.assertIn("1m", artifacts.summary["horizon_summary"])
        self.assertGreaterEqual(artifacts.summary["target_hold_rate"], 0.0)
        self.assertLessEqual(artifacts.summary["target_hold_rate"], 1.0)
        self.assertTrue(np.all(artifacts.sample_weights >= 0.35))

    def test_hold_rows_are_downweighted(self):
        frame = pd.DataFrame(
            {
                "close": np.array([100.0, 100.01, 100.02, 100.03, 100.04, 100.05], dtype=np.float32),
                "atr_pct": np.full(6, 0.01, dtype=np.float32),
            }
        )
        artifacts = build_trade_target_artifacts(frame, horizons=(1, 2), primary_horizon=1, min_abs_return=1e-4, hold_weight=0.25)
        hold_indices = np.flatnonzero(artifacts.primary_hold_mask > 0.5)
        self.assertGreater(len(hold_indices), 0)
        self.assertTrue(np.all(artifacts.sample_weights[hold_indices] <= 0.5))

    def test_market_dynamics_features_merge_and_influence_context(self):
        index = pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC")
        price_frame = pd.DataFrame(
            {
                "close": np.array([100.0, 100.2, 100.1, 100.4], dtype=np.float32),
                "atr_pct": np.full(4, 0.002, dtype=np.float32),
                "bb_width": np.full(4, 0.01, dtype=np.float32),
                "volume_ratio": np.full(4, 1.1, dtype=np.float32),
                "session_overlap": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            },
            index=index,
        )
        dynamics_frame = pd.DataFrame(
            {
                "market_dynamics_confidence": np.array([0.2, 0.8, 0.7, 0.3], dtype=np.float32),
                "market_dynamics_prob_trend_up": np.array([0.1, 0.7, 0.6, 0.2], dtype=np.float32),
                "market_dynamics_prob_trend_down": np.array([0.2, 0.1, 0.1, 0.2], dtype=np.float32),
                "market_dynamics_prob_breakout": np.array([0.3, 0.8, 0.7, 0.2], dtype=np.float32),
                "market_dynamics_prob_range": np.array([0.6, 0.1, 0.2, 0.7], dtype=np.float32),
                "market_dynamics_prob_mean_reversion": np.array([0.5, 0.1, 0.2, 0.6], dtype=np.float32),
                "market_dynamics_prob_panic_news_shock": np.array([0.1, 0.2, 0.2, 0.4], dtype=np.float32),
                "market_dynamics_prob_high_volatility": np.array([0.2, 0.4, 0.3, 0.5], dtype=np.float32),
            },
            index=index,
        )
        merged = merge_market_dynamics_features(price_frame, dynamics_frame)
        self.assertIn("market_dynamics_confidence", merged.columns)
        context = build_gate_context_matrix(merged)
        self.assertEqual(context.shape[0], 4)
        self.assertGreater(context[1, -4], context[0, -4])

    def test_market_dynamics_can_raise_hold_mask(self):
        frame = pd.DataFrame(
            {
                "close": np.array([100.0, 100.03, 100.02, 100.01, 100.00, 100.02], dtype=np.float32),
                "atr_pct": np.full(6, 0.002, dtype=np.float32),
                "market_dynamics_confidence": np.full(6, 0.8, dtype=np.float32),
                "market_dynamics_prob_range": np.full(6, 0.75, dtype=np.float32),
                "market_dynamics_prob_mean_reversion": np.full(6, 0.6, dtype=np.float32),
                "market_dynamics_prob_false_breakout": np.full(6, 0.2, dtype=np.float32),
                "market_dynamics_prob_breakout": np.full(6, 0.1, dtype=np.float32),
                "market_dynamics_prob_panic_news_shock": np.full(6, 0.1, dtype=np.float32),
                "market_dynamics_prob_high_volatility": np.full(6, 0.1, dtype=np.float32),
                "market_dynamics_prob_low_volatility": np.full(6, 0.8, dtype=np.float32),
                "market_dynamics_prob_trend_up": np.full(6, 0.1, dtype=np.float32),
                "market_dynamics_prob_trend_down": np.full(6, 0.1, dtype=np.float32),
            }
        )
        artifacts = build_trade_target_artifacts(frame, horizons=(1, 2), primary_horizon=1, min_abs_return=1e-4, hold_weight=0.25)
        self.assertGreater(float(artifacts.primary_hold_mask.mean()), 0.0)
        self.assertLessEqual(float(artifacts.sample_weights.max()), 0.5)


if __name__ == "__main__":
    unittest.main()
