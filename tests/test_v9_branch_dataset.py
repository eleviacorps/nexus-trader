import unittest

import numpy as np
import pandas as pd

from src.v9 import BRANCH_FEATURES_V9, build_branch_features, build_branch_labels, train_selector_torch


class V9BranchDatasetTests(unittest.TestCase):
    def _sample_frame(self) -> pd.DataFrame:
        rows = []
        for sample_id in range(4):
            actual_5 = 100.5 + sample_id * 0.1
            actual_10 = 101.0 + sample_id * 0.1
            actual_15 = 101.5 + sample_id * 0.1
            for branch_id, terminal in enumerate([101.45, 100.6, 99.9], start=1):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "timestamp": f"2026-01-0{sample_id + 1}T0{sample_id}:00:00Z",
                        "branch_id": branch_id,
                        "dominant_regime": "bullish_trend",
                        "generator_probability": [0.55, 0.3, 0.15][branch_id - 1],
                        "hmm_regime_match": [0.95, 0.5, 0.1][branch_id - 1],
                        "hmm_persistence": 0.8,
                        "hmm_transition_risk": 0.15,
                        "volatility_realism": [0.9, 0.7, 0.45][branch_id - 1],
                        "branch_move_zscore": [0.8, 0.2, -1.4][branch_id - 1],
                        "fair_value_dislocation": [0.01, 0.03, 0.05][branch_id - 1],
                        "mean_reversion_pressure": [0.2, 0.5, 0.9][branch_id - 1],
                        "analog_similarity": [0.9, 0.6, 0.25][branch_id - 1],
                        "analog_disagreement": [0.1, 0.25, 0.7][branch_id - 1],
                        "news_consistency": [0.9, 0.5, 0.2][branch_id - 1],
                        "crowd_consistency": [0.85, 0.55, 0.25][branch_id - 1],
                        "macro_alignment": [0.9, 0.45, 0.15][branch_id - 1],
                        "branch_direction": [1.0, 1.0, -1.0][branch_id - 1],
                        "branch_move_size": [0.014, 0.006, -0.01][branch_id - 1],
                        "branch_volatility": [0.004, 0.003, 0.006][branch_id - 1],
                        "vwap_distance": [0.02, 0.01, -0.03][branch_id - 1],
                        "atr_normalized_move": [1.2, 0.7, 1.6][branch_id - 1],
                        "branch_entropy": [0.2, 0.3, 0.5][branch_id - 1],
                        "branch_confidence": [0.88, 0.6, 0.25][branch_id - 1],
                        "actual_final_return": 0.015,
                        "actual_price_5m": actual_5,
                        "actual_price_10m": actual_10,
                        "actual_price_15m": actual_15,
                        "predicted_price_5m": [100.45, 100.3, 99.8][branch_id - 1] + sample_id * 0.1,
                        "predicted_price_10m": [100.95, 100.5, 99.7][branch_id - 1] + sample_id * 0.1,
                        "predicted_price_15m": terminal + sample_id * 0.1,
                        "anchor_price": 100.0 + sample_id * 0.1,
                        "entry_open_price": 100.1 + sample_id * 0.1,
                        "exit_close_price_15m": actual_15,
                        "volatility_scale": 1.0,
                        "model_direction_prob_15m": 0.62,
                        "model_hold_prob_15m": 0.12,
                        "model_confidence_prob_15m": 0.5,
                        "leaf_branch_fitness": [0.8, 0.5, 0.1][branch_id - 1],
                        "leaf_analog_confidence": [0.85, 0.55, 0.25][branch_id - 1],
                        "leaf_minority_guardrail": [0.2, 0.3, 0.8][branch_id - 1],
                        "leaf_branch_label": ["trend", "drift", "reversal"][branch_id - 1],
                        "winner_label": 1 if branch_id == 1 else 0,
                        "winning_branch": branch_id == 1,
                    }
                )
        return pd.DataFrame(rows)

    def test_build_branch_labels(self) -> None:
        frame = self._sample_frame()
        labeled = build_branch_labels(frame)
        self.assertIn("composite_score", labeled.columns)
        self.assertIn("top_3_branches", labeled.columns)
        self.assertTrue((labeled["composite_score"] >= 0.0).all())
        sample = labeled.loc[labeled["sample_id"] == 0]
        self.assertEqual(int(sample["top_1_branch"].iloc[0]), 1)
        self.assertEqual(int(sample["is_top_1_branch"].sum()), 1)
        self.assertEqual(int(sample["is_top_3_branch"].sum()), 3)

    def test_build_features_and_train_torch_selector(self) -> None:
        frame = self._sample_frame()
        labeled = build_branch_labels(frame)
        features = build_branch_features(labeled)
        for column in BRANCH_FEATURES_V9:
            self.assertIn(column, features.columns)
            self.assertTrue(np.isfinite(features[column].to_numpy(dtype=np.float32)).all())
        _, report = train_selector_torch(features, epochs=2, batch_size=2, validation_fraction=0.25)
        self.assertGreaterEqual(report.top1_accuracy, 0.0)
        self.assertLessEqual(report.top1_accuracy, 1.0)
        self.assertGreaterEqual(report.top3_containment, 0.0)
        self.assertLessEqual(report.top3_containment, 1.0)


if __name__ == "__main__":
    unittest.main()
