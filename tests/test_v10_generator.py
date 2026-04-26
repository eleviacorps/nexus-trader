import unittest

import numpy as np
import pandas as pd

from src.v10 import audit_branch_archive, diversify_archive_sample, infer_generation_regime


class V10GeneratorTests(unittest.TestCase):
    def _sample_frame(self) -> pd.DataFrame:
        rows = []
        for branch_id, terminal in enumerate([101.15, 101.10, 101.05, 101.00], start=1):
            rows.append(
                {
                    "sample_id": 7,
                    "timestamp": "2026-04-01T10:00:00Z",
                    "branch_id": branch_id,
                    "dominant_regime": "bullish_trend",
                    "generator_probability": [0.42, 0.31, 0.17, 0.10][branch_id - 1],
                    "hmm_regime_match": 0.82,
                    "hmm_persistence": 0.80,
                    "hmm_transition_risk": 0.18,
                    "volatility_realism": 0.88,
                    "branch_move_zscore": [0.75, 0.70, 0.66, 0.62][branch_id - 1],
                    "fair_value_dislocation": 0.02,
                    "mean_reversion_pressure": 0.25,
                    "analog_similarity": 0.74,
                    "analog_disagreement": 0.18,
                    "news_consistency": 0.72,
                    "crowd_consistency": 0.69,
                    "macro_alignment": 0.77,
                    "branch_direction": 1.0,
                    "branch_move_size": (terminal / 100.0) - 1.0,
                    "branch_volatility": 0.0025,
                    "vwap_distance": 0.01,
                    "atr_normalized_move": 0.95,
                    "branch_entropy": 0.2,
                    "branch_confidence": 0.74,
                    "actual_final_return": 0.005,
                    "actual_price_5m": 100.25,
                    "actual_price_10m": 100.55,
                    "actual_price_15m": 100.80,
                    "predicted_price_5m": 100.28 - branch_id * 0.01,
                    "predicted_price_10m": 100.62 - branch_id * 0.02,
                    "predicted_price_15m": terminal,
                    "anchor_price": 100.0,
                    "entry_open_price": 100.05,
                    "exit_close_price_15m": 100.80,
                    "volatility_scale": 1.1,
                    "model_direction_prob_15m": 0.64,
                    "model_hold_prob_15m": 0.08,
                    "model_confidence_prob_15m": 0.5,
                    "leaf_branch_fitness": 0.72,
                    "leaf_analog_confidence": 0.66,
                    "leaf_minority_guardrail": 0.10,
                    "leaf_branch_label": "trend",
                    "winner_label": 1 if branch_id == 1 else 0,
                    "winning_branch": branch_id == 1,
                }
            )
        return pd.DataFrame(rows)

    def test_infer_generation_regime(self) -> None:
        profile = infer_generation_regime(self._sample_frame().iloc[0].to_dict())
        self.assertGreater(profile.temperature_ceiling, profile.temperature_floor)
        self.assertGreaterEqual(profile.target_branch_count, 8)

    def test_diversify_archive_sample_increases_directional_spread(self) -> None:
        source = self._sample_frame()
        diversified, report = diversify_archive_sample(source)
        self.assertGreaterEqual(len(diversified), 2)
        self.assertIn(-1.0, diversified["branch_direction"].to_numpy(dtype=np.float32))
        self.assertGreaterEqual(report.minority_share, 0.0)
        self.assertTrue(np.isfinite(diversified["generator_probability"].to_numpy(dtype=np.float32)).all())

    def test_audit_branch_archive_reports_containment(self) -> None:
        summary = audit_branch_archive(self._sample_frame())
        self.assertGreaterEqual(summary.sample_count, 1)
        self.assertGreaterEqual(summary.mean_consensus_strength, 0.0)
        self.assertLessEqual(summary.mean_consensus_strength, 1.0)


if __name__ == "__main__":
    unittest.main()
