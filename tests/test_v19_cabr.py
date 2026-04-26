from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.v13.cabr import CABR, evaluate_cabr_pairwise_accuracy
from src.v19.curriculum_pairs import build_curriculum_pair_payload


class V19CabrTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = []
        for idx in range(24):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(minutes=15 * idx),
                    "sample_id": idx // 3,
                    "setl_target_net_unit_pnl": float((idx % 3) - 1),
                    "actual_final_return": float((idx % 3) - 1),
                    "predicted_price_15m": 2000.0 + idx + (idx % 3),
                    "anchor_price": 2000.0 + idx,
                    "path_error": 0.05 * (idx % 3),
                    "branch_entropy": 0.2 + (0.01 * idx),
                    "path_entropy": 0.2 + (0.01 * idx),
                    "path_smoothness": 0.5,
                    "reversal_likelihood": 0.2,
                    "mean_reversion_likelihood": 0.2,
                    "v10_diversity_score": 0.4,
                    "analog_similarity": 0.6,
                    "leaf_analog_confidence": 0.55,
                    "consensus_strength": 0.5,
                    "dominant_regime": "trending_up" if idx % 2 == 0 else "ranging",
                    "regime_class": "trend_up" if idx % 2 == 0 else "range",
                    "context_regime_confidence": 0.6,
                    "context_atr_percentile_30d": 0.5,
                    "context_rsi_14": 55.0,
                    "context_macd_hist": 0.2,
                    "context_bb_pct": 0.4,
                    "context_days_since_regime_change": 2.0,
                    "context_emotional_momentum": 0.1,
                    "context_emotional_fragility": 0.2,
                    "context_emotional_conviction": 0.3,
                    "context_narrative_age": 1.0,
                    "fear_index_retail": 0.2,
                    "fear_index_institutional": 0.2,
                    "fear_index_algo": 0.2,
                    "context_hurst_overall": 0.55,
                    "context_hurst_positive": 0.6,
                    "context_hurst_negative": 0.45,
                    "context_hurst_asymmetry": 0.15,
                }
            )
        return pd.DataFrame(rows)

    def test_curriculum_pair_builder_produces_easy_and_full_payloads(self) -> None:
        frame = self._frame()
        branch_cols = (
            "branch_entropy",
            "path_entropy",
            "path_smoothness",
            "reversal_likelihood",
            "mean_reversion_likelihood",
            "v10_diversity_score",
            "analog_similarity",
            "leaf_analog_confidence",
            "consensus_strength",
        )
        context_cols = (
            "context_regime_confidence",
            "context_atr_percentile_30d",
            "context_rsi_14",
            "context_macd_hist",
            "context_bb_pct",
            "context_days_since_regime_change",
            "context_emotional_momentum",
            "context_emotional_fragility",
            "context_emotional_conviction",
            "context_narrative_age",
            "fear_index_retail",
            "fear_index_institutional",
            "fear_index_algo",
            "context_hurst_overall",
            "context_hurst_positive",
            "context_hurst_negative",
            "context_hurst_asymmetry",
        )
        payloads = build_curriculum_pair_payload(
            frame,
            branch_feature_names=branch_cols,
            context_feature_names=context_cols,
            max_pairs=500,
        )
        self.assertGreater(len(payloads["easy"]["pairs"]), 0)
        self.assertGreater(len(payloads["full"]["pairs"]), 0)

    def test_model_scores_curriculum_payload(self) -> None:
        frame = self._frame()
        branch_cols = (
            "branch_entropy",
            "path_entropy",
            "path_smoothness",
            "reversal_likelihood",
            "mean_reversion_likelihood",
            "v10_diversity_score",
            "analog_similarity",
            "leaf_analog_confidence",
            "consensus_strength",
        )
        context_cols = (
            "context_regime_confidence",
            "context_atr_percentile_30d",
            "context_rsi_14",
            "context_macd_hist",
            "context_bb_pct",
            "context_days_since_regime_change",
            "context_emotional_momentum",
            "context_emotional_fragility",
            "context_emotional_conviction",
            "context_narrative_age",
            "fear_index_retail",
            "fear_index_institutional",
            "fear_index_algo",
            "context_hurst_overall",
            "context_hurst_positive",
            "context_hurst_negative",
            "context_hurst_asymmetry",
        )
        payload = build_curriculum_pair_payload(
            frame,
            branch_feature_names=branch_cols,
            context_feature_names=context_cols,
            max_pairs=120,
        )["easy"]
        model = CABR(len(branch_cols), len(context_cols))
        metrics = evaluate_cabr_pairwise_accuracy(model, payload, device="cpu")
        self.assertIn("overall_accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
