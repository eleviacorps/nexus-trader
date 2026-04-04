import unittest

import numpy as np
import pandas as pd

from src.v11 import augment_v11_context, build_path_conditioned_features, build_setl_features, infer_crowd_state, run_v11_backtest


class V11ResearchTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = []
        sample_configs = [
            ("bullish_trend", [101.2, 100.9, 99.8], [0.55, 0.30, 0.15], 100.7),
            ("panic_news_shock", [99.6, 100.3, 101.0], [0.45, 0.35, 0.20], 100.2),
            ("range", [100.2, 99.9, 100.5], [0.40, 0.34, 0.26], 100.1),
            ("bearish_trend", [99.2, 99.7, 100.4], [0.50, 0.32, 0.18], 99.4),
            ("bullish_trend", [101.4, 100.8, 100.0], [0.58, 0.24, 0.18], 100.9),
            ("panic_news_shock", [99.4, 100.2, 101.3], [0.44, 0.33, 0.23], 100.1),
        ]
        for sample_id, (regime, terminals, probs, actual_15) in enumerate(sample_configs):
            for branch_id, terminal in enumerate(terminals, start=1):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "timestamp": f"2026-04-{sample_id + 1:02d}T10:00:00Z",
                        "year": 2026,
                        "branch_id": branch_id,
                        "dominant_regime": regime,
                        "generator_probability": probs[branch_id - 1],
                        "hmm_regime_match": [0.92, 0.60, 0.20][branch_id - 1],
                        "hmm_persistence": 0.75,
                        "hmm_transition_risk": 0.18,
                        "volatility_realism": [0.88, 0.72, 0.45][branch_id - 1],
                        "branch_move_zscore": [0.8, 0.1, -0.9][branch_id - 1],
                        "fair_value_dislocation": [0.01, 0.02, 0.04][branch_id - 1],
                        "mean_reversion_pressure": [0.25, 0.45, 0.80][branch_id - 1],
                        "analog_similarity": [0.82, 0.58, 0.35][branch_id - 1],
                        "analog_disagreement": [0.10, 0.22, 0.65][branch_id - 1],
                        "news_consistency": [0.78, 0.52, 0.25][branch_id - 1],
                        "crowd_consistency": [0.80, 0.48, 0.20][branch_id - 1],
                        "macro_alignment": [0.76, 0.55, 0.18][branch_id - 1],
                        "branch_direction": [1.0, 1.0 if branch_id < 3 else -1.0, -1.0][branch_id - 1],
                        "branch_move_size": (terminal / 100.0) - 1.0,
                        "branch_volatility": [0.003, 0.0025, 0.0042][branch_id - 1],
                        "vwap_distance": [0.02, 0.01, -0.02][branch_id - 1],
                        "atr_normalized_move": [1.05, 0.72, 1.40][branch_id - 1],
                        "branch_entropy": [0.21, 0.28, 0.45][branch_id - 1],
                        "branch_confidence": [0.82, 0.56, 0.30][branch_id - 1],
                        "path_error": [0.002, 0.006, 0.015][branch_id - 1],
                        "actual_final_return": (actual_15 / 100.0) - 1.0,
                        "actual_price_5m": 100.25 if regime != "panic_news_shock" else 99.8,
                        "actual_price_10m": 100.45 if regime != "panic_news_shock" else 100.0,
                        "actual_price_15m": actual_15,
                        "predicted_price_5m": [100.22, 100.10, 99.85][branch_id - 1],
                        "predicted_price_10m": [100.55, 100.25, 99.75][branch_id - 1],
                        "predicted_price_15m": terminal,
                        "anchor_price": 100.0,
                        "entry_open_price": 100.02,
                        "exit_close_price_15m": actual_15,
                        "volatility_scale": 1.15 if regime != "panic_news_shock" else 1.8,
                        "model_direction_prob_15m": 0.63,
                        "model_hold_prob_15m": 0.10,
                        "model_confidence_prob_15m": 0.5,
                        "leaf_branch_fitness": [0.76, 0.48, 0.22][branch_id - 1],
                        "leaf_analog_confidence": [0.80, 0.54, 0.28][branch_id - 1],
                        "leaf_minority_guardrail": [0.18, 0.28, 0.74][branch_id - 1],
                        "leaf_branch_label": ["trend", "drift", "reversal"][branch_id - 1],
                        "winner_label": 1 if branch_id == 1 else 0,
                        "winning_branch": branch_id == 1,
                        "v10_generation_mode": "source",
                        "v10_temperature": 1.0,
                        "v10_regime_label": regime,
                        "v10_minority_rescue": False,
                        "branch_label": ["trend", "drift", "reversal"][branch_id - 1],
                        "v10_diversity_score": [0.72, 0.52, 0.34][branch_id - 1],
                        "final_price_accuracy": [0.88, 0.55, 0.30][branch_id - 1],
                        "full_path_similarity": [0.86, 0.60, 0.35][branch_id - 1],
                        "execution_realism": [0.78, 0.54, 0.28][branch_id - 1],
                        "regime_consistency": [0.84, 0.58, 0.32][branch_id - 1],
                        "volatility_realism_v9": [0.88, 0.72, 0.45][branch_id - 1],
                        "composite_score": [0.84, 0.57, 0.31][branch_id - 1],
                        "top_1_branch": 1,
                        "top_3_branches": "[1,2,3]",
                        "inside_confidence_cone": True,
                        "minority_rescue_branch": 3,
                        "is_top_1_branch": 1 if branch_id == 1 else 0,
                        "is_top_3_branch": 1,
                        "is_minority_rescue_branch": 1 if branch_id == 3 else 0,
                        "composite_winner_label": 1 if branch_id == 1 else 0,
                        "path_curvature": [0.10, 0.18, 0.35][branch_id - 1],
                        "path_acceleration": [0.08, 0.14, 0.30][branch_id - 1],
                        "path_entropy": [0.18, 0.26, 0.42][branch_id - 1],
                        "path_smoothness": [0.82, 0.68, 0.45][branch_id - 1],
                        "path_convexity": [0.22, 0.05, -0.18][branch_id - 1],
                        "reversal_likelihood": [0.14, 0.24, 0.72][branch_id - 1],
                        "breakout_likelihood": [0.66, 0.35, 0.18][branch_id - 1],
                        "mean_reversion_likelihood": [0.22, 0.48, 0.80][branch_id - 1],
                        "news_consistency_v9": [0.78, 0.52, 0.25][branch_id - 1],
                        "macro_consistency_v9": [0.76, 0.55, 0.18][branch_id - 1],
                        "crowd_consistency_v9": [0.80, 0.48, 0.20][branch_id - 1],
                        "order_flow_plausibility": [0.82, 0.58, 0.33][branch_id - 1],
                        "analog_density": [15.0, 9.0, 4.0][branch_id - 1],
                        "analog_disagreement_v9": [0.10, 0.22, 0.65][branch_id - 1],
                        "analog_weighted_accuracy": [0.74, 0.45, 0.12][branch_id - 1],
                        "hmm_regime_probability": [0.80, 0.62, 0.28][branch_id - 1],
                        "regime_persistence": 0.75,
                        "regime_transition_risk_v9": 0.18,
                        "garch_zscore": [0.8, 0.1, -0.9][branch_id - 1],
                        "fair_value_distance": [0.01, 0.02, 0.04][branch_id - 1],
                        "fair_value_mean_reversion_prob": [0.32, 0.46, 0.70][branch_id - 1],
                        "atr_normalised_move_v9": [1.05, 0.72, 1.40][branch_id - 1],
                        "historical_move_percentile": [0.72, 0.48, 0.20][branch_id - 1],
                        "branch_disagreement": 0.0022 if regime != "range" else 0.0014,
                        "consensus_direction": 1.0,
                        "consensus_strength": 0.74 if regime != "range" else 0.60,
                        "regime_match_x_analog": [0.61, 0.32, 0.08][branch_id - 1],
                        "volatility_realism_x_fair_value": [35.0, 20.0, 10.0][branch_id - 1],
                        "news_x_crowd": [0.62, 0.25, 0.05][branch_id - 1],
                        "analog_density_x_regime_persistence": [11.25, 6.75, 3.0][branch_id - 1],
                    }
                )
        return pd.DataFrame(rows)

    def test_crowd_state_and_context_build(self) -> None:
        frame = augment_v11_context(self._frame())
        self.assertIn("cesm_state", frame.columns)
        self.assertIn("pmwm_institutional_positioning", frame.columns)
        snapshot = infer_crowd_state(frame.drop_duplicates("sample_id").iloc[0].to_dict())
        self.assertGreaterEqual(snapshot.confidence, 0.0)

    def test_conditioning_and_setl_features(self) -> None:
        frame = augment_v11_context(self._frame())
        conditioned = build_path_conditioned_features(frame.assign(selector_score=0.5), stage_bars=5)
        self.assertIn("pcop_error_now", conditioned.columns)
        selected = conditioned.groupby("sample_id", sort=False).head(1).copy()
        selected["pcop_survival_score"] = 0.5
        setl = build_setl_features(selected, stage_bars=5)
        self.assertIn("predicted_edge", setl.columns)
        self.assertTrue(np.isfinite(setl["predicted_edge"].to_numpy(dtype=np.float32)).all())

    def test_run_v11_backtest(self) -> None:
        summary = run_v11_backtest(self._frame(), validation_fraction=0.34)
        self.assertIn("variants", summary)
        self.assertIn("full_v11", summary["variants"])
        self.assertIn("stage_usage", summary["variants"]["full_v11"])


if __name__ == "__main__":
    unittest.main()
