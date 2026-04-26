from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

from src.v8.branch_selector_v8 import build_branch_archive_frame, evaluate_branch_selector, train_branch_selector_v8, load_branch_selector_v8
from src.v8.fair_value import build_fair_value_frame
from src.v8.garch_volatility import build_garch_like_frame
from src.v8.hmm_regime import build_hmm_regime_frame


@unittest.skipIf(pd is None, "pandas is required")
class V8StackTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2026-01-01", periods=240, freq="min")
        close = np.linspace(100.0, 103.0, len(index), dtype=np.float32) + np.sin(np.arange(len(index)) / 15.0).astype(np.float32) * 0.3
        self.frame = pd.DataFrame(
            {
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "return_1": pd.Series(close).pct_change().fillna(0.0).to_numpy(dtype=np.float32),
                "atr_pct": np.full(len(index), 0.0012, dtype=np.float32),
                "ema_cross": np.tanh(np.linspace(-1.0, 1.0, len(index))).astype(np.float32),
                "volume_ratio": np.full(len(index), 1.0, dtype=np.float32),
                "quant_trend_score": np.tanh(np.linspace(-0.8, 0.8, len(index))).astype(np.float32),
                "quant_kalman_dislocation": np.linspace(-0.01, 0.01, len(index)).astype(np.float32),
                "bb_width": np.full(len(index), 0.0025, dtype=np.float32),
                "forward_return_15m": pd.Series(close).shift(-15).div(pd.Series(close)).sub(1.0).fillna(0.0).to_numpy(dtype=np.float32),
            },
            index=index,
        )

    def test_quant_frames_build(self):
        hmm = build_hmm_regime_frame(self.frame)
        garch = build_garch_like_frame(self.frame)
        fair_value = build_fair_value_frame(self.frame)
        self.assertEqual(len(hmm), len(self.frame))
        self.assertEqual(len(garch), len(self.frame))
        self.assertEqual(len(fair_value), len(self.frame))
        self.assertIn("hmm_dominant_regime", hmm.columns)
        self.assertIn("v8_expected_vol_15m", garch.columns)
        self.assertIn("v8_fair_value_dislocation", fair_value.columns)

    def test_branch_selector_training_and_eval(self):
        rows = []
        for sample_id in range(10):
            for branch_id in range(4):
                winner = 1 if branch_id == (sample_id % 4) else 0
                rows.append(
                    {
                        "sample_id": sample_id,
                        "branch_id": branch_id,
                        "winner_label": winner,
                        "path_error": 0.1 if winner else 0.6 + (branch_id * 0.05),
                        "actual_final_return": 0.01 if winner else -0.01,
                        "generator_probability": 0.9 if winner else 0.2,
                        "hmm_regime_match": 0.9 if winner else 0.2,
                        "hmm_persistence": 0.8,
                        "hmm_transition_risk": 0.2,
                        "volatility_realism": 0.9 if winner else 0.4,
                        "branch_move_zscore": 0.2 if winner else 1.4,
                        "fair_value_dislocation": 0.01 if winner else 0.04,
                        "mean_reversion_pressure": 0.2,
                        "analog_similarity": 0.8 if winner else 0.3,
                        "analog_disagreement": 0.2 if winner else 0.6,
                        "news_consistency": 0.7,
                        "crowd_consistency": 0.6,
                        "macro_alignment": 0.8 if winner else 0.4,
                        "branch_direction": 1.0 if winner else -1.0,
                        "branch_move_size": 0.01 if winner else 0.03,
                        "branch_volatility": 0.002 if winner else 0.006,
                        "vwap_distance": 0.001,
                        "atr_normalized_move": 0.8 if winner else 2.0,
                        "branch_entropy": 0.2,
                        "branch_confidence": 0.85 if winner else 0.35,
                        "dominant_regime": "bullish_trend",
                    }
                )
        frame = build_branch_archive_frame(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selector.pkl"
            train_branch_selector_v8(frame, path)
            selector = load_branch_selector_v8(path)
            report = evaluate_branch_selector(frame, selector)
        self.assertGreaterEqual(report["top1_branch_accuracy"], 0.25)
        self.assertTrue(np.isfinite(report["selector_error_improvement"]))


if __name__ == "__main__":
    unittest.main()
