import unittest

import numpy as np
import pandas as pd

from src.regime.labeling import DYNAMICS_LABELS, build_market_dynamics_labels, summarize_market_dynamics


class MarketDynamicsLabelingTests(unittest.TestCase):
    def test_build_market_dynamics_labels_emits_probabilities_and_dominant_label(self):
        rows = 64
        close = np.linspace(100.0, 106.0, rows, dtype=np.float32)
        frame = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.4,
                "low": close - 0.5,
                "close": close,
                "atr_pct": np.full(rows, 0.0012, dtype=np.float32),
                "bb_width": np.linspace(0.001, 0.01, rows, dtype=np.float32),
                "bb_pct": np.linspace(0.3, 0.8, rows, dtype=np.float32),
                "volume_ratio": np.linspace(0.8, 1.6, rows, dtype=np.float32),
                "displacement": np.linspace(-0.2, 0.5, rows, dtype=np.float32),
                "ema_cross": np.linspace(-0.1, 1.0, rows, dtype=np.float32),
                "dist_to_high": np.linspace(1.5, 0.1, rows, dtype=np.float32),
                "dist_to_low": np.linspace(0.2, 1.7, rows, dtype=np.float32),
                "return_1": np.gradient(close) / close,
                "quant_transition_risk": np.linspace(0.1, 0.7, rows, dtype=np.float32),
                "quant_state_entropy": np.linspace(0.2, 0.8, rows, dtype=np.float32),
                "quant_tail_risk": np.linspace(0.05, 0.6, rows, dtype=np.float32),
                "quant_regime_strength": np.linspace(0.4, 0.9, rows, dtype=np.float32),
                "quant_trend_score": np.linspace(-0.3, 0.9, rows, dtype=np.float32),
            },
            index=pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
        )

        labels = build_market_dynamics_labels(frame)
        self.assertEqual(len(labels), rows)
        self.assertIn("market_dynamics_label", labels.columns)
        self.assertIn("market_dynamics_confidence", labels.columns)
        for label in DYNAMICS_LABELS:
            self.assertIn(f"market_dynamics_prob_{label}", labels.columns)
        probabilities = labels[[f"market_dynamics_prob_{label}" for label in DYNAMICS_LABELS]].to_numpy(dtype=np.float32)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5))

        report = summarize_market_dynamics(labels)
        self.assertEqual(report.rows, rows)
        self.assertIn("trend_up", report.dominant_counts)


if __name__ == "__main__":
    unittest.main()
