import unittest

import numpy as np
import pandas as pd

from src.quant.hybrid import QUANT_FEATURE_COLUMNS, build_quant_features, merge_quant_features


class QuantHybridTests(unittest.TestCase):
    def _sample_price_frame(self) -> pd.DataFrame:
        idx = pd.date_range("2026-01-01", periods=240, freq="5min")
        base = np.linspace(4500.0, 4560.0, len(idx))
        noise = np.sin(np.arange(len(idx)) / 7.0) * 4.0
        close = base + noise
        frame = pd.DataFrame(
            {
                "close": close,
                "return_1": pd.Series(close).pct_change().fillna(0.0).to_numpy(),
                "return_6": pd.Series(close).pct_change(6).fillna(0.0).to_numpy(),
                "return_12": pd.Series(close).pct_change(12).fillna(0.0).to_numpy(),
                "atr_pct": np.full(len(idx), 0.002, dtype=np.float32),
                "bb_width": np.full(len(idx), 0.03, dtype=np.float32),
                "ema_cross": np.tanh(np.linspace(-1.0, 1.0, len(idx))),
                "macd_hist": np.tanh(np.linspace(-0.5, 0.8, len(idx))),
                "rsi_14": np.clip(50.0 + np.sin(np.arange(len(idx)) / 9.0) * 14.0, 1.0, 99.0),
            },
            index=idx,
        )
        return frame

    def test_build_quant_features_outputs_expected_columns(self):
        frame = self._sample_price_frame()
        quant = build_quant_features(frame)
        self.assertEqual(list(quant.columns), QUANT_FEATURE_COLUMNS)
        self.assertEqual(len(quant), len(frame))
        self.assertTrue(np.all(np.isfinite(quant.to_numpy(dtype=np.float32))))
        self.assertTrue(np.all((quant["quant_transition_risk"] >= 0.0) & (quant["quant_transition_risk"] <= 1.0)))

    def test_merge_quant_features_preserves_price_rows(self):
        frame = self._sample_price_frame()
        quant = build_quant_features(frame)
        merged = merge_quant_features(frame, quant)
        self.assertEqual(len(merged), len(frame))
        for column in QUANT_FEATURE_COLUMNS:
            self.assertIn(column, merged.columns)


if __name__ == "__main__":
    unittest.main()
