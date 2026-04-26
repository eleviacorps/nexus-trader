from __future__ import annotations

import unittest

import numpy as np

from src.v19.lepl import LEPL_ACTIONS, LiveExecutionPolicy


class V19LeplTests(unittest.TestCase):
    def test_predict_returns_valid_action(self) -> None:
        policy = LiveExecutionPolicy()
        X = np.asarray(
            [
                policy._features_to_vector({"sjd_stance": "BUY", "sjd_confidence": "HIGH", "sqt_label": "HOT", "cabr_score": 0.8, "hurst_asymmetry": 0.2, "mfg_disagreement": 0.1, "cpm_score": 0.8, "has_open_position": False, "open_position_pnl": 0.0}),
                policy._features_to_vector({"sjd_stance": "HOLD", "sjd_confidence": "VERY_LOW", "sqt_label": "COLD", "cabr_score": 0.2, "hurst_asymmetry": -0.1, "mfg_disagreement": 0.8, "cpm_score": 0.2, "has_open_position": False, "open_position_pnl": 0.0}),
                policy._features_to_vector({"sjd_stance": "SELL", "sjd_confidence": "MODERATE", "sqt_label": "GOOD", "cabr_score": 0.6, "hurst_asymmetry": -0.2, "mfg_disagreement": 0.2, "cpm_score": 0.6, "has_open_position": True, "open_position_pnl": -12.0}),
                policy._features_to_vector({"sjd_stance": "BUY", "sjd_confidence": "LOW", "sqt_label": "NEUTRAL", "cabr_score": 0.5, "hurst_asymmetry": 0.1, "mfg_disagreement": 0.3, "cpm_score": 0.5, "has_open_position": True, "open_position_pnl": 6.0}),
            ],
            dtype=np.float32,
        )
        y = np.asarray(["ENTER", "NOTHING", "CLOSE", "HOLD"])
        policy.fit(X, y)
        action = policy.predict(
            {
                "sjd_stance": "BUY",
                "sjd_confidence": "HIGH",
                "sqt_label": "HOT",
                "cabr_score": 0.82,
                "hurst_asymmetry": 0.18,
                "mfg_disagreement": 0.1,
                "cpm_score": 0.79,
                "has_open_position": False,
                "open_position_pnl": 0.0,
            }
        )
        self.assertIn(action, LEPL_ACTIONS)


if __name__ == "__main__":
    unittest.main()
