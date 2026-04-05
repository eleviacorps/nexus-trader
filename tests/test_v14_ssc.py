from __future__ import annotations

import unittest

import pandas as pd
import torch

from src.v14.ssc import SimulationCritic, build_ssc_labels


class TestV14SSC(unittest.TestCase):
    def test_model_smoke(self) -> None:
        model = SimulationCritic(branch_dim=4, context_dim=3)
        out = model(torch.zeros((2, 4)), torch.zeros((2, 3)))
        self.assertEqual(tuple(out.shape), (2, 3))

    def test_label_builder(self) -> None:
        frame = pd.DataFrame(
            {
                "dominant_regime": ["ranging", "ranging", "trending_up"],
                "branch_label": ["a", "a", "b"],
                "setl_target_net_unit_pnl": [1.0, -1.0, 2.0],
                "branch_direction": [1.0, -1.0, 1.0],
                "bcfe_macd_hist": [0.1, -0.1, 0.2],
                "anchor_price": [100.0, 100.0, 100.0],
                "predicted_price_5m": [101.0, 99.0, 101.5],
                "predicted_price_10m": [102.0, 98.5, 102.0],
                "predicted_price_15m": [103.0, 98.0, 102.5],
            }
        )
        labeled = build_ssc_labels(frame)
        self.assertIn("ssc_assumption_risk", labeled.columns)
        self.assertIn("ssc_context_consistency", labeled.columns)
        self.assertIn("ssc_contradiction_depth", labeled.columns)


if __name__ == "__main__":
    unittest.main()
