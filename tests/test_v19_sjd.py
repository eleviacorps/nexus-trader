from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.v19.sjd_model import JudgmentDistillationModel, load_sjd_bundle, predict_sjd_from_context, sjd_loss, train_sjd_model


class V19SjdTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = JudgmentDistillationModel(input_dim=8)
        outputs = model(torch.randn(4, 8))
        self.assertEqual(outputs["stance_logits"].shape, (4, 3))
        self.assertEqual(outputs["confidence_logits"].shape, (4, 4))
        self.assertEqual(outputs["level_offsets"].shape, (4, 3))

    def test_loss_runs(self) -> None:
        model = JudgmentDistillationModel(input_dim=8)
        outputs = model(torch.randn(4, 8))
        loss = sjd_loss(
            outputs,
            torch.tensor([0, 1, 2, 0]),
            torch.tensor([0, 1, 2, 3]),
            torch.randn(4, 3),
            torch.tensor([1.0, 1.0, 0.0, 1.0]),
        )
        self.assertGreater(float(loss.detach().cpu()), 0.0)

    def test_train_and_predict_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "dataset.parquet"
            features_path = tmp / "features.json"
            checkpoint_path = tmp / "sjd.pt"
            feature_names = ["market.current_price", "simulation.cabr_score", "cat.direction.buy", "cat.direction.sell"]
            rows = []
            for idx in range(12):
                buy = 1.0 if idx % 3 == 0 else 0.0
                sell = 1.0 if idx % 3 == 1 else 0.0
                rows.append(
                    {
                        "feature_vector": json.dumps([2000.0 + idx, 0.6, buy, sell]),
                        "stance": "BUY" if buy else "SELL" if sell else "HOLD",
                        "confidence": "MODERATE",
                        "entry_offset": 2.0,
                        "sl_offset": -8.0,
                        "tp_offset": 12.0,
                    }
                )
            pd.DataFrame(rows).to_parquet(dataset_path, index=False)
            features_path.write_text(json.dumps(feature_names), encoding="utf-8")
            train_sjd_model(
                dataset_path=dataset_path,
                feature_names_path=features_path,
                checkpoint_path=checkpoint_path,
                epochs=1,
                batch_size=4,
                device="cpu",
            )
            bundle = load_sjd_bundle(path=checkpoint_path, device="cpu")
            prediction = predict_sjd_from_context(
                bundle,
                {
                    "market": {"current_price": 2012.0},
                    "simulation": {"direction": "BUY", "cabr_score": 0.6, "cpm_score": 0.6, "hurst_asymmetry": 0.1},
                    "technical_analysis": {"structure": "bullish", "location": "discount"},
                    "bot_swarm": {"aggregate": {"signal": "bullish"}},
                    "sqt": {"label": "HOT"},
                },
            )
            self.assertIn(prediction["final_call"], {"BUY", "SELL", "SKIP"})


if __name__ == "__main__":
    unittest.main()
