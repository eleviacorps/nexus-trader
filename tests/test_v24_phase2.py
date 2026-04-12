import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.v24 import HeuristicMetaAggregator, LearnedMetaAggregator, MetaAggregatorModel, MetaAggregatorModelConfig, build_world_state, load_meta_aggregator


class V24Phase2Tests(unittest.TestCase):
    def test_meta_aggregator_model_forward_shapes(self) -> None:
        model = MetaAggregatorModel(MetaAggregatorModelConfig(static_dim=12, sequence_dim=8, seq_len=16, hidden_dim=32))
        outputs = model(torch.randn(3, 16, 8), torch.randn(3, 12))
        self.assertEqual(tuple(outputs["expected_value"].shape), (3,))
        self.assertEqual(tuple(outputs["expert_weights"].shape), (3, 5))
        self.assertTrue(torch.all(outputs["danger_score"] >= 0.0))
        self.assertTrue(torch.all(outputs["danger_score"] <= 1.0))

    def test_load_meta_aggregator_falls_back_to_heuristic(self) -> None:
        aggregator = load_meta_aggregator(
            preference="auto",
            checkpoint_path=Path("models/v24/does_not_exist.pt"),
            config_path=Path("models/v24/does_not_exist.json"),
        )
        self.assertIsInstance(aggregator, HeuristicMetaAggregator)

    def test_learned_meta_aggregator_loads_synthetic_checkpoint(self) -> None:
        model = MetaAggregatorModel(MetaAggregatorModelConfig(static_dim=4, sequence_dim=8, seq_len=16, hidden_dim=16))
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = root / "meta.pt"
            config_path = root / "meta.json"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": MetaAggregatorModelConfig(static_dim=4, sequence_dim=8, seq_len=16, hidden_dim=16).to_dict(),
                    "static_feature_names": ["market_structure_rr_ratio", "nexus_features_cabr_score", "quant_models_hmm_confidence", "execution_context_v22_risk_score"],
                    "sequence_feature_names": ["a", "b", "c", "d", "e", "f", "g", "h"],
                    "static_mean": [0.0, 0.0, 0.0, 0.0],
                    "static_std": [1.0, 1.0, 1.0, 1.0],
                    "sequence_mean": [0.0] * 8,
                    "sequence_std": [1.0] * 8,
                },
                checkpoint_path,
            )
            config_path.write_text(json.dumps({"heuristic_prior_weight": 0.25, "model_config": MetaAggregatorModelConfig(static_dim=4, sequence_dim=8, seq_len=16, hidden_dim=16).to_dict()}), encoding="utf-8")
            aggregator = LearnedMetaAggregator.from_artifacts(checkpoint_path, config_path)
            state = build_world_state(
                {
                    "signal_time_utc": "2024-12-03T10:15:00Z",
                    "action": "BUY",
                    "rr_ratio": 1.8,
                    "cabr_score": 0.72,
                    "cpm_score": 0.69,
                    "online_hmm_regime_confidence": 0.66,
                },
                ensemble_state={"risk_score": 0.25},
            )
            estimate = aggregator.predict(state, sequence_features=np.zeros((16, 8), dtype=np.float32))
            self.assertGreaterEqual(estimate.profit_probability, 0.0)
            self.assertLessEqual(estimate.profit_probability, 1.0)
            self.assertIn("learned_meta_aggregator", estimate.notes)


if __name__ == "__main__":
    unittest.main()
