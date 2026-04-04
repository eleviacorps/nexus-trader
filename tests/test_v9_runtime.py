import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.simulation.personas import default_personas
from src.v9 import (
    aggregate_persona_signals,
    classify_contradiction,
    default_persona_state,
    load_latest_persona_state,
    projected_move_from_context,
    regret_gate_scores,
    update_persona_calibration,
)
from src.v9.memory_bank import build_memory_bank_index, build_memory_bank_windows, query_memory_bank, train_memory_bank_encoder


class V9RuntimeTests(unittest.TestCase):
    def test_persona_calibration_roundtrip(self) -> None:
        personas = default_personas()
        state = default_persona_state(personas)
        updated = update_persona_calibration(
            state,
            {"retail": 0.8, "institutional": -0.3, "algo": 0.2, "whale": 0.5, "noise": -0.1},
            actual_direction=1.0,
            timestamp="2026-04-04T12:00:00Z",
        )
        self.assertAlmostEqual(sum(updated.capital_weights.values()), 1.0, places=5)
        self.assertGreater(updated.accuracy_ema["retail"], state.accuracy_ema["retail"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "persona.parquet"
            from src.v9.persona_calibration import append_persona_calibration_history

            append_persona_calibration_history(path, updated, actual_direction=1.0)
            loaded = load_latest_persona_state(path, personas)
            self.assertIn("retail", loaded.capital_weights)

    def test_contradiction_classifier_rules(self) -> None:
        assessment = classify_contradiction(prob_5m=0.35, prob_15m=0.7, prob_30m=0.72, conf_5m=0.6, conf_15m=0.7, conf_30m=0.8)
        self.assertEqual(assessment.contradiction_type.value, "short_term_contrary")

    def test_regret_gate_scores(self) -> None:
        probabilities = np.asarray([0.55, 0.8], dtype=np.float32)
        context = np.asarray([[0.2] * 22, [0.8] * 22], dtype=np.float32)
        projected = projected_move_from_context(probabilities, context_features=context)
        scores = regret_gate_scores(probabilities, context_features=context)
        self.assertEqual(projected.shape, (2,))
        self.assertEqual(scores.shape, (2,))
        self.assertGreater(float(scores[1]), float(scores[0]))

    def test_memory_bank_query(self) -> None:
        features = np.random.default_rng(42).normal(size=(220, 8)).astype(np.float32)
        targets = (features[:, 0] > 0).astype(np.int64)
        windows, labels = build_memory_bank_windows(features, targets, window_size=10, sample_stride=5, max_samples=20)
        model, _ = train_memory_bank_encoder(windows, labels, epochs=1, batch_size=8, device="cpu")
        index = build_memory_bank_index(model, windows, device="cpu")
        result = query_memory_bank(model, {"embeddings": index, "labels": labels, "window_size": np.asarray([10])}, windows[0].reshape(10, -1), device="cpu")
        self.assertGreaterEqual(result.analog_confidence, 0.0)
        self.assertLessEqual(result.bullish_probability, 1.0)

    def test_persona_signal_aggregation(self) -> None:
        signals = aggregate_persona_signals(
            {"retail": [{"bot_id": "a", "weight": 1.0}, {"bot_id": "b", "weight": 0.5}]},
            [{"bot_id": "a", "bullish_probability": 0.7}, {"bot_id": "b", "bullish_probability": 0.3}],
        )
        self.assertIn("retail", signals)


if __name__ == "__main__":
    unittest.main()
