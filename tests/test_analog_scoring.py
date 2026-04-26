import shutil
import unittest
from pathlib import Path

import numpy as np

from src.mcts.analog import HistoricalAnalogScorer


class AnalogScoringTests(unittest.TestCase):
    def test_historical_analog_scorer_prefers_similar_rows(self):
        root = Path('tests/.tmp/analog_scoring')
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            features = np.zeros((6, 100), dtype=np.float32)
            targets = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.float32)
            timestamps = np.asarray(
                [
                    "2019-01-01T00:00:00",
                    "2019-01-01T00:05:00",
                    "2019-01-01T00:10:00",
                    "2020-01-01T00:00:00",
                    "2020-01-01T00:05:00",
                    "2020-01-01T00:10:00",
                ]
            )
            features[:3, 0] = [0.5, 0.45, 0.55]
            features[3:, 0] = [-0.5, -0.45, -0.55]
            features[:3, 4] = [70, 68, 72]
            features[3:, 4] = [30, 32, 28]
            features[:, 29] = 1.0
            np.save(root / "features.npy", features)
            np.save(root / "targets.npy", targets)
            np.save(root / "timestamps.npy", timestamps)

            scorer = HistoricalAnalogScorer(
                features_path=root / "features.npy",
                targets_path=root / "targets.npy",
                timestamps_path=root / "timestamps.npy",
                train_years=(2019, 2020),
                sample_stride=1,
                max_samples=100,
                top_k=1,
                window_size=3,
            )
            bullish = scorer.score_window(
                [
                    {"return_1": 0.48, "rsi_14": 69.0, "session_london": 1.0},
                    {"return_1": 0.52, "rsi_14": 70.0, "session_london": 1.0},
                    {"return_1": 0.50, "rsi_14": 71.0, "session_london": 1.0},
                ]
            )
            bearish = scorer.score_window(
                [
                    {"return_1": -0.48, "rsi_14": 31.0, "session_london": 1.0},
                    {"return_1": -0.52, "rsi_14": 30.0, "session_london": 1.0},
                    {"return_1": -0.50, "rsi_14": 29.0, "session_london": 1.0},
                ]
            )
            self.assertGreater(bullish.bullish_probability, 0.5)
            self.assertLess(bearish.bullish_probability, 0.5)
            self.assertGreaterEqual(bullish.confidence, 0.0)
            self.assertLessEqual(bullish.confidence, 1.0)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

