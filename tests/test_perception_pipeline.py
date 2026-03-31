import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from src.pipeline.perception import align_event_matrix, build_crowd_numeric_vectors, load_news_events, reduce_text_embeddings


@unittest.skipIf(pd is None, "pandas is required for perception pipeline tests")
class PerceptionPipelineTests(unittest.TestCase):
    def test_reduce_text_embeddings_shape(self):
        matrix = reduce_text_embeddings(["gold rallies on weak cpi", "dollar rises as yields jump"], output_dim=32)
        self.assertEqual(matrix.shape, (2, 32))

    def test_align_event_matrix_shape(self):
        price_index = pd.date_range("2026-01-01", periods=4, freq="min")
        events = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-01-01 00:01:00", "2026-01-01 00:03:00"]),
                "source_timestamp": pd.to_datetime(["2026-01-01 00:01:00", "2026-01-01 00:03:00"]),
                "news_00": [1.0, 2.0],
                "news_01": [0.5, 0.25],
            }
        )
        aligned, _ = align_event_matrix(price_index, events, ["news_00", "news_01"], tolerance_minutes=5)
        self.assertEqual(aligned.shape, (4, 2))
        self.assertTrue(np.allclose(aligned[0], [0.0, 0.0]))
        self.assertTrue(np.allclose(aligned[1], [1.0, 0.5]))

    def test_load_news_events_reads_simple_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_text(
                "title,seendate,domain,url\nGold rises,20260330093000,example.com,https://example.com/a\n",
                encoding="utf-8",
            )
            frame = load_news_events(Path(tmpdir))
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["source"], "example.com")

    def test_build_crowd_numeric_vectors_width(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
                "value": [20, 25, 40, 60, 80],
                "classification": ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
                "source": ["alt"] * 5,
            }
        )
        matrix = build_crowd_numeric_vectors(frame)
        self.assertEqual(matrix.shape, (5, 32))


if __name__ == "__main__":
    unittest.main()
