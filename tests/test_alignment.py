import unittest
from datetime import datetime, timedelta

from src.utils.alignment import concatenate_feature_blocks, forward_fill_embeddings


class AlignmentTests(unittest.TestCase):
    def test_forward_fill_respects_limit(self):
        start = datetime(2026, 1, 1, 12, 0, 0)
        bars = [start + timedelta(minutes=offset) for offset in range(4)]
        events = [start + timedelta(minutes=1)]
        vectors = [[1.0, 2.0]]

        aligned = forward_fill_embeddings(bars, events, vectors, dims=2, fill_limit_minutes=1)
        self.assertEqual(aligned[0], [0.0, 0.0])
        self.assertEqual(aligned[1], [1.0, 2.0])
        self.assertEqual(aligned[2], [1.0, 2.0])
        self.assertEqual(aligned[3], [0.0, 0.0])

    def test_feature_block_concatenation(self):
        fused = concatenate_feature_blocks([[1.0, 2.0]], [[3.0]], [[4.0, 5.0]])
        self.assertEqual(fused, [[1.0, 2.0, 3.0, 4.0, 5.0]])


if __name__ == "__main__":
    unittest.main()
