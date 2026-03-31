import unittest

import numpy as np

from src.pipeline.fusion import build_fused_feature_matrix, build_sequence_tensor, normalize_binary_targets


class FusionPipelineTests(unittest.TestCase):
    def test_build_fused_feature_matrix_shape(self):
        price = np.zeros((3, 36), dtype=np.float32)
        news = np.ones((3, 32), dtype=np.float32)
        crowd = np.full((3, 32), 2.0, dtype=np.float32)
        fused = build_fused_feature_matrix(price, news, crowd)
        self.assertEqual(fused.shape, (3, 100))

    def test_normalize_binary_targets(self):
        targets = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        normalized = normalize_binary_targets(targets)
        np.testing.assert_array_equal(normalized, np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

    def test_build_sequence_tensor_shape(self):
        features = np.arange(5 * 100, dtype=np.float32).reshape(5, 100)
        targets = np.array([0, 1, 0, 1, 1], dtype=np.float32)
        tensor, seq_targets = build_sequence_tensor(features, targets, sequence_len=3)
        self.assertEqual(tensor.shape, (3, 3, 100))
        np.testing.assert_array_equal(seq_targets, np.array([0, 1, 1], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
