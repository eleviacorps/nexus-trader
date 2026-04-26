import unittest

from src.models.fused_tft import fuse_feature_rows
from src.models.nexus_tft import expand_feature_matrix_columns, expand_feature_vector


class ModelUtilityTests(unittest.TestCase):
    def test_expand_feature_vector(self):
        expanded = expand_feature_vector([1.0, 2.0, 3.0], new_dim=5, fill_value=9.0)
        self.assertEqual(expanded, [1.0, 2.0, 3.0, 9.0, 9.0])

    def test_expand_feature_matrix_columns(self):
        expanded = expand_feature_matrix_columns([[1.0, 2.0], [3.0, 4.0]], old_input_dim=2, new_input_dim=4, fill_value=7.0)
        self.assertEqual(expanded[0], [1.0, 2.0, 7.0, 7.0])
        self.assertEqual(expanded[1], [3.0, 4.0, 7.0, 7.0])

    def test_fuse_feature_rows_width(self):
        price = [[0.0] * 36]
        news = [[1.0] * 32]
        crowd = [[2.0] * 32]
        fused = fuse_feature_rows(price, news, crowd)
        self.assertEqual(len(fused[0]), 100)


if __name__ == "__main__":
    unittest.main()
