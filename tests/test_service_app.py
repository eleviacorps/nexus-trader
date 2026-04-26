import unittest

from src.service.app import classify_probability, load_model_manifest, validate_sequence_shape


class ServiceAppTests(unittest.TestCase):
    def test_validate_sequence_shape_accepts_expected_dimensions(self):
        sequence = [[0.0] * 100 for _ in range(120)]
        validate_sequence_shape(sequence, sequence_len=120, feature_dim=100)

    def test_validate_sequence_shape_rejects_short_sequence(self):
        sequence = [[0.0] * 100 for _ in range(119)]
        with self.assertRaises(ValueError):
            validate_sequence_shape(sequence, sequence_len=120, feature_dim=100)

    def test_validate_sequence_shape_rejects_wrong_feature_width(self):
        sequence = [[0.0] * 100 for _ in range(120)]
        sequence[12] = [0.0] * 99
        with self.assertRaises(ValueError):
            validate_sequence_shape(sequence, sequence_len=120, feature_dim=100)

    def test_classify_probability_uses_threshold(self):
        self.assertEqual(classify_probability(0.61, 0.6), "bullish")
        self.assertEqual(classify_probability(0.59, 0.6), "bearish")

    def test_manifest_fallback_is_present(self):
        manifest = load_model_manifest()
        self.assertIn("sequence_len", manifest)
        self.assertIn("feature_dim", manifest)
        self.assertIn("classification_threshold", manifest)


if __name__ == "__main__":
    unittest.main()
