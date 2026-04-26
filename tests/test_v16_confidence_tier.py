import unittest

from src.v16.confidence_tier import ConfidenceTier, classify_confidence


class V16ConfidenceTierTests(unittest.TestCase):
    def test_very_high_requires_tight_cone_and_strong_scores(self) -> None:
        tier = classify_confidence(0.74, 0.88, 4.5, 0.64)
        self.assertEqual(tier, ConfidenceTier.VERY_HIGH)

    def test_uncertain_when_cabr_near_random(self) -> None:
        tier = classify_confidence(0.49, 0.82, 3.0, 0.61)
        self.assertEqual(tier, ConfidenceTier.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
