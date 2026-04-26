import unittest

from src.v13.lrtd import LiveRegimeTransitionDetector


class LRTDTests(unittest.TestCase):
    def test_transition_suppression(self) -> None:
        detector = LiveRegimeTransitionDetector(transition_threshold=0.30)
        series = [
            ('trending_up', 0.80),
            ('trending_up', 0.74),
            ('trending_down', 0.62),
            ('trending_up', 0.58),
            ('ranging', 0.55),
            ('trending_down', 0.50),
        ]
        for regime, conf in series:
            detector.update(regime, conf)
        self.assertGreater(detector.transition_risk(), 0.0)
        self.assertTrue(detector.should_suppress())


if __name__ == '__main__':
    unittest.main()
