import unittest

from src.v16.sqt import SimulationQualityTracker


class V16SQTTests(unittest.TestCase):
    def test_label_moves_to_hot_when_accuracy_is_high(self) -> None:
        tracker = SimulationQualityTracker(window_bars=12, cold_threshold=9)
        for _ in range(10):
            tracker.record("BUY", "BUY", "high")
        tracker.record("SELL", "BUY", "low")
        tracker.record("BUY", "BUY", "high")
        self.assertEqual(tracker.label, "HOT")

    def test_should_pause_on_cold_streak(self) -> None:
        tracker = SimulationQualityTracker(window_bars=24, cold_threshold=18)
        for _ in range(18):
            tracker.record("BUY", "SELL", "low")
        for _ in range(6):
            tracker.record("BUY", "BUY", "high")
        self.assertTrue(tracker.should_pause())


if __name__ == "__main__":
    unittest.main()
