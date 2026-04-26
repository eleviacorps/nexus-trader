import unittest

from src.v16.confidence_tier import ConfidenceTier
from src.v16.sel import sel_lot_size, should_execute
from src.v16.sqt import SimulationQualityTracker


class V16SELTests(unittest.TestCase):
    def test_precision_only_executes_high_or_better(self) -> None:
        tracker = SimulationQualityTracker()
        execute, reason = should_execute(ConfidenceTier.MODERATE, "precision", tracker)
        self.assertFalse(execute)
        self.assertIn("precision_mode_below_threshold", reason)

    def test_lot_size_scales_with_equity(self) -> None:
        lower = sel_lot_size(1000.0, ConfidenceTier.HIGH, "GOOD", mode="frequency")
        higher = sel_lot_size(1100.0, ConfidenceTier.HIGH, "GOOD", mode="frequency")
        self.assertGreater(higher, lower)


if __name__ == "__main__":
    unittest.main()
