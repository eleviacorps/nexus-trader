import unittest

from src.v18.mfg_beliefs import MFGBeliefState, PersonaBelief


class V18MfgTests(unittest.TestCase):
    def test_persona_belief_updates_expected_drift(self) -> None:
        belief = PersonaBelief(name="retail")
        before = belief.expected_drift
        belief.update(0.0015)
        self.assertNotEqual(before, belief.expected_drift)
        self.assertAlmostEqual(float(belief.belief.sum()), 1.0, places=6)

    def test_mfg_summary_reports_disagreement(self) -> None:
        state = MFGBeliefState()
        bars = [
            {"close": 2300.0},
            {"close": 2302.0},
            {"close": 2298.0},
            {"close": 2305.0},
            {"close": 2307.0},
        ]
        state.update_from_bars(bars)
        summary = state.summary()
        self.assertIn("disagreement", summary)
        self.assertIn("personas", summary)
        self.assertGreaterEqual(summary["disagreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
