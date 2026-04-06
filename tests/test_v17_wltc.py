import unittest

from src.v17.wltc import WinnerLoserCycle, build_wltc_states


class V17WLTCTests(unittest.TestCase):
    def test_testosterone_rises_on_win_streak(self) -> None:
        cycle = WinnerLoserCycle("retail")
        for value in [0.002, 0.003, 0.004, 0.002]:
            cycle.update(value)
        self.assertGreater(cycle.testosterone_index, 0.0)
        self.assertGreater(cycle.strategy_weight_modifier()["momentum"], 1.0)

    def test_build_states_covers_all_personas(self) -> None:
        bars = [
            {"close": 100.0},
            {"close": 101.0},
            {"close": 102.0},
            {"close": 101.5},
        ]
        states = build_wltc_states(bars)
        self.assertIn("retail", states)
        self.assertIn("noise", states)
        self.assertGreaterEqual(states["retail"].fear_index, 0.0)


if __name__ == "__main__":
    unittest.main()
