import unittest

from src.mcts.cone import build_probability_cone
from src.mcts.reverse_collapse import reverse_collapse
from src.mcts.tree import SimulationNode, assert_leaf_count, iter_leaves
from src.simulation.personas import default_personas


class MctsTests(unittest.TestCase):
    def test_exact_leaf_count(self):
        from src.mcts.tree import expand_binary_tree

        personas = default_personas()
        row = {
            "close": 3000.0,
            "atr_14": 10.0,
            "ema_cross": 1.0,
            "rsi_14": 65.0,
            "rsi_7": 70.0,
            "macd_hist": 1.0,
            "bb_pct": 0.8,
            "body_pct": 0.7,
            "dist_to_high": 0.5,
            "dist_to_low": 2.0,
            "hh": 1.0,
            "ll": 0.0,
        }
        root = expand_binary_tree(row, personas, max_depth=5)
        assert_leaf_count(root, expected=32)
        self.assertEqual(len(iter_leaves(root)), 32)
        self.assertTrue(all(len(leaf.path_prices) == 5 for leaf in iter_leaves(root)))

    def test_branch_rows_roll_forward(self):
        from src.mcts.tree import expand_binary_tree

        personas = default_personas()
        row = {
            "close": 3000.0,
            "open": 2998.0,
            "high": 3002.0,
            "low": 2995.0,
            "atr_14": 10.0,
            "ema_cross": -0.5,
            "rsi_14": 41.0,
            "rsi_7": 38.0,
            "macd_hist": -0.6,
            "macd": -0.4,
            "macd_sig": -0.2,
            "bb_pct": 0.22,
            "body_pct": 0.55,
            "dist_to_high": 1.4,
            "dist_to_low": 0.6,
            "hh": 0.0,
            "ll": 1.0,
            "macro_bias": 0.25,
            "news_bias": 0.15,
            "crowd_bias": -0.10,
            "crowd_extreme": 0.4,
            "llm_market_bias": 0.2,
        }
        root = expand_binary_tree(row, personas, max_depth=3)
        leaves = iter_leaves(root)
        self.assertEqual(len(leaves), 8)
        self.assertTrue(all("close" in leaf.row_snapshot for leaf in leaves))
        self.assertTrue(any(abs(leaf.row_snapshot["close"] - row["close"]) > 0.01 for leaf in leaves))

    def test_reverse_collapse_behavior(self):
        bullish = [SimulationNode(seed=i, depth=5, probability_weight=1.0, dominant_driver="buying") for i in range(4)]
        for leaf in bullish:
            leaf.state = type("State", (), {"directional_bias": 0.9})()
        bearish = [SimulationNode(seed=i + 10, depth=5, probability_weight=1.0, dominant_driver="selling") for i in range(4)]
        for leaf in bearish:
            leaf.state = type("State", (), {"directional_bias": -0.9})()

        agree = reverse_collapse(bullish)
        disagree = reverse_collapse(bullish + bearish)
        self.assertGreater(agree.consensus_score, disagree.consensus_score)
        self.assertLess(agree.uncertainty_width, disagree.uncertainty_width)

    def test_probability_cone_widens_with_uncertainty(self):
        narrow = build_probability_cone(type("Collapse", (), {"mean_probability": 0.6, "uncertainty_width": 0.1, "consensus_score": 0.8})())
        wide = build_probability_cone(type("Collapse", (), {"mean_probability": 0.6, "uncertainty_width": 0.4, "consensus_score": 0.3})())
        self.assertLess(narrow.points[-1].upper - narrow.points[-1].lower, wide.points[-1].upper - wide.points[-1].lower)


if __name__ == "__main__":
    unittest.main()
