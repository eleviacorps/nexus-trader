from __future__ import annotations

import unittest

import torch

from src.v21.offline_rl import CQLBatch, ConservativeQNetwork, conservative_q_loss
from src.v21.rl_executor_v21 import V21HierarchicalExecutor


class V21RLTests(unittest.TestCase):
    def test_hierarchical_executor_returns_valid_action(self) -> None:
        executor = V21HierarchicalExecutor()
        decision = executor.decide(
            regime_probs=[0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
            direction_signal=0.3,
            confidence=0.7,
            macro_state=[0.2] * 8,
            volatility=0.1,
            kelly_fraction=0.05,
            branch_rewards=[0.1, 0.2, -0.05],
        )
        self.assertIn(decision.action, {"BUY", "SELL", "HOLD"})
        self.assertEqual(len(decision.hyper_weights), 6)

    def test_conservative_q_loss_is_finite(self) -> None:
        q_network = ConservativeQNetwork(state_dim=5, action_dim=3)
        target_network = ConservativeQNetwork(state_dim=5, action_dim=3)
        batch = CQLBatch(
            states=torch.randn(8, 5),
            actions=torch.randint(0, 3, (8,)),
            rewards=torch.randn(8),
            next_states=torch.randn(8, 5),
            dones=torch.randint(0, 2, (8,), dtype=torch.float32),
        )
        loss = conservative_q_loss(q_network, target_network, batch)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
