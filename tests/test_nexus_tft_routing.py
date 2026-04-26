from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.models.nexus_tft import NexusTFT, NexusTFTConfig


@unittest.skipIf(torch is None, "PyTorch is required for routing tests.")
class NexusTFTRoutingTests(unittest.TestCase):
    def test_forward_returns_regime_diagnostics(self):
        model = NexusTFT(
            NexusTFTConfig(
                input_dim=8,
                hidden_dim=16,
                lstm_layers=1,
                dropout=0.0,
                output_dim=3,
                regime_count=4,
                router_hidden_dim=12,
            )
        )
        batch = torch.rand(2, 6, 8)
        prediction, diagnostics = model(batch, return_diagnostics=True)

        self.assertEqual(tuple(prediction.shape), (2, 3))
        self.assertIn("regime_probabilities", diagnostics)
        regime_probabilities = diagnostics["regime_probabilities"]
        self.assertEqual(tuple(regime_probabilities.shape), (2, 4))
        sums = regime_probabilities.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        self.assertTrue(torch.all((prediction >= 0.0) & (prediction <= 1.0)))

    def test_forward_can_return_importance_and_diagnostics(self):
        model = NexusTFT(
            NexusTFTConfig(
                input_dim=10,
                hidden_dim=20,
                lstm_layers=1,
                dropout=0.0,
                output_dim=1,
                regime_count=4,
                router_hidden_dim=16,
            )
        )
        batch = torch.rand(3, 4, 10)
        prediction, importance, diagnostics = model(batch, return_feature_importance=True, return_diagnostics=True)

        self.assertEqual(tuple(prediction.shape), (3,))
        self.assertEqual(tuple(importance.shape), (3, 10))
        self.assertEqual(tuple(diagnostics["regime_probabilities"].shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
