from __future__ import annotations

import unittest

import torch

from src.v21.xlstm_backbone import NexusXLSTM, VariableSelectionNetwork, sLSTMBlock


class V21XLSTMTests(unittest.TestCase):
    def test_vsn_preserves_shape(self) -> None:
        model = VariableSelectionNetwork(n_features=8, d_model=16, n_regimes=6)
        x = torch.randn(2, 5, 8)
        regime_ids = torch.zeros(2, 5, dtype=torch.long)
        weighted, weights = model(x, regime_ids)
        self.assertEqual(tuple(weighted.shape), (2, 5, 8))
        self.assertEqual(tuple(weights.shape), (2, 5, 8))

    def test_slstm_block_returns_sequence(self) -> None:
        block = sLSTMBlock(d_model=16, d_state=12)
        x = torch.randn(2, 7, 16)
        y = block(x)
        self.assertEqual(tuple(y.shape), (2, 7, 16))

    def test_nexus_xlstm_outputs_all_heads(self) -> None:
        model = NexusXLSTM(n_features=12, d_model=32, n_layers=2, n_regimes=6)
        x = torch.randn(4, 10, 12)
        regime_ids = torch.zeros(4, 10, dtype=torch.long)
        outputs = model(x, regime_ids)
        self.assertEqual(tuple(outputs["vol_env"].shape), (4, 3))
        self.assertEqual(tuple(outputs["regime"].shape), (4, 6))
        self.assertEqual(tuple(outputs["range"].shape), (4, 3))
        self.assertEqual(tuple(outputs["vsn_weights"].shape), (4, 12))


if __name__ == "__main__":
    unittest.main()
