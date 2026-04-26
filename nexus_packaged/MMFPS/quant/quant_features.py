"""Quantitative feature extractor for MMFPS."""

from __future__ import annotations

import sys
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import torch
import torch.nn as nn
from torch import Tensor


class QuantFeatureExtractor(nn.Module):
    def __init__(self, path_dim: int = 20, embed_dim: int = 32):
        super().__init__()

        self.path_dim = path_dim
        self.embed_dim = embed_dim

        self.extractor = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, paths: Tensor) -> Tensor:
        B = paths.shape[0]

        returns = (paths[:, :, -1] - paths[:, :, 0]) / (paths[:, :, 0] + 1e-8)

        mean_ret = returns.mean(dim=1)
        std_ret = returns.std(dim=1)
        max_ret = returns.max(dim=1)[0]
        min_ret = returns.min(dim=1)[0]

        mean_path_prices = paths.mean(dim=1).mean(dim=1)
        path_spread = (paths.max(dim=2)[0].max(dim=1)[0] - paths.min(dim=2)[0].min(dim=1)[0])

        stats = torch.stack([
            mean_ret, std_ret, max_ret, min_ret,
            mean_path_prices, path_spread,
        ], dim=-1)

        return self.extractor(stats)