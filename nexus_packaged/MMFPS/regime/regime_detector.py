"""Probabilistic regime detection for MMFPS.

Regimes are NOT deterministic. Output is a distribution over regime types.
"""

from __future__ import annotations

import sys
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple


class RegimeState(NamedTuple):
    prob_trend_up: Tensor
    prob_chop: Tensor
    prob_reversal: Tensor
    regime_embedding: Tensor
    regime_type: Tensor


class RegimeDetector(nn.Module):
    """Probabilistic regime detector using context features.

    Outputs P(regime | context) as a categorical distribution, not a hard label.
    Also produces a regime embedding for conditioning the generator.
    """

    TREND_UP = 0
    CHOP = 1
    REVERSAL = 2
    NUM_REGIMES = 3

    def __init__(self, feature_dim: int = 144, embed_dim: int = 32):
        super().__init__()

        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.regime_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, self.NUM_REGIMES),
        )

        self.regime_embeddings = nn.Embedding(self.NUM_REGIMES, embed_dim)

    def forward(self, context: Tensor) -> RegimeState:
        enc = self.encoder(context)

        logits = self.regime_head(enc)

        prob = torch.softmax(logits, dim=-1)
        regime_type = prob.argmax(dim=-1)

        regime_embedding = self.regime_embeddings(regime_type)

        return RegimeState(
            prob_trend_up=prob[:, 0],
            prob_chop=prob[:, 1],
            prob_reversal=prob[:, 2],
            regime_embedding=regime_embedding,
            regime_type=regime_type,
        )

    def sample_regime(self, context: Tensor) -> RegimeState:
        enc = self.encoder(context)
        logits = self.regime_head(enc)

        prob = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(prob)
        regime_type = dist.sample()

        regime_embedding = self.regime_embeddings(regime_type)

        return RegimeState(
            prob_trend_up=prob[:, 0],
            prob_chop=prob[:, 1],
            prob_reversal=prob[:, 2],
            regime_embedding=regime_embedding,
            regime_type=regime_type,
        )