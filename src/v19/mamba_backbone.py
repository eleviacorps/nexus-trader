from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


BackboneKind = Literal["mamba2", "xlstm", "hybrid"]


class _FallbackSequenceBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=0.10 if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.norm(hidden[-1])


@dataclass(frozen=True)
class V19BackboneConfig:
    input_dim: int
    hidden_dim: int = 256
    horizons: tuple[int, ...] = (15, 30)
    kind: BackboneKind = "hybrid"
    regime_classes: int = 5


class V19ResearchBackbone(nn.Module):
    """
    Research-only sequence backbone.

    The production stack still uses the existing generator path. This module gives us
    a clean experimental surface for Mamba/xLSTM style sequence research without
    destabilizing the live simulator.
    """

    def __init__(self, config: V19BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _FallbackSequenceBlock(config.input_dim, config.hidden_dim)
        self.direction_head = nn.Linear(config.hidden_dim, len(config.horizons))
        self.volatility_head = nn.Linear(config.hidden_dim, 2)
        self.regime_head = nn.Linear(config.hidden_dim, config.regime_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        return {
            "direction_logits": self.direction_head(hidden),
            "volatility_envelope": self.volatility_head(hidden),
            "regime_logits": self.regime_head(hidden),
        }
