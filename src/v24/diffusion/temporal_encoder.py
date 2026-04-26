"""Hybrid GRU + cross-attention temporal encoder for Phase 0.5.

Architecture:
  2-layer GRU processes past context (B, T_past, in_features) →
    - Full hidden sequence (B, T_past, d_gru) for temporal cross-attention
    - Final hidden state (B, d_gru) for FiLM conditioning in every ResBlock

The GRU hidden sequence provides temporal context at every spatial position
via cross-attention in decoder levels. The final hidden state provides a
summary embedding for FiLM modulation in every ResBlock.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TemporalEncoder(nn.Module):
    """2-layer GRU temporal encoder with projection for FiLM + cross-attention.

    Args:
        in_features: Input feature dimension per timestep (default 144 for 6M fused).
        d_gru: GRU hidden dimension (default 256).
        num_layers: Number of GRU layers (default 2).
        film_dim: Output dimension for FiLM conditioning (matches U-Net time_dim).
        dropout: Dropout between GRU layers (default 0.1).
    """

    def __init__(
        self,
        in_features: int = 144,
        d_gru: int = 256,
        num_layers: int = 2,
        film_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.d_gru = d_gru
        self.film_dim = film_dim

        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=d_gru,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.film_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_gru, film_dim),
            nn.SiLU(),
            nn.Linear(film_dim, film_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, past_context: Tensor, hidden: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode past context into temporal embeddings.

        Args:
            past_context: (B, T_past, in_features) — past bar features.
            hidden: Optional initial GRU hidden state (num_layers, B, d_gru).

        Returns:
            hidden_seq: (B, T_past, d_gru) — full hidden sequence for cross-attention.
            film_emb: (B, film_dim) — projected final hidden for FiLM conditioning.
            final_hidden: (B, d_gru) — raw final hidden state.
        """
        gru_out, gru_hidden = self.gru(past_context, hidden)

        final_hidden = gru_hidden[-1]

        film_emb = self.film_proj(final_hidden)

        return gru_out, film_emb, final_hidden
