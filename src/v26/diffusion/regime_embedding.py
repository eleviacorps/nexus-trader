"""Regime Embedding Module for V26 Phase 1.

Converts regime probability vectors into dense embeddings that can be
injected into the diffusion model alongside temporal conditioning.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class RegimeEmbedding(nn.Module):
    """Embed regime probability vectors into dense representations.

    Input: (B, 9) regime probability vector (9 regimes)
    Output: (B, 16) regime embedding

    Architecture:
        - Linear projection: 9 -> 16
        - Optional learned embedding table (default: disabled)
        - LayerNorm + GELU activation

    Args:
        num_regimes: Number of regime classes (default 9).
        embed_dim: Output embedding dimension (default 16).
        use_learned_embedding: If True, add a learned regime embedding table
            that gets combined with the linear projection output.
        dropout: Dropout rate on the embedding output.
    """

    def __init__(
        self,
        num_regimes: int = 9,
        embed_dim: int = 16,
        use_learned_embedding: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_regimes = num_regimes
        self.embed_dim = embed_dim
        self.use_learned_embedding = use_learned_embedding

        # Linear projection from regime probs to embedding space
        self.linear_proj = nn.Linear(num_regimes, embed_dim)

        # Optional learned regime embedding table
        if use_learned_embedding:
            self.regime_embed_table = nn.Embedding(num_regimes, embed_dim)
        else:
            self.register_buffer("regime_embed_table", None)

        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters with small random values."""
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.zeros_(self.linear_proj.bias)

    def forward(self, regime_probs: Tensor) -> Tensor:
        """Convert regime probability vector to regime embedding.

        Args:
            regime_probs: (B, num_regimes) or (num_regimes,) regime probability
                vector. Should sum to 1 across the regime dimension.

        Returns:
            (B, embed_dim) or (embed_dim,) regime embedding.
        """
        # Handle single vector input
        squeeze_output = False
        if regime_probs.dim() == 1:
            regime_probs = regime_probs.unsqueeze(0)
            squeeze_output = True

        # Linear projection: (B, 9) -> (B, 16)
        emb = self.linear_proj(regime_probs)

        # Optional learned embedding combination
        if self.use_learned_embedding and self.regime_embed_table is not None:
            # Weighted combination using regime probabilities
            # regime_probs: (B, 9), regime_embed_table: (9, 16)
            learned_emb = torch.matmul(regime_probs, self.regime_embed_table.weight)
            emb = emb + learned_emb

        emb = self.norm(emb)
        emb = self.activation(emb)
        emb = self.dropout(emb)

        if squeeze_output:
            emb = emb.squeeze(0)

        return emb

    def forward_from_class(self, regime_class: Tensor) -> Tensor:
        """Alternative forward from regime class indices (for testing/training).

        Args:
            regime_class: (B,) or scalar int regime class indices in [0, num_regimes).

        Returns:
            (B, embed_dim) or (embed_dim,) regime embedding.
        """
        squeeze_output = False
        if regime_class.dim() == 0:
            regime_class = regime_class.unsqueeze(0)
            squeeze_output = True

        # Get embedding from lookup table
        if self.regime_embed_table is not None:
            emb = self.regime_embed_table(regime_class)
        else:
            # Convert class to one-hot and use linear projection
            one_hot = torch.zeros(
                regime_class.shape[0], self.num_regimes,
                device=regime_class.device, dtype=torch.float32
            )
            one_hot.scatter_(1, regime_class.unsqueeze(1), 1.0)
            emb = self.linear_proj(one_hot)

        emb = self.norm(emb)
        emb = self.activation(emb)
        emb = self.dropout(emb)

        if squeeze_output:
            emb = emb.squeeze(0)

        return emb
