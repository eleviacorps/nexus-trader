"""Chunk ranker for scoring and pruning branches."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class ChunkRanker(nn.Module):
    """Ranks generated chunks based on multiple criteria."""

    def __init__(
        self,
        num_regimes: int = 9,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_regimes = num_regimes
        self.device = torch.device(device)

    def score_chunks(
        self,
        chunks: Tensor,
        regime_probs: Tensor,
        previous_context: Optional[Tensor] = None,
    ) -> Tensor:
        """Score multiple chunk candidates."""
        scores = torch.zeros(chunks.shape[0], device=self.device)

        for i in range(chunks.shape[0]):
            chunk = chunks[i]
            realism = self._realism_score(chunk)
            regime_consistency = self._regime_consistency(chunk, regime_probs)
            boundary = 0.0
            if previous_context is not None:
                boundary = self._boundary_continuity(
                    previous_context[:, -10:] if previous_context.dim() == 3 else previous_context[-10:],
                    chunk[:10]
                )
            scores[i] = realism + 0.3 * regime_consistency + 0.2 * boundary

        return scores

    def _realism_score(self, chunk: Tensor) -> float:
        chunk_std = chunk.std().item()
        chunk_mean = chunk.abs().mean().item()
        if chunk_std < 0.01:
            realism = chunk_std * 10
        elif chunk_std > 0.5:
            realism = 0.5 / chunk_std
        else:
            realism = 1.0 - abs(chunk_std - 0.1)
        return realism

    def _regime_consistency(self, chunk: Tensor, regime_probs: Tensor) -> float:
        return 0.5

    def _boundary_continuity(self, context_end: Tensor, chunk_start: Tensor) -> float:
        if context_end.shape != chunk_start.shape:
            return 0.0
        return torch.nn.functional.cosine_similarity(
            context_end.flatten().unsqueeze(0),
            chunk_start.flatten().unsqueeze(0)
        ).item()

    def prune_branches(
        self,
        chunks: Tensor,
        scores: Tensor,
        keep_top_k: int,
    ) -> tuple:
        if chunks.shape[0] <= keep_top_k:
            return chunks, scores
        _, top_idx = torch.topk(scores, keep_top_k)
        return chunks[top_idx], scores[top_idx]


def create_chunk_ranker(device="cpu"):
    return ChunkRanker(device=device)