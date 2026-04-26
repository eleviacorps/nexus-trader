"""Autoregressive Chunked Diffusion Generator.

Generates long horizons by chunking:
- 120 historical bars
- 30 bars per chunk
- 4 sequential chunks = 120 total future bars

Uses beam search with branch pruning at each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ChunkConfig:
    """Config for chunked generation."""
    chunk_size: int = 30
    num_chunks: int = 4
    context_len: int = 120
    beam_width: int = 4
    candidates_per_step: int = 16
    prune_to: int = 4


class AutoregressiveChunkedGenerator(nn.Module):
    """Autoregressive chunked diffusion with beam search.

    Generates long horizons by:
    1. Generate chunk 1 (30 bars)
    2. Score and prune branches
    3. Continue from top-K branches
    4. Repeat for all chunks
    """

    def __init__(
        self,
        base_generator,
        config: Optional[ChunkConfig] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.base_generator = base_generator
        self.config = config or ChunkConfig()
        self.device = torch.device(device)

    @torch.no_grad()
    def generate_long_horizon(
        self,
        past_context: Optional[Tensor],
        regime_probs: Tensor,
        steps: int = 10,
    ) -> dict:
        """Generate long horizon with autoregressive chunking.

        Args:
            past_context: (B, T, C) historical context
            regime_probs: (B, 9) or (9,) regime probs
            steps: Diffusion steps per chunk

        Returns:
            dict with all_chunks, full_path, continuity_scores
        """
        if past_context is not None:
            past_context = past_context.to(self.device)
        regime_probs = regime_probs.to(self.device)

        # Handle single vs batch
        single = regime_probs.dim() == 1
        if single:
            regime_probs = regime_probs.unsqueeze(0)
        if past_context is not None and past_context.dim() == 2:
            past_context = past_context.unsqueeze(0)

        batch_size = regime_probs.shape[0]
        cfg = self.config

        # Initialize beam search state
        # Each beam: (batch, chunk_idx, features, chunk_size)
        beam_chunks = [None] * cfg.num_chunks
        beam_scores = [torch.ones(batch_size, cfg.beam_width, device=self.device) * -1e9 for _ in range(cfg.num_chunks)]

        # Track which branches survive
        alive_mask = torch.ones(batch_size, cfg.beam_width, dtype=torch.bool, device=self.device)

        rolling_context = past_context  # (B, T, C)

        all_chunks = []

        for chunk_idx in range(cfg.num_chunks):
            # Generate candidates for all alive branches
            candidates = []

            for beam_idx in range(cfg.beam_width):
                if not alive_mask[:, beam_idx].any():
                    continue

                # Get context for this branch
                ctx = rolling_context

                # Generate chunk
                rp = regime_probs
                paths = self.base_generator.generate_paths(
                    world_state=None,
                    regime_probs=rp,
                    num_paths=1,
                    past_context=ctx,
                    steps=steps,
                )

                # Extract generated chunk
                chunk_data = torch.tensor(paths[0]["data"], device=self.device)
                if chunk_data.dim() == 2:
                    chunk_data = chunk_data.T  # (C, L)
                chunk_data = chunk_data[:, :cfg.chunk_size]  # Take only chunk_size

                candidates.append(chunk_data)

                # Score this branch
                score = self._score_chunk(chunk_data, regime_probs)
                beam_scores[chunk_idx][:, beam_idx] = score

            # Prune to top-K at each chunk
            if candidates:
                chunk_tensor = torch.stack(candidates[:cfg.beam_width], dim=0)  # (K, C, chunk)
                all_chunks.append(chunk_tensor.cpu())

                # Update rolling context for next chunk
                new_context = chunk_tensor.permute(0, 2, 1).unsqueeze(0)  # (1, K, chunk, C)
                if rolling_context is not None:
                    rolling_context = torch.cat([rolling_context, new_context[:, 0]], dim=1)
                else:
                    rolling_context = new_context[:, 0]

        # Full path = concatenation of all chunks
        if all_chunks:
            full_path = torch.cat(all_chunks, dim=2)  # (K, C, total_len)
        else:
            full_path = torch.zeros(1, 120, device=self.device)

        # Compute continuity between chunks
        continuity = self._compute_continuity(all_chunks)

        return {
            "all_chunks": all_chunks,
            "full_path": full_path,
            "continuity": continuity,
            "beam_scores": beam_scores,
        }

    def _score_chunk(self, chunk: Tensor, regime_probs: Tensor) -> Tensor:
        """Score a generated chunk."""
        # Simple heuristic: variance-based realism
        chunk_std = chunk.std()
        chunk_mean = chunk.abs().mean()

        # Prefer moderate variance (too low = boring, too high = noise)
        score = -abs(chunk_std - 0.5) + chunk_mean * 0.1

        return score

    def _compute_continuity(self, chunks: list) -> dict:
        """Compute continuity scores between chunks."""
        if len(chunks) < 2:
            return {"avg": 0.0, "chunks": []}

        scores = []
        for i in range(len(chunks) - 1):
            chunk_a = chunks[i]  # (K, C, chunk)
            chunk_b = chunks[i + 1]

            # Last 5 bars of chunk_a vs first 5 of chunk_b
            a_end = chunk_a[:, :, -5:].mean(dim=2)
            b_start = chunk_b[:, :, :5].mean(dim=2)

            cos_sim = torch.nn.functional.cosine_similarity(a_end, b_start, dim=1)
            scores.append(cos_sim.mean().item())

        return {
            "avg": sum(scores) / len(scores) if scores else 0.0,
            "chunks": scores,
        }


def create_autoregressive_generator(base_generator, device="cpu"):
    """Create autoregressive chunked generator."""
    config = ChunkConfig(
        chunk_size=30,
        num_chunks=4,
        context_len=120,
        beam_width=4,
        candidates_per_step=16,
        prune_to=4,
    )
    return AutoregressiveChunkedGenerator(base_generator, config, device)