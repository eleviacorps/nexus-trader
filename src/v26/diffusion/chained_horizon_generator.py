"""Phase 2B: Chained-context multi-horizon generator.

Simple approach:
1. Generate short horizon from past context
2. Append last N bars of short to context for medium generation
3. Append last N bars of medium to context for long generation

No summary encoders needed - use generated paths as context for next horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ChainedHorizonConfig:
    """Config for chained horizon generation."""
    short_len: int = 120
    medium_len: int = 120
    long_len: int = 120
    context_overlap: int = 30  # Bars to carry forward
    num_paths: int = 1
    sampling_steps: int = 50


class ChainedMultiHorizonGenerator(nn.Module):
    """Multi-horizon generator using chained context.

    Instead of summary encoders, this generator passes the last N bars
    of each generated horizon as context for the next horizon.
    """

    def __init__(
        self,
        base_generator,
        config: Optional[ChainedHorizonConfig] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.base_generator = base_generator
        self.config = config or ChainedHorizonConfig()
        self.device = torch.device(device)

    @torch.no_grad()
    def generate_multi_horizon(
        self,
        world_state,
        past_context: Optional[Tensor],
        regime_probs: Tensor,
        steps: int = 50,
    ) -> dict:
        """Generate multi-horizon with chained context.

        Args:
            world_state: Current market state
            past_context: (B, T, C) temporal context
            regime_probs: (B, 9) or (9,) regime probs
            steps: Diffusion sampling steps

        Returns:
            dict with short, medium, long paths and continuity scores
        """
        if past_context is not None:
            past_context = past_context.to(self.device)
        regime_probs = regime_probs.to(self.device)

        # Handle single vs batch
        single_sample = regime_probs.dim() == 1
        if single_sample:
            regime_probs = regime_probs.unsqueeze(0)
        if past_context is not None and past_context.dim() == 2:
            past_context = past_context.unsqueeze(0)

        batch_size = regime_probs.shape[0]
        overlap = self.config.context_overlap

        # === Short horizon ===
        short_paths = []
        for b in range(batch_size):
            rp = regime_probs[b:b+1]
            pc = past_context[b:b+1] if past_context is not None else None
            paths = self.base_generator.generate_paths(
                world_state=world_state,
                regime_probs=rp,
                num_paths=self.config.num_paths,
                past_context=pc,
                steps=steps,
            )
            # Extract paths
            for p in paths:
                data = torch.tensor(p["data"], device=self.device)
                if data.dim() == 2:
                    data = data.T  # (C, L)
                short_paths.append(data)
        short_tensor = torch.stack(short_paths)  # (B, C, L)

        # === Medium horizon (with short context) ===
        medium_paths = []
        for b in range(batch_size):
            rp = regime_probs[b:b+1]
            # Append last overlap bars from short to context - needs (T, C) format
            short_end = short_tensor[b:b+1, :, -overlap:].permute(0, 2, 1)  # (1, overlap, C)
            if past_context is not None:
                # Concatenate: past + short_tail -> (1, T+overlap, C)
                pc = torch.cat([past_context[b:b+1], short_end], dim=1)
            else:
                pc = short_end
            paths = self.base_generator.generate_paths(
                world_state=world_state,
                regime_probs=rp,
                num_paths=self.config.num_paths,
                past_context=pc,
                steps=steps,
            )
            for p in paths:
                data = torch.tensor(p["data"], device=self.device)
                if data.dim() == 2:
                    data = data.T
                medium_paths.append(data)
        medium_tensor = torch.stack(medium_paths)

        # === Long horizon (with medium context) ===
        long_paths = []
        for b in range(batch_size):
            rp = regime_probs[b:b+1]
            # Append last overlap bars from medium
            med_end = medium_tensor[b:b+1, :, -overlap:].permute(0, 2, 1)
            if past_context is not None:
                pc = torch.cat([past_context[b:b+1], short_tensor[b:b+1, :, -overlap:].permute(0, 2, 1), med_end], dim=1)
            else:
                pc = torch.cat([short_tensor[b:b+1, :, -overlap:].permute(0, 2, 1), med_end], dim=1)
            paths = self.base_generator.generate_paths(
                world_state=world_state,
                regime_probs=rp,
                num_paths=self.config.num_paths,
                past_context=pc,
                steps=steps,
            )
            for p in paths:
                data = torch.tensor(p["data"], device=self.device)
                if data.dim() == 2:
                    data = data.T
                long_paths.append(data)
        long_tensor = torch.stack(long_paths)

        # === Compute continuity scores ===
        # Short -> Medium continuity
        short_end = short_tensor[:, :, -10:].mean(dim=2)
        med_start = medium_tensor[:, :, :10].mean(dim=2)
        short_med_score = torch.nn.functional.cosine_similarity(
            short_end, med_start, dim=1
        ).mean().item()

        # Medium -> Long continuity
        med_end = medium_tensor[:, :, -10:].mean(dim=2)
        long_start = long_tensor[:, :, :10].mean(dim=2)
        med_long_score = torch.nn.functional.cosine_similarity(
            med_end, long_start, dim=1
        ).mean().item()

        avg_continuity = (short_med_score + med_long_score) / 2

        return {
            "short": short_tensor,
            "medium": medium_tensor,
            "long": long_tensor,
            "short_med_continuity": short_med_score,
            "med_long_continuity": med_long_score,
            "avg_continuity": avg_continuity,
        }


def create_chained_generator(base_generator, device="cpu"):
    """Create a chained multi-horizon generator."""
    config = ChainedHorizonConfig(
        short_len=120,
        medium_len=120,
        long_len=120,
        context_overlap=30,
        num_paths=1,
    )
    return ChainedMultiHorizonGenerator(base_generator, config, device)