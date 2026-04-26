"""Multi-Horizon Path Stacking for V26 Phase 2.

Hierarchical generation across short/medium/long horizons with
latent conditioning between levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from src.v26.diffusion.regime_embedding import RegimeEmbedding
from src.v26.diffusion.regime_generator import RegimeDiffusionPathGenerator


@dataclass
class HorizonConfig:
    """Configuration for a single horizon."""
    name: str
    seq_len: int
    num_paths: int
    start_bar: int  # Starting bar index in overall path


@dataclass
class GeneratedHorizon:
    """Output from a single horizon generation."""
    name: str
    paths: Tensor  # (num_paths, C, seq_len)
    latent_summary: Tensor  # (num_paths, summary_dim) - for conditioning next horizon
    regime_probs: Tensor  # (num_paths, 9)


@dataclass
class MultiHorizonResult:
    """Complete multi-horizon generation result."""
    short: GeneratedHorizon
    medium: GeneratedHorizon
    long: GeneratedHorizon
    combined_paths: Tensor  # Concatenated across horizons
    horizon_consistency_score: float
    regime_distribution: Dict[str, float]


class HorizonSummaryEncoder(nn.Module):
    """Encodes a generated horizon into a latent summary for conditioning.

    Compresses (num_paths, C, seq_len) → (num_paths, summary_dim)
    using temporal pooling + learned projection.
    """

    def __init__(self, in_channels: int = 144, seq_len: int = 120, summary_dim: int = 64):
        super().__init__()
        self.summary_dim = summary_dim
        self.in_channels = in_channels
        self.seq_len = seq_len

        # Temporal pooling: conv1d + adaptive pooling
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.LayerNorm([64, seq_len]),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.LayerNorm([64, seq_len]),
            nn.GELU(),
        )

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(8)  # (B, 64, 8)

        # Project to summary dimension
        self.projection = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, summary_dim),
            nn.LayerNorm(summary_dim),
        )

    def forward(self, paths: Tensor) -> Tensor:
        """Encode paths to latent summary.

        Args:
            paths: (B, C, L) generated paths

        Returns:
            (B, summary_dim) latent summary
        """
        # Temporal encoding
        x = self.temporal_encoder(paths)  # (B, 64, L)

        # Pool to fixed length
        x = self.pool(x)  # (B, 64, 8)

        # Flatten and project
        x = x.flatten(1)  # (B, 512)
        summary = self.projection(x)  # (B, summary_dim)

        return summary

    def projection_to_path(self, summary: Tensor) -> Tensor:
        """Project latent summary back to path space.

        Args:
            summary: (B, summary_dim) latent summary

        Returns:
            (B, C, L) path reconstruction
        """
        flat = self.path_projection(summary)  # (B, C*L)
        return flat.view(-1, self.in_channels, self.seq_len)


class MultiHorizonGenerator(nn.Module):
    """Hierarchical multi-horizon path generator.

    Generates coherent paths across short/medium/long horizons
    with latent conditioning between levels.
    """

    SHORT_CONFIG = HorizonConfig("short", 120, 14, 0)
    MEDIUM_CONFIG = HorizonConfig("medium", 120, 4, 120)
    LONG_CONFIG = HorizonConfig("long", 120, 2, 240)

    def __init__(
        self,
        base_generator: Optional[RegimeDiffusionPathGenerator] = None,
        summary_dim: int = 64,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.base_generator = base_generator
        self.summary_dim = summary_dim
        self.in_channels = 144
        self.seq_len = 120

        # Horizon summary encoders
        self.short_summary = HorizonSummaryEncoder(self.in_channels, self.seq_len, summary_dim).to(self.device)
        self.medium_summary = HorizonSummaryEncoder(self.in_channels, self.seq_len, summary_dim).to(self.device)

        # Regime embedding for conditioning
        self.regime_embed = RegimeEmbedding(num_regimes=9, embed_dim=16).to(self.device)

        # Extended conditioning projections
        temporal_dim = 256
        regime_dim = 16
        # Medium: temporal + regime + short_summary
        self.medium_condition_proj = nn.Linear(
            temporal_dim + regime_dim + summary_dim, 256
        ).to(self.device)
        # Long: temporal + regime + short_summary + medium_summary
        self.long_condition_proj = nn.Linear(
            temporal_dim + regime_dim + 2 * summary_dim, 256
        ).to(self.device)

        # Projection from summary back to path (for reconstruction loss)
        self.path_projection = nn.Sequential(
            nn.Linear(summary_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.in_channels * self.seq_len),
        ).to(self.device)

    def _get_top_k_regimes(self, regime_probs: Tensor, k: int = 3) -> Tuple[Tensor, Tensor]:
        """Get top-k regime indices and their probabilities.

        Args:
            regime_probs: (9,) or (B, 9) regime probabilities

        Returns:
            (indices, probs) each of shape (k,)
        """
        if regime_probs.dim() == 1:
            regime_probs = regime_probs.unsqueeze(0)

        # Average across batch if needed
        probs = regime_probs.mean(dim=0)  # (9,)
        top_k = torch.topk(probs, k)
        return top_k.indices, top_k.values

    def _distribute_paths_across_regimes(
        self, num_paths: int, regime_probs: Tensor
    ) -> Dict[int, int]:
        """Distribute paths across top-3 regimes: 70/20/10 split.

        Args:
            num_paths: Total number of paths to generate
            regime_probs: (9,) regime probability distribution

        Returns:
            Dict mapping regime_idx -> num_paths for that regime
        """
        top_indices, top_values = self._get_top_k_regimes(regime_probs, k=3)

        # 70/20/10 distribution
        n1 = int(0.70 * num_paths)
        n2 = int(0.20 * num_paths)
        n3 = num_paths - n1 - n2

        return {
            int(top_indices[0]): n1,
            int(top_indices[1]): n2,
            int(top_indices[2]): n3,
        }

    def _generate_horizon(
        self,
        world_state,
        past_context: Optional[Tensor],
        regime_probs: Tensor,
        horizon_config: HorizonConfig,
        extra_context: Optional[Tensor] = None,
        steps: int = 50,
    ) -> GeneratedHorizon:
        """Generate paths for a single horizon.

        Args:
            world_state: Current market state
            past_context: Temporal context (B, T, C)
            regime_probs: (9,) regime probabilities
            horizon_config: Horizon-specific config
            extra_context: Optional additional conditioning

        Returns:
            GeneratedHorizon with paths and latent summary
        """
        # Distribute paths across top-3 regimes
        regime_distribution = self._distribute_paths_across_regimes(
            horizon_config.num_paths, regime_probs
        )

        all_paths = []
        all_regime_onehot = []

        for regime_idx, num_regime_paths in regime_distribution.items():
            if num_regime_paths == 0:
                continue

            # Create one-hot regime vector
            regime_onehot = torch.zeros(9, device=self.device)
            regime_onehot[regime_idx] = 1.0

            # Generate paths for this regime
            for _ in range(num_regime_paths):
                paths = self.base_generator.generate_paths(
                    world_state=world_state,
                    regime_probs=regime_onehot,
                    num_paths=1,
                    past_context=past_context,
                    steps=steps,
                )
                # Convert to tensor (assuming paths[0]["data"] is list)
                path_data = torch.tensor(paths[0]["data"], device=self.device)
                if path_data.dim() == 2:
                    path_data = path_data.T  # (C, L)
                all_paths.append(path_data)
                all_regime_onehot.append(regime_onehot)

        # Stack paths
        paths_tensor = torch.stack(all_paths)  # (num_paths, C, L)

        # Compute latent summary
        summary_encoder = {
            "short": self.short_summary,
            "medium": self.medium_summary,
        }.get(horizon_config.name, self.short_summary)

        latent_summary = summary_encoder(paths_tensor)

        # Regime probs for each path
        regime_probs_batch = torch.stack(all_regime_onehot)

        return GeneratedHorizon(
            name=horizon_config.name,
            paths=paths_tensor,
            latent_summary=latent_summary,
            regime_probs=regime_probs_batch,
        )

    def _generate_horizon_batched(
        self,
        world_state,
        past_context: Optional[Tensor],
        regime_probs: Tensor,
        horizon_config: HorizonConfig,
        steps: int = 50,
    ) -> GeneratedHorizon:
        """Generate paths for a single horizon - TRULY BATCHED diffusion for speed.

        Args:
            world_state: Current market state
            past_context: Temporal context (B, T, C) - full batch
            regime_probs: (B, 9) regime probabilities - batched!
            horizon_config: Horizon-specific config
            steps: Number of diffusion sampling steps

        Returns:
            GeneratedHorizon with paths and latent summary
        """
        batch_size = regime_probs.shape[0]
        num_paths_per_sample = max(1, horizon_config.num_paths // batch_size)

        all_past_contexts = []
        all_regime_vectors = []

        for b in range(batch_size):
            rp = regime_probs[b]
            regime_distribution = self._distribute_paths_across_regimes(
                num_paths_per_sample, rp
            )

            for regime_idx, num_regime_paths in regime_distribution.items():
                if num_regime_paths == 0:
                    continue

                regime_onehot = torch.zeros(9, device=self.device)
                regime_onehot[regime_idx] = 1.0

                for _ in range(num_regime_paths):
                    all_past_contexts.append(past_context[b])
                    all_regime_vectors.append(regime_onehot)

        if not all_past_contexts:
            paths_tensor = torch.zeros(1, 144, horizon_config.seq_len, device=self.device)
            regime_probs_batch = torch.zeros(1, 9, device=self.device)
        else:
            batched_context = torch.stack(all_past_contexts)
            batched_regimes = torch.stack(all_regime_vectors)

            total_paths = len(all_regime_vectors)

            paths = self.base_generator.generate_paths(
                world_state=world_state,
                regime_probs=batched_regimes,
                num_paths=total_paths,
                past_context=batched_context,
                steps=steps,
            )

            all_paths = []
            for p in paths:
                path_data = torch.tensor(p["data"], device=self.device)
                if path_data.dim() == 2:
                    path_data = path_data.T
                all_paths.append(path_data)

            paths_tensor = torch.stack(all_paths)
            regime_probs_batch = batched_regimes

        summary_encoder = self.short_summary if horizon_config.name == "short" else self.medium_summary
        latent_summary = summary_encoder(paths_tensor)

        return GeneratedHorizon(
            name=horizon_config.name,
            paths=paths_tensor,
            latent_summary=latent_summary,
            regime_probs=regime_probs_batch,
        )

    def _compute_horizon_consistency(
        self, short: GeneratedHorizon, medium: GeneratedHorizon, long: GeneratedHorizon
    ) -> float:
        """Compute consistency score between horizons.

        Measures whether short path logically leads into medium,
        and medium into long.
        """
        scores = []

        # Short-to-medium consistency
        # Check if end of short matches start of medium
        short_end = short.paths[:, :, -10:].mean(dim=2)  # (n_short, C)
        medium_start = medium.paths[:, :, :10].mean(dim=2)  # (n_med, C)

        # Pairwise similarity
        sim_matrix = torch.nn.functional.cosine_similarity(
            short_end.unsqueeze(1), medium_start.unsqueeze(0), dim=2
        )  # (n_short, n_med)
        scores.append(sim_matrix.mean().item())

        # Medium-to-long consistency
        med_end = medium.paths[:, :, -10:].mean(dim=2)
        long_start = long.paths[:, :, :10].mean(dim=2)

        sim_matrix2 = torch.nn.functional.cosine_similarity(
            med_end.unsqueeze(1), long_start.unsqueeze(0), dim=2
        )
        scores.append(sim_matrix2.mean().item())

        # Overall consistency
        return sum(scores) / len(scores)

    def generate_multi_horizon(
        self,
        world_state,
        past_context: Optional[Tensor],
        regime_probs: Tensor,
        steps: int = 50,
    ) -> MultiHorizonResult:
        """Generate hierarchical multi-horizon paths.

        Args:
            world_state: Current market state
            past_context: (B, T, C) temporal context - now accepts full batch!
            regime_probs: (B, 9) regime probability distribution - batched!
            steps: Number of diffusion sampling steps

        Returns:
            MultiHorizonResult with all horizons and consistency score
        """
        # Ensure tensors on correct device
        if past_context is not None:
            past_context = past_context.to(self.device)
        regime_probs = regime_probs.to(self.device)

        # Handle both single sample (9,) and batch (B, 9)
        if regime_probs.dim() == 1:
            regime_probs = regime_probs.unsqueeze(0)  # (1, 9)
            single_sample = True
        else:
            single_sample = False

        batch_size = regime_probs.shape[0]

        # Generate short horizon - NOW BATCHED
        short = self._generate_horizon_batched(
            world_state, past_context, regime_probs, self.SHORT_CONFIG, steps=steps
        )

        # Generate medium horizon with short summary conditioning
        medium = self._generate_horizon_batched(
            world_state, past_context, regime_probs, self.MEDIUM_CONFIG
        )

        # Generate long horizon with both summaries
        long = self._generate_horizon_batched(
            world_state, past_context, regime_probs, self.LONG_CONFIG
        )

        # Compute horizon consistency
        consistency_score = self._compute_horizon_consistency(short, medium, long)

        # Combine paths (truncate overlapping regions)
        combined = torch.cat([
            short.paths,  # First 120 bars
            medium.paths[:, :, :],  # Next 120 bars
            long.paths[:, :, :],  # Final 120 bars
        ], dim=2) if short.paths.size(0) == medium.paths.size(0) == long.paths.size(0) else self._interleave_paths(short, medium, long)

        # Regime distribution across all paths
        all_regimes = torch.cat([short.regime_probs, medium.regime_probs, long.regime_probs])
        regime_dist = {f"regime_{i}": all_regimes[:, i].mean().item() for i in range(9)}

        return MultiHorizonResult(
            short=short,
            medium=medium,
            long=long,
            combined_paths=combined,
            horizon_consistency_score=consistency_score,
            regime_distribution=regime_dist,
        )

    def _interleave_paths(
        self, short: GeneratedHorizon, medium: GeneratedHorizon, long: GeneratedHorizon
    ) -> Tensor:
        """Interleave paths from different horizons if counts differ."""
        # Simple approach: repeat to match counts
        max_paths = max(short.paths.size(0), medium.paths.size(0), long.paths.size(0))

        short_expanded = short.paths.repeat(max_paths // short.paths.size(0) + 1, 1, 1)[:max_paths]
        med_expanded = medium.paths.repeat(max_paths // medium.paths.size(0) + 1, 1, 1)[:max_paths]
        long_expanded = long.paths.repeat(max_paths // long.paths.size(0) + 1, 1, 1)[:max_paths]

        return torch.cat([short_expanded, med_expanded, long_expanded], dim=2)