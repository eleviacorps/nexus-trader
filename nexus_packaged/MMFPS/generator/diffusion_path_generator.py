"""Diffusion-based path generator for MMFPS.

KEY FEATURES:
- Independent latent per path (NO noise reuse)
- Proper market scale outputs
- Regime + quant conditioning via FiLM and cross-attention
- Diversity regularization
- Returns in percentage (e.g., 0.01 = 1%)
"""

from __future__ import annotations

import sys
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple

from .unet_1d import DiffusionUNet1D, UNet1DConfig
from .scheduler import NoiseScheduler


class GeneratorOutput(NamedTuple):
    paths: Tensor
    latent_z: Tensor
    regime_emb: Tensor
    quant_emb: Tensor
    diversity_loss: Tensor


@dataclass
class DiffusionGeneratorConfig:
    """Configuration for MMFPS diffusion generator - scaled to ~150M params."""
    in_channels: int = 144
    horizon: int = 20
    base_channels: int = 256
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    time_dim: int = 512
    ctx_dim: int = 144
    regime_dim: int = 64
    quant_dim: int = 64
    num_timesteps: int = 1000
    num_paths: int = 128
    sampling_steps: int = 50
    guidance_scale: float = 1.0
    scale_output: bool = True


class DiffusionPathGenerator(nn.Module):
    """Diffusion-based multi-path generator with proper conditioning.

    Returns outputs in PERCENTAGE scale (e.g., 0.01 = 1% return).
    Each path uses INDEPENDENT noise - NO sharing or reuse.
    
    Conditioning: temporal_emb (GRU) + regime_emb + quant_emb
    """

    def __init__(
        self,
        config: Optional[DiffusionGeneratorConfig] = None,
    ) -> None:
        super().__init__()
        
        self.config = config or DiffusionGeneratorConfig()
        cfg = self.config

        # Temporal Encoder (GRU) - processes full sequence
        self.temporal_gru = nn.GRU(
            input_size=cfg.in_channels,
            hidden_size=cfg.time_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal_film = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.time_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_dim, cfg.time_dim),
        )

        # UNet backbone
        unet_config = UNet1DConfig(
            in_channels=cfg.in_channels,
            horizon=cfg.horizon,
            base_channels=cfg.base_channels,
            channel_multipliers=cfg.channel_multipliers,
            time_dim=cfg.time_dim,
            ctx_dim=cfg.ctx_dim,
            regime_dim=cfg.regime_dim,
            quant_dim=cfg.quant_dim,
        )
        self.unet = DiffusionUNet1D(unet_config)

        # Noise scheduler
        self.scheduler = NoiseScheduler(cfg.num_timesteps)

        # Output scaling - convert to realistic market percentage
        if cfg.scale_output:
            self.output_scale = nn.Parameter(torch.tensor(0.02))  # 2% max move
        else:
            self.output_scale = nn.Parameter(torch.tensor(1.0))

        # Diversity loss weight
        self.diversity_weight = 0.1
    
    def encode_temporal(self, sequence: Tensor) -> Tensor:
        """Encode full sequence with GRU -> FiLM embedding.
        
        Args:
            sequence: (B, seq_len, in_channels)
            
        Returns:
            temporal_emb: (B, time_dim)
        """
        outputs, hidden = self.temporal_gru(sequence)
        final_hidden = hidden[-1]  # (B, time_dim)
        return self.temporal_film(final_hidden)

    @torch.no_grad()
    def generate_paths(
        self,
        context: Tensor,
        regime_emb: Tensor,
        quant_emb: Tensor,
        temporal_sequence: Optional[Tensor] = None,
        num_paths: Optional[int] = None,
        sampling_steps: Optional[int] = None,
    ) -> Tensor:
        """Generate multiple paths with PARALLEL batched generation.

        Args:
            context: Context features (B, ctx_dim)
            regime_emb: Regime embedding (B, regime_dim)
            quant_emb: Quant embedding (B, quant_dim)
            temporal_sequence: Full sequence for GRU encoding (B, seq_len, in_channels)
            num_paths: Number of paths to generate
            sampling_steps: DDIM sampling steps

        Returns:
            Paths tensor (B, num_paths, horizon, channels)
        """
        B = context.shape[0]
        n_paths = num_paths or self.config.num_paths
        steps = sampling_steps or self.config.sampling_steps
        
        # Encode temporal sequence if provided
        if temporal_sequence is not None:
            temporal_emb = self.encode_temporal(temporal_sequence)
        else:
            temporal_emb = torch.zeros(B, self.config.time_dim, device=context.device)
        
        # Expand conditioning for all paths at once (parallel generation)
        ctx_exp = context.unsqueeze(1).expand(-1, n_paths, -1).reshape(B * n_paths, -1)
        reg_exp = regime_emb.unsqueeze(1).expand(-1, n_paths, -1).reshape(B * n_paths, -1)
        quant_exp = quant_emb.unsqueeze(1).expand(-1, n_paths, -1).reshape(B * n_paths, -1)
        temp_exp = temporal_emb.unsqueeze(1).expand(-1, n_paths, -1).reshape(B * n_paths, -1)

        # Generate all paths in parallel
        paths = self._sample_parallel(
            context=ctx_exp,
            regime_emb=reg_exp,
            quant_emb=quant_exp,
            temporal_emb=temp_exp,
            num_steps=steps,
        )

        # Reshape: (B*n_paths, horizon, C) -> (B, n_paths, horizon, C)
        paths = paths.reshape(B, n_paths, self.config.horizon, self.config.in_channels)
        
        return paths

    @torch.no_grad()
    def _sample_parallel(
        self,
        context: Tensor,
        regime_emb: Tensor,
        quant_emb: Tensor,
        temporal_emb: Tensor,
        num_steps: int = 50,
    ) -> Tensor:
        """Sample ALL paths in parallel using single DDIM pass."""
        device = context.device
        B = context.shape[0]  # This is B * n_paths
        C = self.config.in_channels
        H = self.config.horizon

        # DDIM sampling
        step_size = self.scheduler.num_timesteps // num_steps
        timesteps = list(reversed(range(0, self.scheduler.num_timesteps, step_size)))

        # Independent noise for ALL paths (batched together)
        x = torch.randn(B, C, H, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            
            # Predict epsilon for ALL paths at once (include temporal_emb)
            cond_kwargs = {
                "context": context,
                "regime_emb": regime_emb,
                "quant_emb": quant_emb,
                "temporal_emb": temporal_emb,
            }
            eps_pred = self.unet(x, t, **cond_kwargs)

            # DDIM step
            t_next_val = timesteps[min(i + 1, len(timesteps) - 1)]
            t_next = torch.full((B,), t_next_val, device=device, dtype=torch.long)
            
            x = self._ddim_step(x, t, t_next, eps_pred)

        # Scale to realistic market range
        x = x * self.output_scale

        return x.permute(0, 2, 1)  # (B, horizon, C)

    def _ddim_step(
        self,
        x_t: Tensor,
        t: Tensor,
        t_next: Tensor,
        eps_pred: Tensor,
    ) -> Tensor:
        """DDIM sampling step."""
        sqrt_alphas_cumprod_t = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_t = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # Predict x0
        x0_pred = (x_t - sqrt_one_minus_t * eps_pred) / sqrt_alphas_cumprod_t
        x0_pred = torch.clamp(x0_pred, -2.0, 2.0)

        # Direction to next
        sqrt_alphas_cumprod_next = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t_next, x_t.shape)
        sqrt_one_minus_next = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_next, x_t.shape)

        direction = sqrt_one_minus_next * eps_pred
        x_next = sqrt_alphas_cumprod_next * x0_pred + direction

        return x_next

    def forward(
        self,
        context: Tensor,
        regime_emb: Tensor,
        quant_emb: Tensor,
        targets: Optional[Tensor] = None,
    ) -> GeneratorOutput:
        """Forward pass for training.

        Args:
            context: Context features (B, ctx_dim)
            regime_emb: Regime embedding (B, regime_dim)
            quant_emb: Quant embedding (B, quant_dim)
            targets: Target paths (B, n_paths, horizon, C) or None for generation

        Returns:
            GeneratorOutput with paths and metrics
        """
        B = context.shape[0]
        n_paths = self.config.num_paths

        if targets is not None:
            # Training mode: compute diffusion loss
            # targets shape: (B, n_paths, horizon, C)
            n_paths_actual = targets.shape[1]
            
            # Flatten paths for diffusion training
            targets_flat = targets[:, :, :, :].permute(0, 1, 3, 2).reshape(B * n_paths_actual, self.config.in_channels, self.config.horizon)
            
            # Sample random timesteps
            t = torch.randint(0, self.scheduler.num_timesteps, (B * n_paths_actual,), device=context.device)
            
            # Add noise
            noise = torch.randn_like(targets_flat)
            noisy_targets = self.scheduler.q_sample(targets_flat, t, noise)
            
            # Predict noise
            # Expand conditioning for all paths
            ctx_exp = context.unsqueeze(1).expand(-1, n_paths_actual, -1).reshape(B * n_paths_actual, -1)
            reg_exp = regime_emb.unsqueeze(1).expand(-1, n_paths_actual, -1).reshape(B * n_paths_actual, -1)
            quant_exp = quant_emb.unsqueeze(1).expand(-1, n_paths_actual, -1).reshape(B * n_paths_actual, -1)
            
            # Encode temporal from targets for conditioning
            if targets is not None:
                # targets: (B, 1, horizon, C) -> squeeze to (B, horizon, C)
                target_seq = targets.squeeze(1)  # (B, horizon, C)
                temp_exp = self.encode_temporal(target_seq)
                temp_exp = temp_exp.unsqueeze(1).expand(-1, n_paths_actual, -1).reshape(B * n_paths_actual, -1)
            else:
                temp_exp = torch.zeros(B * n_paths_actual, self.config.time_dim, device=context.device)
            
            cond_kwargs = {
                "context": ctx_exp,
                "regime_emb": reg_exp,
                "quant_emb": quant_exp,
                "temporal_emb": temp_exp,
            }
            eps_pred = self.unet(noisy_targets, t, **cond_kwargs)
            
            # Diffusion loss
            diff_loss = F.mse_loss(eps_pred, noise)
            
            # Generate paths for diversity loss computation (with gradients)
            gen_paths_for_loss = self.generate_paths(context, regime_emb, quant_emb)
            
            # Diversity loss on generated paths
            diversity_loss = self._compute_diversity_loss(gen_paths_for_loss)
            
            total_loss = diff_loss + self.diversity_weight * diversity_loss
            
            # Return generated paths (for metrics)
            gen_paths = gen_paths_for_loss.detach()
            
            return GeneratorOutput(
                paths=gen_paths,
                latent_z=torch.randn(B, n_paths, 16, device=context.device),
                regime_emb=regime_emb,
                quant_emb=quant_emb,
                diversity_loss=total_loss,
            )
        else:
            # Generation mode
            gen_paths = self.generate_paths(context, regime_emb, quant_emb)
            latent_z = torch.randn(B, n_paths, 16, device=context.device)
            
            diversity_loss = self._compute_diversity_loss(gen_paths)
            
            return GeneratorOutput(
                paths=gen_paths,
                latent_z=latent_z,
                regime_emb=regime_emb,
                quant_emb=quant_emb,
                diversity_loss=diversity_loss,
            )

    def _compute_diversity_loss(self, paths: Tensor) -> Tensor:
        """Compute diversity loss to prevent mode collapse.
        
        Higher diversity = lower loss (we minimize this).
        """
        B, n_paths, horizon, C = paths.shape
        
        # Compute per-path returns
        returns = (paths[:, :, -1, :] - paths[:, :, 0, :]) / (paths[:, :, 0, :].abs() + 1e-8)
        returns = returns.mean(dim=-1)  # (B, n_paths)
        
        # Standard deviation of returns (want high)
        ret_std = returns.std(dim=1).mean()
        
        # Range of returns (want high)
        ret_range = (returns.max(dim=1)[0] - returns.min(dim=1)[0]).mean()
        
        # Pairwise distance (want high)
        returns_exp = returns.unsqueeze(2) - returns.unsqueeze(1)
        pairwise_dist = returns_exp.abs()
        eye = torch.eye(n_paths, device=pairwise_dist.device).unsqueeze(0)
        pairwise_dist = pairwise_dist * (1 - eye)
        avg_sep = pairwise_dist.sum() / (B * n_paths * (n_paths - 1) + 1e-8)
        
        # Diversity loss: minimize negative of these
        diversity_loss = -(ret_std + 0.1 * ret_range + 0.01 * avg_sep)
        
        return diversity_loss

    @torch.no_grad()
    def quick_generate(
        self,
        context: Tensor,
        regime_emb: Tensor,
        quant_emb: Tensor,
        temporal_sequence: Optional[Tensor] = None,
        num_paths: int = 32,
    ) -> Tensor:
        """Quick generation with fewer steps."""
        return self.generate_paths(
            context=context,
            regime_emb=regime_emb,
            quant_emb=quant_emb,
            temporal_sequence=temporal_sequence,
            num_paths=num_paths,
            sampling_steps=20,
        )