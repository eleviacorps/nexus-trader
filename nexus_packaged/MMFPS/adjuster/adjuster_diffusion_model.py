"""Diffusion-based residual adjuster model.

The adjuster predicts epsilon noise for a residual trajectory:
delta = future - selected_path
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class DiffusionSchedule(nn.Module):
    """DDPM linear schedule with helper methods."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        t = int(config.timesteps)
        betas = torch.linspace(config.beta_start, config.beta_end, t, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # 1-indexing buffers to match diffusion timestep convention.
        self.register_buffer("betas", F.pad(betas, (1, 0), value=0.0), persistent=False)
        self.register_buffer("alphas", F.pad(alphas, (1, 0), value=1.0), persistent=False)
        self.register_buffer("alpha_bars", F.pad(alpha_bars, (1, 0), value=1.0), persistent=False)

    @property
    def timesteps(self) -> int:
        return int(self.config.timesteps)

    def extract(self, arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        out = arr.gather(0, t.long())
        shape = [t.shape[0]] + [1] * (ndim - 1)
        return out.view(*shape)

    def q_sample(self, clean_delta: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.extract(self.alpha_bars.sqrt(), t, clean_delta.ndim)
        sqrt_1m = self.extract((1.0 - self.alpha_bars).sqrt(), t, clean_delta.ndim)
        return sqrt_ab * clean_delta + sqrt_1m * noise

    def predict_x0_from_eps(
        self,
        noisy_future: torch.Tensor,
        eps_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_ab = self.extract(self.alpha_bars.sqrt(), t, noisy_future.ndim)
        sqrt_1m = self.extract((1.0 - self.alpha_bars).sqrt(), t, noisy_future.ndim)
        return (noisy_future - sqrt_1m * eps_pred) / (sqrt_ab + 1e-8)

    def p_sample_step(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t_scalar: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One reverse DDPM step for a scalar timestep."""
        t = torch.full((x_t.shape[0],), t_scalar, device=x_t.device, dtype=torch.long)
        alpha_t = self.extract(self.alphas, t, x_t.ndim)
        alpha_bar_t = self.extract(self.alpha_bars, t, x_t.ndim)
        beta_t = self.extract(self.betas, t, x_t.ndim)

        coef_1 = 1.0 / torch.sqrt(alpha_t + 1e-8)
        coef_2 = beta_t / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
        mean = coef_1 * (x_t - coef_2 * eps_pred)

        if t_scalar > 1:
            if noise is None:
                noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta_t) * noise
        return mean


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(-1)
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * -(math.log(10000.0) / max(half - 1, 1))
        )
        ang = t * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class _ContextBranch(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        seq = x.unsqueeze(-1)
        _, h = self.gru(seq)
        return h[-1]


class AdjusterDiffusionModel(nn.Module):
    """Conditional residual diffusion model."""

    def __init__(
        self,
        horizon: int = 20,
        time_dim: int = 32,
        path_channels: int = 64,
        hidden_dim: int = 64,
        regime_dim: int = 4,
        quant_dim: int = 4,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.time_dim = int(time_dim)

        # Path branch: [Noisy Future, selected_path] -> Conv1D -> GRU -> 64
        self.path_conv = nn.Sequential(
            nn.Conv1d(2, path_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.path_gru = nn.GRU(path_channels, hidden_dim, batch_first=True)

        # Multi-scale context branch
        self.ctx_120 = _ContextBranch(hidden_dim=hidden_dim)
        self.ctx_240 = _ContextBranch(hidden_dim=hidden_dim)
        self.ctx_480 = _ContextBranch(hidden_dim=hidden_dim)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(dim=time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )

        # Feature branch: regime(4) + quant(4) + xgb(1)
        feat_in = int(regime_dim + quant_dim + 1)
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_in, 32),
            nn.GELU(),
        )

        # Fusion MLP(256 -> 128 -> 64), then project to horizon.
        fusion_in = hidden_dim + (hidden_dim * 3) + time_dim + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.out = nn.Linear(64, self.horizon)

    def forward(
        self,
        noisy_future: torch.Tensor,
        selected_path: torch.Tensor,
        ctx_120: torch.Tensor,
        ctx_240: torch.Tensor,
        ctx_480: torch.Tensor,
        regime: torch.Tensor,
        quant: torch.Tensor,
        xgb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Path branch
        path_in = torch.stack([noisy_future, selected_path], dim=1)  # (B, 2, H)
        p = self.path_conv(path_in)  # (B, 64, H)
        p = p.transpose(1, 2).contiguous()  # (B, H, 64)
        _, p_h = self.path_gru(p)
        p_feat = p_h[-1]  # (B, 64)

        # Context branch
        c120 = self.ctx_120(ctx_120)
        c240 = self.ctx_240(ctx_240)
        c480 = self.ctx_480(ctx_480)
        c_feat = torch.cat([c120, c240, c480], dim=-1)  # (B, 192)

        # Time + features
        t_feat = self.time_proj(self.time_embed(t))
        f_feat = self.feature_proj(torch.cat([regime, quant, xgb], dim=-1))

        fused = torch.cat([p_feat, c_feat, t_feat, f_feat], dim=-1)
        h = self.fusion(fused)
        eps_pred = self.out(h)
        return eps_pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
