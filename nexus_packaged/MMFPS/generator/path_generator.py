"""Multi-modal path generator with regime conditioning.

Generates diverse future price paths conditioned on context, regime, and quant features.
Each path represents a distinct plausible future.
"""

from __future__ import annotations

import sys
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple


class GeneratorOutput(NamedTuple):
    paths: Tensor
    latent_z: Tensor
    regime_info: dict
    diversity_score: Tensor


class MultiModalPathGenerator(nn.Module):
    """Generates N diverse future paths conditioned on context + regime.

    Key innovations:
    - Regime conditioning via embedding injection
    - Latent space diversity enforcement
    - Per-regime movement statistics
    - Explicit diversity constraint
    """

    TREND_UP = 0
    CHOP = 1
    REVERSAL = 2

    REGIME_PARAMS = {
        TREND_UP: {
            "mean_return": 0.003,
            "volatility": 0.002,
            "trend_strength": 0.7,
        },
        CHOP: {
            "mean_return": 0.0,
            "volatility": 0.0015,
            "trend_strength": 0.2,
        },
        REVERSAL: {
            "mean_return": -0.003,
            "volatility": 0.003,
            "trend_strength": 0.6,
        },
    }

    def __init__(
        self,
        feature_dim: int = 144,
        regime_embed_dim: int = 32,
        quant_embed_dim: int = 32,
        path_dim: int = 20,
        num_paths: int = 64,
        latent_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.regime_embed_dim = regime_embed_dim
        self.quant_embed_dim = quant_embed_dim
        self.path_dim = path_dim
        self.num_paths = num_paths
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        context_dim = feature_dim + regime_embed_dim + quant_embed_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.path_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, path_dim * num_paths),
        )

        self.noise_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.diversity_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_paths * latent_dim),
        )

        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        context: Tensor,
        regime_state,
        quant_features: Tensor,
        num_paths: int | None = None,
    ) -> GeneratorOutput:
        B = context.shape[0]
        n_paths = num_paths or self.num_paths

        combined = torch.cat([context, regime_state.regime_embedding, quant_features], dim=-1)
        enc = self.context_encoder(combined)

        base = self.path_head(enc).view(B, n_paths, self.path_dim)
        base = F.layer_norm(base, [self.path_dim])
        base = torch.tanh(base) * 0.02

        noise = torch.randn(B, n_paths, self.latent_dim, device=context.device, dtype=context.dtype)
        noise_enc = self.noise_encoder(noise.view(B * n_paths, self.latent_dim))
        noise_enc = noise_enc.view(B, n_paths, -1)

        regime_probs = torch.stack([
            regime_state.prob_trend_up,
            regime_state.prob_chop,
            regime_state.prob_reversal,
        ], dim=-1)

        expected_mean = (
            regime_probs[:, 0] * self.REGIME_PARAMS[self.TREND_UP]["mean_return"] +
            regime_probs[:, 1] * self.REGIME_PARAMS[self.CHOP]["mean_return"] +
            regime_probs[:, 2] * self.REGIME_PARAMS[self.REVERSAL]["mean_return"]
        ).unsqueeze(-1)

        expected_vol = (
            regime_probs[:, 0] * self.REGIME_PARAMS[self.TREND_UP]["volatility"] +
            regime_probs[:, 1] * self.REGIME_PARAMS[self.CHOP]["volatility"] +
            regime_probs[:, 2] * self.REGIME_PARAMS[self.REVERSAL]["volatility"]
        ).unsqueeze(-1)

        trend_strength = (
            regime_probs[:, 0] * self.REGIME_PARAMS[self.TREND_UP]["trend_strength"] +
            regime_probs[:, 1] * self.REGIME_PARAMS[self.CHOP]["trend_strength"] +
            regime_probs[:, 2] * self.REGIME_PARAMS[self.REVERSAL]["trend_strength"]
        ).unsqueeze(-1)

        steps = torch.linspace(0, 1, self.path_dim, device=context.device).unsqueeze(0).unsqueeze(0)
        base_trend = expected_mean.unsqueeze(1) * steps * trend_strength.unsqueeze(1) * 10

        scale = F.softplus(self.log_scale) * expected_vol.unsqueeze(1) * 10

        path_noise = noise_enc[:, :, : self.path_dim] * scale
        raw_paths = base + base_trend + path_noise

        cumulative = torch.cumsum(raw_paths, dim=2)
        paths = cumulative

        diversity_z = torch.randn(B, n_paths, self.latent_dim, device=context.device, dtype=context.dtype)
        diversity_loss = self.diversity_encoder(enc).view(B, n_paths, self.latent_dim)

        regime_info = {
            "prob_trend_up": regime_state.prob_trend_up,
            "prob_chop": regime_state.prob_chop,
            "prob_reversal": regime_state.prob_reversal,
            "expected_mean": expected_mean,
            "expected_vol": expected_vol,
            "trend_strength": trend_strength,
        }

        diversity_score = self._compute_diversity(paths)

        return GeneratorOutput(
            paths=paths,
            latent_z=diversity_z,
            regime_info=regime_info,
            diversity_score=diversity_score,
        )

    def _compute_diversity(self, paths: Tensor) -> Tensor:
        B, n_paths, path_dim = paths.shape

        path_returns = (paths[:, :, -1] - paths[:, :, 0]) / (paths[:, :, 0] + 1e-8)

        return path_returns.std(dim=1).mean()


class DiversityLoss(nn.Module):
    """Enforces diversity among generated paths to prevent mode collapse."""

    def __init__(self, margin: float = 0.05, weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(self, paths: Tensor) -> dict[str, Tensor]:
        B, n_paths, path_dim = paths.shape

        path_returns = (paths[:, :, -1] - paths[:, :, 0]) / (paths[:, :, 0] + 1e-8)

        ret_std = path_returns.std(dim=1).mean()
        ret_range = (path_returns.max(dim=1)[0] - path_returns.min(dim=1)[0]).mean()

        pairwise_diff = path_returns.unsqueeze(2) - path_returns.unsqueeze(1)
        pairwise_diff = pairwise_diff.abs()
        eye = torch.eye(n_paths, device=pairwise_diff.device).unsqueeze(0)
        pairwise_diff = pairwise_diff * (1 - eye)
        avg_separation = pairwise_diff.sum() / (B * n_paths * (n_paths - 1) + 1e-8)

        diversity_loss = -ret_std - 0.1 * ret_range + 0.01 * avg_separation

        loss = self.weight * diversity_loss

        return {
            "diversity_loss": loss,
            "ret_std": ret_std.detach(),
            "ret_range": ret_range.detach(),
            "avg_separation": avg_separation.detach(),
        }