"""Noise scheduler for DDPM and DDIM sampling.

Implements cosine beta schedule, forward diffusion q_sample,
DDPM epsilon-MSE training loss, and both DDPM and DDIM reverse sampling.
All methods consistently use epsilon-prediction (fixes the original
x-prediction vs epsilon-loss mismatch).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class NoiseScheduler(nn.Module):
    """Cosine-schedule noise scheduler for diffusion training and sampling.

    Args:
        num_timesteps: Total diffusion steps T (default 1000).
        cosine_s: Offset for cosine schedule (default 0.008).
    """

    def __init__(self, num_timesteps: int = 1000, cosine_s: float = 0.008) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps

        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        x = torch.cos((steps / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = x / x[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)

        alphas = 1.0 - betas
        alphas_cumprod_f = alphas_cumprod[1:].float()
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod_f)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod_f)

        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod_f[:-1]])
        posterior_variance = betas.float() * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod_f)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas.float() * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod_f)
        posterior_mean_coef2 = (1.0 - alphas.float()) * torch.sqrt(alphas_cumprod_f) / (1.0 - alphas_cumprod_f)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod_f)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", posterior_log_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _extract(self, a: Tensor, t: Tensor, shape: tuple[int, ...]) -> Tensor:
        b = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(b, *([1] * (len(shape) - 1)))

    @staticmethod
    def _autocorr(x: Tensor, lag: int = 1) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x - x.mean(-1, keepdim=True)
        var = (x ** 2).mean(-1).clamp(min=1e-8)
        lag_cov = (x[:, :, lag:] * x[:, :, :-lag]).mean(-1)
        return (lag_cov / var).mean()

    @staticmethod
    def _vol_clustering(x: Tensor, lag: int = 1) -> Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        sq = torch.abs(x)
        sq_centered = sq - sq.mean(-1, keepdim=True)
        var = (sq_centered ** 2).mean(-1).clamp(min=1e-8)
        lag_cov = (sq_centered[:, :, lag:] * sq_centered[:, :, :-lag]).mean(-1)
        return (lag_cov / var).mean()

    def training_loss_regime(
        self, model: nn.Module, x_start: Tensor, context: Tensor,
        t: Optional[Tensor] = None,
        temporal_seq: Optional[Tensor] = None,
        temporal_emb: Optional[Tensor] = None,
        regime_emb: Optional[Tensor] = None,
        acf_weight: float = 0.10,
        vol_weight: float = 0.10,
        std_weight: float = 0.05,
        return_idx: int = 0,
    ) -> dict[str, Tensor]:
        """Training loss with regime conditioning for V26 Phase 1.

        Args:
            model: The diffusion U-Net model.
            x_start: Clean input (B, C, L).
            context: Conditioning context (B, ctx_dim).
            t: Optional timestep indices (B,).
            temporal_seq: Temporal encoder hidden sequence (B, T_past, d_gru).
            temporal_emb: Temporal encoder FiLM embedding (B, temporal_dim).
            regime_emb: Regime embedding (B, regime_dim) for regime conditioning.
            acf_weight: Weight for autocorrelation realism loss.
            vol_weight: Weight for volatility clustering realism loss.
            std_weight: Weight for standard deviation realism loss.
            return_idx: Index of the return channel for realism losses.

        Returns:
            Dictionary containing total loss and component losses.
        """
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)

        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        # Combine temporal_emb and regime_emb if both provided
        combined_temporal_emb = temporal_emb
        if temporal_emb is not None and regime_emb is not None:
            # Concatenate regime to temporal embedding (assumes model supports this)
            # Model must be updated to handle combined embedding dimension
            combined_temporal_emb = torch.cat([temporal_emb, regime_emb], dim=-1)
        elif regime_emb is not None:
            # Use regime_emb directly if no temporal_emb
            combined_temporal_emb = regime_emb

        eps_pred = model(x_t, t, context, temporal_seq=temporal_seq, temporal_emb=combined_temporal_emb)

        diffusion_loss = torch.nn.functional.mse_loss(eps_pred, noise)

        s1 = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0_pred = (x_t - s2 * eps_pred) / s1.clamp(min=1e-8)
        x0_pred = torch.clamp(x0_pred, -2.5, 2.5)

        ret = x0_pred[:, return_idx:return_idx + 1, :].squeeze(1)
        ret_real = x_start[:, return_idx:return_idx + 1, :].squeeze(1)

        acf_fake = self._autocorr(ret, lag=1)
        acf_real = self._autocorr(ret_real.detach(), lag=1)
        acf_loss = torch.abs(acf_fake - acf_real)
        if torch.sign(acf_fake) != torch.sign(acf_real):
            acf_loss = acf_loss * 2.0

        vol_fake = self._vol_clustering(ret, lag=1)
        vol_real = self._vol_clustering(ret_real.detach(), lag=1)
        vol_loss = torch.abs(vol_fake - vol_real)

        std_real = torch.std(ret_real.detach())
        std_fake = torch.std(ret)
        std_loss = torch.abs(torch.log(std_fake.clamp(min=1e-8) / std_real.clamp(min=1e-8)))

        total = diffusion_loss + acf_weight * acf_loss + vol_weight * vol_loss + std_weight * std_loss

        return {
            "total": total,
            "diffusion": diffusion_loss,
            "acf": acf_loss,
            "vol": vol_loss,
            "std": std_loss,
        }

    def training_loss_with_realism(
        self, model: nn.Module, x_start: Tensor, context: Tensor,
        t: Optional[Tensor] = None,
        temporal_seq: Optional[Tensor] = None,
        temporal_emb: Optional[Tensor] = None,
        acf_weight: float = 0.10,
        vol_weight: float = 0.10,
        std_weight: float = 0.05,
        return_idx: int = 0,
    ) -> dict[str, Tensor]:
        """Backward-compatible training loss with realism (without regime conditioning).

        This is a wrapper that calls training_loss_regime without regime_emb.
        """
        return self.training_loss_regime(
            model, x_start, context,
            t=t,
            temporal_seq=temporal_seq,
            temporal_emb=temporal_emb,
            regime_emb=None,
            acf_weight=acf_weight,
            vol_weight=vol_weight,
            std_weight=std_weight,
            return_idx=return_idx,
        )

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        s1 = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return s1 * x_start + s2 * noise

    def training_loss(self, model: nn.Module, x_start: Tensor, context: Tensor, t: Optional[Tensor] = None,
                      temporal_seq: Optional[Tensor] = None, temporal_emb: Optional[Tensor] = None) -> Tensor:
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        noise_pred = model(x_t, t, context, temporal_seq=temporal_seq, temporal_emb=temporal_emb)
        return torch.nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: Tensor, t: Tensor, context: Tensor,
                 temporal_seq: Optional[Tensor] = None, temporal_emb: Optional[Tensor] = None) -> Tensor:
        eps_pred = model(x_t, t, context, temporal_seq=temporal_seq, temporal_emb=temporal_emb)
        b1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        b2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        s1 = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0_pred = (x_t - s2 * eps_pred) / s1
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
        mean = b1 * x0_pred + b2 * x_t
        log_var = self._extract(self.posterior_log_variance, t, x_t.shape)
        nonzero = (t > 0).float().reshape(-1, *([1] * (x_t.dim() - 1)))
        noise = torch.randn_like(x_t)
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def ddpm_sample(self, model: nn.Module, shape: tuple[int, ...], context: Tensor, device: torch.device,
                    temporal_seq: Optional[Tensor] = None, temporal_emb: Optional[Tensor] = None) -> Tensor:
        x = torch.randn(shape, device=device)
        for t_val in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, context, temporal_seq=temporal_seq, temporal_emb=temporal_emb)
        return x

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, shape: tuple[int, ...], context: Tensor,
                    num_steps: int = 50, eta: float = 0.0, device: torch.device = None,
                    temporal_seq: Optional[Tensor] = None, temporal_emb: Optional[Tensor] = None) -> Tensor:
        if device is None:
            device = context.device
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)
        for i in range(len(timesteps)):
            t = torch.full((shape[0],), timesteps[i], device=device, dtype=torch.long)
            eps_pred = model(x, t, context, temporal_seq=temporal_seq, temporal_emb=temporal_emb)
            s1 = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
            s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            x0_pred = (x - s2 * eps_pred) / s1
            x0_pred = torch.clamp(x0_pred, -2.5, 2.5)

            if i < len(timesteps) - 1:
                t_next = torch.full((shape[0],), timesteps[i + 1], device=device, dtype=torch.long)
                s1_next = self._extract(self.sqrt_alphas_cumprod, t_next, x.shape)
                s2_next = self._extract(self.sqrt_one_minus_alphas_cumprod, t_next, x.shape)
                sigma = eta * torch.sqrt(s2_next / s2 * (1 - s1 ** 2 / s1_next ** 2))
                direction = torch.sqrt(1 - s1_next ** 2 - sigma ** 2) * eps_pred
                x = s1_next * x0_pred + direction
                if eta > 0:
                    x = x + sigma * torch.randn_like(x)
        return x