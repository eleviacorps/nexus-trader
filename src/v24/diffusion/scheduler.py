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

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """Forward diffusion: add noise to x_start at timestep t.

        Args:
            x_start: Clean data of shape (B, C, L).
            t: Timestep indices of shape (B,).
            noise: Optional pre-sampled noise (default: standard Gaussian).

        Returns:
            Noisy sample x_t of shape (B, C, L).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        s1 = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return s1 * x_start + s2 * noise

    def training_loss(self, model: nn.Module, x_start: Tensor, context: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """Compute DDPM epsilon-MSE training loss.

        Args:
            model: The epsilon-prediction model (callable with x_t, t, context).
            x_start: Clean data of shape (B, C, L).
            context: Conditioning context of shape (B, ctx_dim).
            t: Optional pre-sampled timesteps (default: uniform random).

        Returns:
            Scalar MSE loss between predicted and true noise.
        """
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        noise_pred = model(x_t, t, context)
        return torch.nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: Tensor, t: Tensor, context: Tensor) -> Tensor:
        """Single DDPM reverse step: x_t -> x_{t-1}.

        Args:
            model: The epsilon-prediction model.
            x_t: Current noisy sample of shape (B, C, L).
            t: Current timestep indices of shape (B,).
            context: Conditioning context of shape (B, ctx_dim).

        Returns:
            Less noisy sample x_{t-1} of shape (B, C, L).
        """
        eps_pred = model(x_t, t, context)
        b1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        b2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        s1 = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0_pred = (x_t - s2 * eps_pred) / s1
        x0_pred = torch.clamp(x0_pred, -5.0, 5.0)
        mean = b1 * x0_pred + b2 * x_t
        log_var = self._extract(self.posterior_log_variance, t, x_t.shape)
        nonzero = (t > 0).float().reshape(-1, *([1] * (x_t.dim() - 1)))
        noise = torch.randn_like(x_t)
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def ddpm_sample(self, model: nn.Module, shape: tuple[int, ...], context: Tensor, device: torch.device) -> Tensor:
        """Full DDPM reverse sampling from T to 0.

        Args:
            model: The epsilon-prediction model.
            shape: Output shape (B, C, L).
            context: Conditioning context of shape (B, ctx_dim).
            device: Device for tensors.

        Returns:
            Generated sample of shape (B, C, L).
        """
        x = torch.randn(shape, device=device)
        for t_val in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, context)
        return x

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, shape: tuple[int, ...], context: Tensor,
                    num_steps: int = 50, eta: float = 0.0, device: torch.device = None) -> Tensor:
        """DDIM fast sampling with fewer steps.

        Args:
            model: The epsilon-prediction model.
            shape: Output shape (B, C, L).
            context: Conditioning context of shape (B, ctx_dim).
            num_steps: Number of denoising steps (default 50).
            eta: Stochasticity parameter (0 = deterministic DDIM).
            device: Device for tensors.

        Returns:
            Generated sample of shape (B, C, L).
        """
        if device is None:
            device = context.device
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)
        for i in range(len(timesteps)):
            t = torch.full((shape[0],), timesteps[i], device=device, dtype=torch.long)
            eps_pred = model(x, t, context)
            s1 = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
            s2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            x0_pred = (x - s2 * eps_pred) / s1
            x0_pred = torch.clamp(x0_pred, -5.0, 5.0)

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
