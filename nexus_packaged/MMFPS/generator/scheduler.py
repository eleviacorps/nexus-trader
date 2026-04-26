"""Noise scheduler for diffusion process."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class NoiseSchedulerConfig:
    """Configuration for noise scheduler."""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    clipping: float = 0.999


class NoiseScheduler(nn.Module):
    """DDPM noise scheduler with cosine beta schedule."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clipping: float = 0.999,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.clipping = clipping

        # Cosine beta schedule
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
        x = torch.cos((steps / num_timesteps + 0.008) / (1 + 0.008) * math.pi * 0.5) ** 2
        alphas_cumprod = x / x[0]
        alphas_cumprod = torch.clamp(alphas_cumprod, max=clipping)
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=beta_start, max=beta_end)
        alphas = 1 - betas

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod[1:])
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod[1:]))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod[1:]))

        # Posterior variance
        alphas_cumprod_for_post = alphas_cumprod[1:]  # (num_timesteps,)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod_for_post[:-1]])
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod_for_post)
        self.register_buffer("posterior_variance", posterior_variance)
        
        # Posterior mean coefficients
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_for_post) / (1 - alphas_cumprod_for_post)
        posterior_mean_coef2 = (1 - betas) * torch.sqrt(alphas_cumprod_for_post) / (1 - alphas_cumprod_for_post)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _extract(self, a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
        """Extract values from tensor based on timesteps."""
        batch_size = x_shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion: add noise to data."""
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_t * noise

    def predict_epsilon(self, model: nn.Module, x_t: Tensor, t: Tensor, **cond_kwargs) -> Tensor:
        """Predict noise from noisy input."""
        return model(x_t, t, **cond_kwargs)

    def p_sample(self, x_t: Tensor, t: Tensor, model: nn.Module, **cond_kwargs) -> Tensor:
        """Reverse diffusion step (DDPM)."""
        B = x_t.shape[0]
        
        eps_pred = model(x_t, t, **cond_kwargs)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # Predict x0
        x0_pred = (x_t - sqrt_one_minus_t * eps_pred) / sqrt_alphas_cumprod_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # Get next step
        t_next = torch.clamp(t - 1, min=0)
        sqrt_alphas_cumprod_next = self._extract(self.sqrt_alphas_cumprod, t_next, x_t.shape)
        sqrt_one_minus_next = self._extract(self.sqrt_one_minus_alphas_cumprod, t_next, x_t.shape)
        
        direction = sqrt_one_minus_next * eps_pred
        x_next = sqrt_alphas_cumprod_next * x0_pred + direction
        
        # Add noise for t > 0
        mask = (t > 0).float().reshape(B, 1, 1)
        noise = torch.randn_like(x_t)
        x_next = x_next + mask * (0.5 * self._extract(self.posterior_variance, t, x_t.shape)).sqrt() * noise
        
        return x_next

    def ddim_step(self, x_t: Tensor, t: Tensor, t_next: Tensor, model: nn.Module, 
                 eta: float = 0.0, **cond_kwargs) -> Tensor:
        """DDIM sampling step (faster)."""
        B = x_t.shape[0]
        
        eps_pred = model(x_t, t, **cond_kwargs)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # Predict x0
        x0_pred = (x_t - sqrt_one_minus_t * eps_pred) / sqrt_alphas_cumprod_t
        x0_pred = torch.clamp(x0_pred, -2.0, 2.0)
        
        # Direction to next step
        sqrt_alphas_cumprod_next = self._extract(self.sqrt_alphas_cumprod, t_next, x_t.shape)
        sqrt_one_minus_next = self._extract(self.sqrt_one_minus_alphas_cumprod, t_next, x_t.shape)
        
        direction = sqrt_one_minus_next * eps_pred
        
        if eta > 0:
            # Stochastic DDPM
            var = self._extract(self.posterior_variance, t, x_t.shape)
            x_next = sqrt_alphas_cumprod_next * x0_pred + direction + eta * var.sqrt() * torch.randn_like(x_t)
        else:
            # Deterministic DDIM
            x_next = sqrt_alphas_cumprod_next * x0_pred + direction
        
        return x_next