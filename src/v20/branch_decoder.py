from __future__ import annotations

import torch
from torch import nn


class BranchDecoder(nn.Module):
    def __init__(self, d_mamba: int = 256, d_latent: int = 64, horizon: int = 30, n_branches: int = 64) -> None:
        super().__init__()
        self.d_latent = int(d_latent)
        self.horizon = int(horizon)
        self.n_branches = int(n_branches)
        self.encoder = nn.Sequential(
            nn.Linear(d_mamba + horizon * 5, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.mu_proj = nn.Linear(256, d_latent)
        self.logvar_proj = nn.Linear(256, d_latent)
        self.decoder = nn.Sequential(
            nn.Linear(d_mamba + d_latent, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, horizon * 5),
        )

    def generate_branches(self, mamba_hidden: torch.Tensor, n_branches: int | None = None) -> torch.Tensor:
        count = int(n_branches or self.n_branches)
        if mamba_hidden.dim() == 1:
            z_samples = torch.randn(count, self.d_latent, device=mamba_hidden.device)
            h_expanded = mamba_hidden.unsqueeze(0).expand(count, -1)
            paths = self.decoder(torch.cat([h_expanded, z_samples], dim=-1))
            return paths.view(count, self.horizon, 5)
        batch_size = int(mamba_hidden.size(0))
        z_samples = torch.randn(batch_size, count, self.d_latent, device=mamba_hidden.device)
        h_expanded = mamba_hidden.unsqueeze(1).expand(-1, count, -1)
        decoded = self.decoder(torch.cat([h_expanded, z_samples], dim=-1).reshape(batch_size * count, -1))
        return decoded.view(batch_size, count, self.horizon, 5)
