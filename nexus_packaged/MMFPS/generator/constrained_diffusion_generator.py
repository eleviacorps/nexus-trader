"""Context-Constrained Diffusion Path Generator for MMFPS vNext.

ARCHITECTURE:
1. ContextEncoder - encodes context to distribution envelope (mu, sigma, bounds)
2. DiffusionGenerator - generates normalized residuals in z-space
3. Reconstruction - converts z -> returns using envelope

KEY INSIGHT:
Instead of generating arbitrary futures → fix later
We: Define valid distribution → generate within it
"""

from __future__ import annotations

import sys
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .unet_1d import DiffusionUNet1D, UNet1DConfig
from .scheduler import NoiseScheduler


class DistributionEnvelope(NamedTuple):
    """Predicted distribution envelope from context encoder."""
    mu: Tensor           # Expected return (drift)
    sigma: Tensor       # Volatility scale
    low_bound: Tensor   # Lower bound (p5)
    high_bound: Tensor  # Upper bound (p95)


class GeneratorOutput(NamedTuple):
    paths: Tensor
    envelope: DistributionEnvelope
    diversity_loss: Tensor


@dataclass
class DiffusionGeneratorConfig:
    """Configuration for MMFPS diffusion generator."""
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
    
    # Target distribution (from real data analysis)
    target_std: float = 0.05   # Match realistic spread (~5%)
    target_mean: float = 0.0   # Centered at zero


class ContextEncoder(nn.Module):
    """Encodes market context into distribution envelope.
    
    Predicts: mu, sigma, low_bound, high_bound
    This defines what "valid reality" looks like for conditioning.
    """
    
    def __init__(
        self,
        in_channels: int = 144,
        seq_len: int = 120,
        hidden_dim: int = 512,
        out_features: int = 4,
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Temporal processing withTransformer-style layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        
        # Pool and project to distribution envelope
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, out_features),
        )
        
        # Initialize output projections for realistic envelope
        self._initEnvelope()
    
    def _initEnvelope(self):
        """Initialize envelope parameters for realistic values."""
        with torch.no_grad():
            # mu: near 0 (neutral return)
            self.fc[-1].bias[0] = 0.0
            # log_sigma: ~ -3.0 (sigma ≈ 0.05) - REALISTIC for short horizon
            self.fc[-1].bias[1] = -3.0
            # low_bound: p5 ≈ -0.1 (-10%)
            self.fc[-1].bias[2] = -0.1
            # high_bound: p95 ≈ 0.1 (+10%)
            self.fc[-1].bias[3] = 0.1
    
    def forward(self, sequence: Tensor) -> DistributionEnvelope:
        """Encode sequence to distribution envelope.
        
        Args:
            sequence: (B, seq_len, in_channels) or (B, in_channels, seq_len)
            
        Returns:
            DistributionEnvelope with mu, sigma, low_bound, high_bound
        """
        # Handle both input formats
        if sequence.dim() == 3:
            # (B, seq_len, C) -> (B, C, seq_len)
            x = sequence.transpose(1, 2)
        else:
            x = sequence
        
        # Temporal encoding
        h = self.conv_layers(x)
        
        # Project to envelope
        out = self.fc(h)
        
        # Parse outputs
        mu = out[:, 0]
        log_sigma = out[:, 1]
        low = out[:, 2]
        high = out[:, 3]
        
        # Transform to proper ranges
        sigma = F.softplus(log_sigma) + 0.01  # Ensure positive
        
        # Bounds: use tanh to soft-clip
        low_bound = torch.tanh(low) * 0.5  # Soft bound to ±50%
        high_bound = torch.tanh(high) * 0.5
        
        return DistributionEnvelope(
            mu=mu,
            sigma=sigma,
            low_bound=low_bound,
            high_bound=high_bound,
        )


class ConstrainedDiffusionGenerator(nn.Module):
    """Context-constrained diffusion generator.
    
    Two-stage generation:
    1. Encode context → distribution envelope
    2. Generate z → transform using envelope
    """
    
    def __init__(
        self,
        config: Optional[DiffusionGeneratorConfig] = None,
    ) -> None:
        super().__init__()
        
        self.config = config or DiffusionGeneratorConfig()
        cfg = self.config
        
        # Context encoder (NEW)
        self.context_encoder = ContextEncoder(
            in_channels=cfg.in_channels,
            seq_len=cfg.horizon,
            hidden_dim=cfg.time_dim,
            out_features=4,
        )
        
        # Temporal encoder for conditioning
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
        
        # UNet backbone for z-generation
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
        
        # Target distribution for loss
        self.register_buffer('target_std', torch.tensor(cfg.target_std))
        self.register_buffer('target_mean', torch.tensor(cfg.target_mean))
    
    def encode_context(self, sequence: Tensor) -> DistributionEnvelope:
        """Encode context to distribution envelope."""
        return self.context_encoder(sequence)
    
    def encode_temporal(self, sequence: Tensor) -> Tensor:
        """Encode temporal sequence for conditioning."""
        outputs, hidden = self.temporal_gru(sequence)
        final_hidden = hidden[-1]
        return self.temporal_film(final_hidden)
    
    def z_to_returns(
        self,
        z: Tensor,
        envelope: DistributionEnvelope,
    ) -> Tensor:
        """Transform z-space to returns using envelope - with HARD bounds."""
        z_price = z[:, :, 0:1]
        
        # Clamp z to prevent extreme values BEFORE transformation
        z_price = torch.clamp(z_price, -3.0, 3.0)
        
        # Transform: returns = mu + sigma * z
        mu = envelope.mu.view(-1, 1, 1)
        sigma = envelope.sigma.view(-1, 1, 1)
        
        returns = mu + sigma * z_price
        
        # HARD clamp to bounds (critical for stability)
        low = envelope.low_bound.view(-1, 1, 1)
        high = envelope.high_bound.view(-1, 1, 1)
        
        returns = torch.clamp(returns, low, high)
        
        # Replace first channel
        result = z.clone()
        result[:, :, 0:1] = returns
        
        return result
    
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
        """Generate multiple paths with envelope constraint."""
        B = context.shape[0]
        n_paths = num_paths or self.config.num_paths
        steps = sampling_steps or self.config.sampling_steps
        
        # Step 1: Get distribution envelope from context
        if temporal_sequence is not None:
            envelope = self.encode_context(temporal_sequence)
            temporal_emb = self.encode_temporal(temporal_sequence)
        else:
            # Default envelope if no sequence provided - TIGHT bounds
            envelope = DistributionEnvelope(
                mu=torch.zeros(B, device=context.device),
                sigma=torch.ones(B, device=context.device) * 0.05,
                low_bound=torch.ones(B, device=context.device) * -0.1,
                high_bound=torch.ones(B, device=context.device) * 0.1,
            )
            temporal_emb = torch.zeros(B, self.config.time_dim, device=context.device)
        
        # Expand conditioning
        ctx_exp = context.unsqueeze(1).expand(-1, n_paths, -1).reshape(B * n_paths, -1)
        reg_exp = regime_emb.unsqueeze(1).expand(-1, n_paths, -1).reshape(B * n_paths, -1)
        quant_exp = quant_emb.unsqueeze(1).expand(-1, n_paths, -1).reshape(B * n_paths, -1)
        temp_exp = temporal_emb.unsqueeze(1).expand(-1, n_paths, -1).reshape(B * n_paths, -1)
        
        # Expand envelope for each generated path (B -> B*n_paths)
        total_paths = B * n_paths
        # Use repeat_interleave for proper expansion
        mu_exp = envelope.mu.repeat_interleave(n_paths)
        sigma_exp = envelope.sigma.repeat_interleave(n_paths)
        low_exp = envelope.low_bound.repeat_interleave(n_paths)
        high_exp = envelope.high_bound.repeat_interleave(n_paths)
        z_paths = self._sample_parallel(
            context=ctx_exp,
            regime_emb=reg_exp,
            quant_emb=quant_exp,
            temporal_emb=temp_exp,
            num_steps=steps,
        )
        
        # Step 3: Transform to returns using envelope
        path_envelope = DistributionEnvelope(
            mu=mu_exp,
            sigma=sigma_exp,
            low_bound=low_exp,
            high_bound=high_exp,
        )
        
        returns = self.z_to_returns(z_paths, path_envelope)
        
        # Final hard clamp - cannot exceed bounds even with numerical issues
        returns = torch.clamp(returns, -0.5, 0.5)
        
        # Reshape: (B*n_paths, horizon, C) -> (B, n_paths, horizon, C)
        returns = returns.reshape(B, n_paths, self.config.horizon, self.config.in_channels)
        
        return returns
    
    @torch.no_grad()
    def _sample_parallel(
        self,
        context: Tensor,
        regime_emb: Tensor,
        quant_emb: Tensor,
        temporal_emb: Tensor,
        num_steps: int = 50,
    ) -> Tensor:
        """Sample in z-space using DDIM."""
        device = context.device
        B = context.shape[0]
        C = self.config.in_channels
        H = self.config.horizon
        
        step_size = self.scheduler.num_timesteps // num_steps
        timesteps = list(reversed(range(0, self.scheduler.num_timesteps, step_size)))
        
        # Start from standard normal
        x = torch.randn(B, C, H, device=device)
        
        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            
            cond_kwargs = {
                "context": context,
                "regime_emb": regime_emb,
                "quant_emb": quant_emb,
                "temporal_emb": temporal_emb,
            }
            eps_pred = self.unet(x, t, **cond_kwargs)
            
            t_next_val = timesteps[min(i + 1, len(timesteps) - 1)]
            t_next = torch.full((B,), t_next_val, device=device, dtype=torch.long)
            
            x = self._ddim_step(x, t, t_next, eps_pred)
        
        return x.permute(0, 2, 1)
    
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
        
        x0_pred = (x_t - sqrt_one_minus_t * eps_pred) / sqrt_alphas_cumprod_t
        x0_pred = torch.clamp(x0_pred, -2.0, 2.0)
        
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
        """Forward pass for training."""
        B = context.shape[0]
        n_paths = self.config.num_paths
        
        if targets is not None:
            # Training mode
            n_paths_actual = targets.shape[1]
            
            # Flatten paths
            targets_flat = targets.permute(0, 1, 3, 2).reshape(B * n_paths_actual, self.config.in_channels, self.config.horizon)
            
            # Get context for envelope prediction
            if targets.dim() == 4:
                target_seq = targets[:, 0, :, :]  # (B, horizon, C)
            else:
                target_seq = targets
            
            # Predict envelope from context
            envelope = self.encode_context(target_seq)
            
            # Transform targets to z-space for diffusion loss
            returns_for_z = target_seq[:, :, 0:1]  # (B, H, 1)
            mu = envelope.mu.view(-1, 1, 1)
            sigma = envelope.sigma.view(-1, 1, 1)
            z_targets = (returns_for_z - mu) / sigma
            
            # Expand z_targets to match all paths
            z_targets_exp = z_targets.unsqueeze(1).expand(-1, n_paths_actual, -1, -1).reshape(B * n_paths_actual, 1, self.config.horizon)
            
            # Replace first channel with z-version
            z_targets_full = targets_flat.clone()
            z_targets_full[:, 0:1, :] = z_targets_exp
            
            # Sample timesteps and add noise
            t = torch.randint(0, self.scheduler.num_timesteps, (B * n_paths_actual,), device=context.device)
            noise = torch.randn_like(z_targets_full)
            noisy_targets = self.scheduler.q_sample(z_targets_full, t, noise)
            
            # Expand conditioning
            ctx_exp = context.unsqueeze(1).expand(-1, n_paths_actual, -1).reshape(B * n_paths_actual, -1)
            reg_exp = regime_emb.unsqueeze(1).expand(-1, n_paths_actual, -1).reshape(B * n_paths_actual, -1)
            quant_exp = quant_emb.unsqueeze(1).expand(-1, n_paths_actual, -1).reshape(B * n_paths_actual, -1)
            temp_exp = self.encode_temporal(target_seq).unsqueeze(1).expand(-1, n_paths_actual, -1).reshape(B * n_paths_actual, -1)
            
            cond_kwargs = {
                "context": ctx_exp,
                "regime_emb": reg_exp,
                "quant_emb": quant_exp,
                "temporal_emb": temp_exp,
            }
            eps_pred = self.unet(noisy_targets, t, **cond_kwargs)
            
            # Diffusion loss
            diff_loss = F.mse_loss(eps_pred, noise)
            
            # Only compute auxiliary losses at reasonable timesteps (< 800)
            # At high t, sqrt_alpha is near zero causing x0_pred division instability
            use_aux = (t < 800).float().mean()
            
            if use_aux > 0:
                sqrt_alphas_cumprod_t = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t, noisy_targets.shape)
                sqrt_one_minus_t = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t, noisy_targets.shape)
                sqrt_alphas_cumprod_t = torch.clamp(sqrt_alphas_cumprod_t, min=1e-6)
                x0_pred = (noisy_targets - sqrt_one_minus_t * eps_pred) / sqrt_alphas_cumprod_t
                x0_pred = torch.clamp(x0_pred, min=-5.0, max=5.0)
                x0_pred = x0_pred.permute(0, 2, 1)
                
                raw_magnitude = x0_pred.abs()
                extreme_penalty = (raw_magnitude > 0.15).float().mean() * 2.0
                
                raw_returns = x0_pred[:, :, 0]
                raw_std = raw_returns.std(dim=1).mean()
                raw_mean = raw_returns.mean()
                
                std_loss = (raw_std - self.target_std) ** 2
                mean_loss = (raw_mean - self.target_mean) ** 2
                
                # Correlation-aware temporal loss: normalize r_t, r_{t-1} independently so correlation is scale-invariant
                r_t = raw_returns[:, :-1]
                r_tp1 = raw_returns[:, 1:]
                r_t_norm = r_t / (r_t.std(dim=1, keepdim=True) + 1e-6)
                r_tp1_norm = r_tp1 / (r_tp1.std(dim=1, keepdim=True) + 1e-6)
                corr = (r_t_norm * r_tp1_norm).mean()
                corr_loss = torch.clamp(1.0 - corr, min=0.0, max=2.0)
                
                # Variance floor: prevent std collapse below viable volatility
                min_std = 0.035
                std_floor_loss = F.relu(min_std - raw_std)
                
                # NO velocity loss - it causes std collapse
                temporal_loss = 0.05 * corr_loss + 0.2 * std_floor_loss
                
                aux_loss = 0.7 * std_loss + 1.0 * mean_loss + 0.1 * extreme_penalty + temporal_loss
            else:
                aux_loss = torch.tensor(0.0, device=noisy_targets.device)
            
            total_loss = diff_loss + use_aux * aux_loss
            total_loss = torch.clamp(total_loss, max=100.0)
            
            diversity_loss = torch.tensor(0.0, device=context.device)
            
            return GeneratorOutput(
                paths=targets.detach(),  # Return targets, not generated
                envelope=envelope,
                diversity_loss=total_loss,
            )
        else:
            # Generation mode
            gen_paths = self.generate_paths(context, regime_emb, quant_emb)
            envelope = self.encode_context(gen_paths[:, 0, :, :])
            diversity_loss = self._compute_diversity_loss(gen_paths)
            
            return GeneratorOutput(
                paths=gen_paths,
                envelope=envelope,
                diversity_loss=diversity_loss,
            )
    
    def _compute_diversity_loss(self, paths: Tensor) -> Tensor:
        """Compute diversity loss - MATCH target distribution, NOT maximize."""
        B, n_paths, horizon, C = paths.shape
        
        # Get returns from first channel
        returns = paths[:, :, -1, 0]  # (B, n_paths)
        
        # Distribution matching (NOT maximization)
        ret_std = returns.std(dim=1).mean()
        ret_mean = returns.mean()
        
        std_loss = (ret_std - self.target_std) ** 2
        mean_loss = (ret_mean - self.target_mean) ** 2
        
        # Small magnitude penalty for stability
        magnitude_penalty = (paths ** 2).mean() * 0.01
        
        return std_loss + mean_loss + magnitude_penalty
    
    @torch.no_grad()
    def quick_generate(
        self,
        context: Tensor,
        regime_emb: Tensor,
        quant_emb: Tensor,
        temporal_sequence: Optional[Tensor] = None,
        num_paths: int = 32,
    ) -> Tensor:
        """Quick generation."""
        return self.generate_paths(
            context=context,
            regime_emb=regime_emb,
            quant_emb=quant_emb,
            temporal_sequence=temporal_sequence,
            num_paths=num_paths,
            sampling_steps=20,
        )