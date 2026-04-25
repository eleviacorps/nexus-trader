"""UNet1D backbone for MMFPS diffusion generator.

Architecture improvements over V24:
- FiLM conditioning for regime + quant embeddings
- Cross-attention for context conditioning
- Proper independent latent per path (no noise reuse)
- Diversity-aware output scaling
- Realistic market scale outputs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class UNet1DConfig:
    """Configuration for UNet1D backbone - scaled to ~150M params."""
    in_channels: int = 144
    horizon: int = 20
    base_channels: int = 256
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    time_dim: int = 512
    ctx_dim: int = 144
    regime_dim: int = 64
    quant_dim: int = 64
    num_res_blocks: int = 2
    dropout: float = 0.1


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMConditioning(nn.Module):
    """FiLM (Feature-wise Linear Modulation) conditioning."""
    
    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, feature_dim * 2),
            nn.SiLU(),
        )

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        params = self.net(cond)
        scale, shift = params.chunk(2, dim=-1)
        return h * (1 + scale[:, :, None]) + shift[:, :, None]


class ResBlock1d(nn.Module):
    """Residual block with FiLM conditioning."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        dropout: float = 0.1,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
        )
        self.film = FiLMConditioning(time_dim, out_ch)
        self.cond_film = FiLMConditioning(cond_dim, out_ch) if cond_dim > 0 else None
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
        cond_emb: Optional[Tensor] = None,
    ) -> Tensor:
        h = self.conv1(x)
        h = self.film(h, t_emb)
        if self.cond_film is not None and cond_emb is not None:
            h = self.cond_film(h, cond_emb)
        h = self.conv2(h)
        return h + self.residual(x)


class SelfAttention1d(nn.Module):
    """Self-attention for temporal features."""
    
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.proj_out = nn.Conv1d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, L = x.shape
        h = self.norm(x).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + self.proj_out(h.permute(0, 2, 1))


class CrossAttention1d(nn.Module):
    """Cross-attention for context conditioning."""
    
    def __init__(self, dim: int, ctx_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.proj_ctx = nn.Linear(ctx_dim, dim) if ctx_dim != dim else nn.Identity()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.proj_out = nn.Conv1d(dim, dim, 1)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        B, C, L = x.shape
        h = self.norm(x).permute(0, 2, 1)
        ctx = self.proj_ctx(context)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(1)
        h, _ = self.attn(h, ctx, ctx)
        return x + self.proj_out(h.permute(0, 2, 1))


class Downsample1d(nn.Module):
    """Downsampling layer."""
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsampling layer."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DiffusionUNet1D(nn.Module):
    """1D U-Net for epsilon-prediction conditional diffusion.

    Channel flow (base=128, multipliers=(1,2,4)):
    conv_in: in_channels -> 128
    down0: 128 -> 128 (2 blocks) -> downsample -> skip
    down1: 128 -> 256 (2 blocks) -> downsample -> skip
    down2: 256 -> 512 (2 blocks) -> skip (no downsample)
    bottleneck: 512 -> 512 (res + attn + cross_attn + res)
    up2: 512 -> upsample -> cat -> 512 -> 512 (2 blocks)
    up1: 512 -> upsample -> cat -> 512 -> 256 (2 blocks)
    up0: 256 -> cat -> 256 -> 128 (2 blocks)
    conv_out: 128 -> in_channels

    Conditioning:
    - time_emb: FiLM at every ResBlock
    - ctx_emb: Cross-attention at bottleneck + decoder
    - regime_emb: FiLM at every ResBlock + decoder cross-attention
    - quant_emb: FiLM at every ResBlock + decoder cross-attention
    """

    def __init__(
        self,
        config: Optional[UNet1DConfig] = None,
        in_channels: int = 144,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
        ctx_dim: int = 144,
        regime_dim: int = 32,
        quant_dim: int = 32,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        if config is not None:
            in_channels = config.in_channels
            base_channels = config.base_channels
            channel_multipliers = config.channel_multipliers
            time_dim = config.time_dim
            ctx_dim = config.ctx_dim
            regime_dim = config.regime_dim
            quant_dim = config.quant_dim
            num_res_blocks = config.num_res_blocks
            dropout = config.dropout

        self.in_channels = in_channels
        self.ctx_dim = ctx_dim
        self.regime_dim = regime_dim
        self.quant_dim = quant_dim

        ch = [base_channels * m for m in channel_multipliers]
        num_levels = len(ch)
        
        cond_dim = regime_dim + quant_dim  # Combined conditioning dimension

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv1d(in_channels, ch[0], 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        for level in range(num_levels):
            blocks = nn.ModuleList()
            in_c = ch[level - 1] if level > 0 else ch[0]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock1d(in_c, ch[level], time_dim, dropout, cond_dim))
                in_c = ch[level]
            self.encoder_blocks.append(blocks)
            self.encoder_downsample.append(
                Downsample1d(ch[level]) if level < num_levels - 1 else None
            )

        # Bottleneck
        self.bottleneck_res1 = ResBlock1d(ch[-1], ch[-1], time_dim, dropout, cond_dim)
        self.bottleneck_attn = SelfAttention1d(ch[-1])
        self.bottleneck_cross = CrossAttention1d(ch[-1], ctx_dim) if ctx_dim > 0 else None
        self.bottleneck_res2 = ResBlock1d(ch[-1], ch[-1], time_dim, dropout, cond_dim)

        # Decoder
        self.decoder_upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_cross_attns = nn.ModuleList()

        reversed_levels = list(reversed(range(num_levels)))
        for idx, level in enumerate(reversed_levels):
            higher_ch = ch[reversed_levels[idx - 1]] if idx > 0 else ch[-1]
            self.decoder_upsample.append(
                Upsample1d(higher_ch, ch[level]) if idx > 0 else None
            )
            blocks = nn.ModuleList()
            cross_attns = nn.ModuleList()
            skip_c = ch[level]
            in_c = ch[level] + skip_c
            for _ in range(num_res_blocks):
                blocks.append(ResBlock1d(in_c, ch[level], time_dim, dropout, cond_dim))
                cross_attns.append(CrossAttention1d(ch[level], ctx_dim) if ctx_dim > 0 else None)
                in_c = ch[level]
            self.decoder_blocks.append(blocks)
            self.decoder_cross_attns.append(cross_attns)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch[0]),
            nn.SiLU(),
            nn.Conv1d(ch[0], in_channels, 3, padding=1),
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Optional[Tensor] = None,
        regime_emb: Optional[Tensor] = None,
        quant_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass - predicts epsilon.

        Args:
            x: Noisy input (B, C, L).
            t: Timestep indices (B,).
            context: Context embedding (B, ctx_dim).
            regime_emb: Regime embedding (B, regime_dim).
            quant_emb: Quant embedding (B, quant_dim).

        Returns:
            Predicted epsilon (B, C, L).
        """
        t_emb = self.time_embed(t)
        
        # Combine regime and quant conditioning
        if regime_emb is not None and quant_emb is not None:
            cond_emb = torch.cat([regime_emb, quant_emb], dim=-1)
        elif regime_emb is not None:
            cond_emb = regime_emb
        elif quant_emb is not None:
            cond_emb = quant_emb
        else:
            cond_emb = None

        h = self.conv_in(x)

        # Encoder
        skips = []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                h = block(h, t_emb, cond_emb)
            skips.append(h)
            if self.encoder_downsample[level] is not None:
                h = self.encoder_downsample[level](h)

        # Bottleneck
        h = self.bottleneck_res1(h, t_emb, cond_emb)
        h = self.bottleneck_attn(h)
        if self.bottleneck_cross is not None and context is not None:
            h = self.bottleneck_cross(h, context)
        h = self.bottleneck_res2(h, t_emb, cond_emb)

        # Decoder
        for level_idx in range(len(self.decoder_blocks)):
            if self.decoder_upsample[level_idx] is not None:
                h = self.decoder_upsample[level_idx](h)
            blocks = self.decoder_blocks[level_idx]
            cross_attns = self.decoder_cross_attns[level_idx]
            for j, (block, cross) in enumerate(zip(blocks, cross_attns)):
                if j == 0:
                    skip = skips.pop()
                    diff = h.shape[2] - skip.shape[2]
                    if diff > 0:
                        skip = F.pad(skip, (0, diff))
                    elif diff < 0:
                        h = F.pad(h, (0, -diff))
                    h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb, cond_emb)
                if cross is not None and context is not None:
                    h = cross(h, context)

        return self.conv_out(h)