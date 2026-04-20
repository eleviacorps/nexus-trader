"""1D U-Net for conditional diffusion with FiLM time conditioning and cross-attention context.

Architecture:
    Encoder:   Conv1d blocks with downsample(2x) at each level
    Bottleneck: ResBlock + SelfAttention + ResBlock
    Decoder:   Upsample(2x) + skip-concat + Conv1d, cross-attention at deepest level
    Output:    Conv1d -> epsilon prediction (B, C, L)

Time conditioning: FiLM (Feature-wise Linear Modulation) via sinusoidal embeddings.
Context conditioning: Cross-attention at bottleneck + first decoder level.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
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
    def __init__(self, time_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, feature_dim * 2),
        )

    def forward(self, h: Tensor, t_emb: Tensor) -> Tensor:
        params = self.net(t_emb)
        scale, shift = params.chunk(2, dim=-1)
        return h * (1 + scale[:, :, None]) + shift[:, :, None]


class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1) -> None:
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
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(x)
        h = self.film(h, t_emb)
        h = self.conv2(h)
        return h + self.residual(x)


class SelfAttention1d(nn.Module):
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
    def __init__(self, dim: int, ctx_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.proj_ctx = nn.Linear(ctx_dim, dim) if ctx_dim != dim else nn.Identity()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.proj_out = nn.Conv1d(dim, dim, 1)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        B, C, L = x.shape
        h = self.norm(x).permute(0, 2, 1)
        ctx = self.proj_ctx(context).unsqueeze(1)
        h, _ = self.attn(h, ctx, ctx)
        return x + self.proj_out(h.permute(0, 2, 1))


class Downsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DiffusionUNet1D(nn.Module):
    """1D U-Net for epsilon-prediction conditional diffusion.

    Channel flow for base=128, multipliers=(1,2,4):
        conv_in:  100 -> 128
        down0:    128 -> 128 (2 blocks) -> downsample -> skip
        down1:    128 -> 256 (2 blocks) -> downsample -> skip
        down2:    256 -> 512 (2 blocks) -> skip (no downsample at last level)
        bottleneck: 512 -> 512 (res + attn + cross_attn + res)
        up2:      512 -> upsample -> cat(512, skip_512=512) -> 512 -> 512 (2 blocks)
        up1:      512 -> upsample -> cat(512, skip_256=256) -> 512 -> 256 (2 blocks)
        up0:      256 -> cat(256, skip_128=128) -> 256 -> 128 (2 blocks + cross_attn)
        conv_out: 128 -> 100

    Args:
        in_channels: Number of input feature channels (e.g. 100 for fused features).
        base_channels: Base channel count (default 128).
        channel_multipliers: Per-level multipliers (default (1, 2, 4)).
        time_dim: Sinusoidal time embedding dimension.
        num_res_blocks: ResBlocks per encoder/decoder level.
        ctx_dim: Context vector dimension for cross-attention (0 = disabled).
        dropout: Dropout rate in ResBlocks.
    """

    def __init__(
        self,
        in_channels: int = 100,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 256,
        num_res_blocks: int = 2,
        ctx_dim: int = 100,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.ctx_dim = ctx_dim

        ch = [base_channels * m for m in channel_multipliers]
        num_levels = len(ch)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv1d(in_channels, ch[0], 3, padding=1)

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        for level in range(num_levels):
            blocks = nn.ModuleList()
            in_c = ch[level - 1] if level > 0 else ch[0]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock1d(in_c, ch[level], time_dim, dropout))
                in_c = ch[level]
            self.encoder_blocks.append(blocks)
            self.encoder_downsample.append(
                Downsample1d(ch[level]) if level < num_levels - 1 else None
            )

        # --- Bottleneck ---
        self.bottleneck_res1 = ResBlock1d(ch[-1], ch[-1], time_dim, dropout)
        self.bottleneck_attn = SelfAttention1d(ch[-1])
        self.bottleneck_cross = CrossAttention1d(ch[-1], ctx_dim) if ctx_dim > 0 else None
        self.bottleneck_res2 = ResBlock1d(ch[-1], ch[-1], time_dim, dropout)

        # --- Decoder ---
        # Decoder processes levels in reverse: level 2 (deepest) first, then 1, then 0
        self.decoder_upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_cross_attns = nn.ModuleList()

        reversed_levels = list(reversed(range(num_levels)))
        for idx, level in enumerate(reversed_levels):
            next_level = reversed_levels[idx - 1] if idx > 0 else level
            higher_ch = ch[reversed_levels[idx - 1]] if idx > 0 else ch[-1]
            self.decoder_upsample.append(
                Upsample1d(higher_ch, ch[level]) if idx > 0 else None
            )
            blocks = nn.ModuleList()
            cross_attns = nn.ModuleList()
            skip_c = ch[level]
            in_c = ch[level] + skip_c
            for j in range(num_res_blocks + 1):
                use_cross = (level == 0) and ctx_dim > 0
                blocks.append(ResBlock1d(in_c, ch[level], time_dim, dropout))
                cross_attns.append(CrossAttention1d(ch[level], ctx_dim) if use_cross else None)
                in_c = ch[level]
            self.decoder_blocks.append(blocks)
            self.decoder_cross_attns.append(cross_attns)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch[0]),
            nn.SiLU(),
            nn.Conv1d(ch[0], in_channels, 3, padding=1),
        )

    def forward(self, x: Tensor, t: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """Forward pass — predicts epsilon given noisy input, timestep, and context.

        Args:
            x: Noisy input (B, C, L).
            t: Timestep indices (B,).
            context: Conditioning context (B, ctx_dim).

        Returns:
            Predicted epsilon (B, C, L).
        """
        t_emb = self.time_embed(t)
        h = self.conv_in(x)

        # Encoder — collect one skip per level (after all blocks at that level)
        skips = []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            if self.encoder_downsample[level] is not None:
                h = self.encoder_downsample[level](h)

        # Bottleneck
        h = self.bottleneck_res1(h, t_emb)
        h = self.bottleneck_attn(h)
        if self.bottleneck_cross is not None and context is not None:
            h = self.bottleneck_cross(h, context)
        h = self.bottleneck_res2(h, t_emb)

        # Decoder — consume one skip per level (only at first block)
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
                h = block(h, t_emb)
                if cross is not None and context is not None:
                    h = cross(h, context)

        return self.conv_out(h)
