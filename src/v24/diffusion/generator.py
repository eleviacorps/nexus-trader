"""DiffusionPathGeneratorV2 — Classifier-free guidance path generation.

Phase 0.5 additions:
- Temporal encoder integration for path generation
- Denormalize() method for converting normalized outputs back to original scale
- Tighter x0_pred clamp [-3, 3] (was [-5, 5])
- CFG with zero-temporal-context for unconditional prediction
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor

from src.v24.diffusion.unet_1d import DiffusionUNet1D
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.temporal_encoder import TemporalEncoder
from src.v24.world_state import WorldState


@dataclass
class GeneratorConfig:
    in_channels: int = 144
    sequence_length: int = 120
    base_channels: int = 128
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    time_dim: int = 256
    num_timesteps: int = 1000
    ctx_dim: int = 144
    guidance_scale: float = 3.0
    num_paths: int = 32
    sampling_steps: int = 50
    dropout: float = 0.1
    temporal_gru_dim: int = 256
    temporal_layers: int = 2
    context_len: int = 256
    norm_stats_path: Optional[str] = None


class DiffusionPathGeneratorV2:
    """Conditional diffusion path generator with classifier-free guidance and temporal encoder.

    Args:
        config: Generator configuration.
        model: Pre-trained DiffusionUNet1D model (if None, created from config).
        scheduler: Noise scheduler (if None, created from config).
        temporal_encoder: Optional pre-built TemporalEncoder (if None, created from config).
        device: Device for computation.
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        model: Optional[DiffusionUNet1D] = None,
        scheduler: Optional[NoiseScheduler] = None,
        temporal_encoder: Optional[TemporalEncoder] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or GeneratorConfig()
        self.device = torch.device(device)

        temporal_dim = self.config.time_dim if self.config.temporal_gru_dim > 0 else 0

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = DiffusionUNet1D(
                in_channels=self.config.in_channels,
                base_channels=self.config.base_channels,
                channel_multipliers=self.config.channel_multipliers,
                time_dim=self.config.time_dim,
                ctx_dim=self.config.ctx_dim,
                temporal_dim=temporal_dim,
                d_gru=self.config.temporal_gru_dim,
                dropout=0.0,
            ).to(self.device)

        if scheduler is not None:
            self.scheduler = scheduler.to(self.device)
        else:
            self.scheduler = NoiseScheduler(self.config.num_timesteps).to(self.device)

        if self.config.temporal_gru_dim > 0:
            if temporal_encoder is not None:
                self.temporal_encoder = temporal_encoder.to(self.device)
            else:
                self.temporal_encoder = TemporalEncoder(
                    in_features=self.config.in_channels,
                    d_gru=self.config.temporal_gru_dim,
                    num_layers=self.config.temporal_layers,
                    film_dim=self.config.time_dim,
                ).to(self.device)
        else:
            self.temporal_encoder = None

        self._norm_means = None
        self._norm_stds = None
        if self.config.norm_stats_path is not None:
            self._load_norm_stats(self.config.norm_stats_path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        model_state = state["model"] if "model" in state else state
        self.model.load_state_dict(model_state)
        self.model.eval()
        if self.temporal_encoder is not None and "temporal_encoder" in state:
            self.temporal_encoder.load_state_dict(state["temporal_encoder"])
            self.temporal_encoder.eval()

    def _load_norm_stats(self, path: str) -> None:
        with open(path, "r") as f:
            stats = json.load(f)
        self._norm_means = np.array(stats["means"], dtype=np.float32)
        self._norm_stds = np.array(stats["stds"], dtype=np.float32)
        self._norm_stds = np.where(self._norm_stds < 1e-8, 1.0, self._norm_stds)

    def denormalize(self, synthetic_normed: np.ndarray) -> np.ndarray:
        """Convert normalized-space outputs back to original scale.

        Args:
            synthetic_normed: (B, L, C) or (L, C) array in normalized space.

        Returns:
            Denormalized array in original feature space.
        """
        if self._norm_means is None:
            raise ValueError("norm_stats_path not set — cannot denormalize")
        means = self._norm_means
        stds = self._norm_stds
        if synthetic_normed.ndim == 2:
            return synthetic_normed * stds[None, :] + means[None, :]
        return synthetic_normed * stds[None, None, :] + means[None, None, :]

    @torch.no_grad()
    def _encode_temporal(self, past_context: Tensor) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Run temporal encoder on past context, returning (temporal_seq, temporal_emb)."""
        if self.temporal_encoder is None:
            return None, None
        temporal_seq, temporal_emb, _ = self.temporal_encoder(past_context)
        return temporal_seq, temporal_emb

    @torch.no_grad()
    def _cfg_model_forward(self, x: Tensor, t: Tensor, context: Tensor,
                           temporal_seq: Optional[Tensor] = None,
                           temporal_emb: Optional[Tensor] = None) -> Tensor:
        """Classifier-free guidance: interpolate conditional and unconditional predictions."""
        eps_cond = self.model(x, t, context, temporal_seq=temporal_seq, temporal_emb=temporal_emb)
        zero_ctx = torch.zeros_like(context)
        zero_temp_emb = torch.zeros_like(temporal_emb) if temporal_emb is not None else None
        zero_temp_seq = torch.zeros_like(temporal_seq) if temporal_seq is not None else None
        eps_uncond = self.model(x, t, zero_ctx, temporal_seq=zero_temp_seq, temporal_emb=zero_temp_emb)
        w = self.config.guidance_scale
        return eps_uncond + w * (eps_cond - eps_uncond)

    def generate_paths(
        self,
        world_state: WorldState | Mapping[str, Any],
        num_paths: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        past_context: Optional[Tensor] = None,
        regime_probs: Optional[Tensor] = None,  # V26 Phase 1: backward compat
    ) -> list[dict[str, Any]]:
        """Generate multiple future market paths conditioned on world state.

        Args:
        world_state: Current market state for conditioning.
        num_paths: Number of paths (default from config).
        steps: DDIM sampling steps (default from config).
        guidance_scale: CFG strength (default from config).
        past_context: Optional (B, context_len, C) tensor of past bars for temporal encoding.
        regime_probs: Optional (B, 9) or (9,) regime probability vector (V26, ignored in V24).

        Returns:
        List of path dicts with data, confidence, and metadata.
        """
        n = num_paths or self.config.num_paths
        s = steps or self.config.sampling_steps
        if guidance_scale is not None:
            self.config.guidance_scale = guidance_scale

        context = self._world_state_to_context(world_state)
        context_batch = context.unsqueeze(0).expand(n, -1).to(self.device)

        temporal_seq = None
        temporal_emb = None
        if self.temporal_encoder is not None and past_context is not None:
            pc = past_context.to(self.device)
            if pc.dim() == 2:
                pc = pc.unsqueeze(0).expand(n, -1, -1)
            elif pc.shape[0] == 1 and n > 1:
                pc = pc.expand(n, -1, -1)
            temporal_seq, temporal_emb = self._encode_temporal(pc)

        self.model.eval()
        if self.temporal_encoder is not None:
            self.temporal_encoder.eval()
        shape = (n, self.config.in_channels, self.config.sequence_length)

        paths_tensor = self._sample_with_cfg(shape, context_batch, s, temporal_seq, temporal_emb)

        paths_tensor = paths_tensor.cpu().float()
        paths_np = paths_tensor.permute(0, 2, 1).numpy()

        confidence = self._compute_learned_confidence(paths_np)

        paths = []
        for i in range(n):
            paths.append({
                "path_id": i,
                "data": paths_np[i].tolist(),
                "confidence": float(confidence[i]),
                "metadata": {
                    "generator": "diffusion_unet1d_v2",
                    "path_length": self.config.sequence_length,
                    "features": self.config.in_channels,
                    "guidance_scale": self.config.guidance_scale,
                    "sampling_steps": s,
                    "sampling_method": "ddim",
                    "temporal_conditioning": self.config.temporal_gru_dim > 0,
                },
            })
        return paths

    @torch.no_grad()
    def _sample_with_cfg(self, shape: tuple, context: Tensor, num_steps: int,
                         temporal_seq: Optional[Tensor] = None,
                         temporal_emb: Optional[Tensor] = None) -> Tensor:
        """DDIM sampling with classifier-free guidance."""
        device = self.device
        step_size = self.scheduler.num_timesteps // num_steps
        timesteps = list(reversed(range(0, self.scheduler.num_timesteps, step_size)))

        x = torch.randn(shape, device=device)
        for i, t_val in enumerate(timesteps):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            eps_pred = self._cfg_model_forward(x, t, context, temporal_seq, temporal_emb)
            s1 = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t, x.shape)
            s2 = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape)
            x0_pred = (x - s2 * eps_pred) / s1
            x0_pred = torch.clamp(x0_pred, -2.0, 2.0)

            if i < len(timesteps) - 1:
                t_next_val = timesteps[i + 1]
                t_next = torch.full((shape[0],), t_next_val, device=device, dtype=torch.long)
                s1_n = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t_next, x.shape)
                s2_n = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_next, x.shape)
                direction = torch.sqrt(1 - s1_n ** 2) * eps_pred
                x = s1_n * x0_pred + direction
        return x

    def _world_state_to_context(self, world_state: WorldState | Mapping[str, Any]) -> Tensor:
        """Convert WorldState to a context vector for the diffusion model."""
        if isinstance(world_state, WorldState):
            flat = world_state.to_flat_features()
            values = list(flat.values())[:self.config.ctx_dim]
        elif isinstance(world_state, dict):
            values = []
            for k, v in sorted(world_state.items())[:self.config.ctx_dim]:
                if isinstance(v, (int, float)):
                    values.append(float(v))
                elif isinstance(v, str):
                    values.append(hash(v) % 1000 / 1000.0)
                else:
                    values.append(0.0)
        else:
            values = [0.0] * self.config.ctx_dim

        while len(values) < self.config.ctx_dim:
            values.append(0.0)
        values = values[:self.config.ctx_dim]
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _compute_learned_confidence(paths: np.ndarray) -> np.ndarray:
        """Compute per-path confidence from ensemble variance.

        High variance across the path indicates uncertainty → lower confidence.
        Low variance indicates agreement → higher confidence.
        """
        n = paths.shape[0]
        path_vars = np.var(paths, axis=(1, 2))
        max_var = path_vars.max() + 1e-8
        confidences = 1.0 - (path_vars / (2.0 * max_var))
        confidences = np.clip(confidences, 0.1, 0.99)
        return confidences.astype(np.float32)
