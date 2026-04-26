"""Regime-conditioned Diffusion Path Generator for V26 Phase 1.

Extends DiffusionPathGeneratorV2 with regime conditioning via:
1. RegimeEmbedding module for regime probability vectors
2. Combined FiLM conditioning (temporal + regime) in U-Net
3. Cross-attention injection of regime embeddings in decoder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor

from src.v24.diffusion.unet_1d import DiffusionUNet1D
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.temporal_encoder import TemporalEncoder
from src.v24.world_state import WorldState
from src.v26.diffusion.regime_embedding import RegimeEmbedding


@dataclass
class RegimeGeneratorConfig:
    """Configuration for regime-conditioned path generator."""

    # Base generator config
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

    # Regime-specific config
    num_regimes: int = 9
    regime_embed_dim: int = 16
    regime_conditioning_strength: float = 1.0

    # FiLM dimension (temporal_gru + regime_embed)
    # Set to temporal_gru_dim + regime_embed_dim by default
    temporal_film_dim: int = 0  # 0 means auto-compute


class RegimeDiffusionPathGenerator:
    """Regime-conditioned diffusion path generator.

    Extends V24.5 temporal conditioning with regime-aware generation:
    - RegimeEmbedding: converts (9,) probability vectors to (16,) embeddings
    - Combined FiLM: temporal_emb and regime_emb concatenated for U-Net ResBlocks
    - Cross-attention: regime embeddings available in decoder blocks

    Args:
        config: Generator configuration.
        model: Pre-trained DiffusionUNet1D model with regime_dim support.
        scheduler: Noise scheduler.
        temporal_encoder: Optional TemporalEncoder for temporal conditioning.
        regime_embedder: Optional RegimeEmbedding module.
        device: Device for computation.
    """

    def __init__(
        self,
        config: Optional[RegimeGeneratorConfig] = None,
        model: Optional[DiffusionUNet1D] = None,
        scheduler: Optional[NoiseScheduler] = None,
        temporal_encoder: Optional[TemporalEncoder] = None,
        regime_embedder: Optional[RegimeEmbedding] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or RegimeGeneratorConfig()
        self.device = torch.device(device)

        # Determine temporal dimension for FiLM (temporal_gru + regime_embed)
        # This should equal what Phase 1 was trained with: 272
        if self.config.temporal_film_dim > 0:
            temporal_film_dim = self.config.temporal_film_dim
        else:
            temporal_film_dim = self.config.temporal_gru_dim + self.config.regime_embed_dim

        # Determine regime dimension - set to 0 since regime is already included in temporal_dim
        # (The FiLM layer combines temporal + regime, so we don't pass regime separately)
        regime_dim = 0  # Regime already baked into temporal_film_dim

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = DiffusionUNet1D(
                in_channels=self.config.in_channels,
                base_channels=self.config.base_channels,
                channel_multipliers=self.config.channel_multipliers,
                time_dim=self.config.time_dim,
                ctx_dim=self.config.ctx_dim,
                temporal_dim=temporal_film_dim,  # 272 = 256 + 16
                d_gru=self.config.temporal_gru_dim,
                regime_dim=regime_dim,  # 0 - regime already in temporal_film_dim
                dropout=self.config.dropout,
            ).to(self.device)

        if scheduler is not None:
            self.scheduler = scheduler.to(self.device)
        else:
            self.scheduler = NoiseScheduler(self.config.num_timesteps).to(self.device)

        # Temporal encoder
        if self.config.temporal_gru_dim > 0:
            if temporal_encoder is not None:
                self.temporal_encoder = temporal_encoder.to(self.device)
            else:
                self.temporal_encoder = TemporalEncoder(
                    in_features=self.config.in_channels,
                    d_gru=self.config.temporal_gru_dim,
                    num_layers=self.config.temporal_layers,
                    film_dim=temporal_film_dim,
                ).to(self.device)
        else:
            self.temporal_encoder = None

        # Regime embedder
        if regime_embedder is not None:
            self.regime_embedder = regime_embedder.to(self.device)
        else:
            self.regime_embedder = RegimeEmbedding(
                num_regimes=self.config.num_regimes,
                embed_dim=self.config.regime_embed_dim,
                use_learned_embedding=False,
            ).to(self.device)
        
        # Re-encode regime from Phase 1 checkpoint
        self._regime_embedder_bias = self.config.regime_embed_dim

        self._norm_means = None
        self._norm_stds = None
        if self.config.norm_stats_path is not None:
            self._load_norm_stats(self.config.norm_stats_path)

    def load_checkpoint(self, path: str, strict: bool = False) -> dict[str, Any]:
        """Load checkpoint with regime embedding support.

        Can load V24 Phase 0.7 checkpoints (without regime) by ignoring
        regime-specific parameters.

        Args:
            path: Path to checkpoint file.
            strict: If False, ignore regime parameters not in checkpoint.

        Returns:
            Dict with loading info (loaded keys, missing keys, etc.).
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        model_state = state.get("model", state)

        # Filter state dict for backward compatibility
        model_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(model_state.keys())

        # Remove regime-specific keys if not present in checkpoint (V24 backward compat)
        regime_keys = {k for k in checkpoint_keys if "regime" in k.lower()}
        model_regime_keys = {k for k in model_keys if "regime" in k.lower()}

        if not regime_keys and model_regime_keys:
            # Checkpoint has no regime keys but model expects them
            # Initialize regime parameters fresh
            compatible_state = {k: v for k, v in model_state.items() if k in model_keys}
            self.model.load_state_dict(compatible_state, strict=False)
            self.model.eval()
            info = {
                "loaded_keys": len(compatible_state),
                "missing_keys": len(model_keys - checkpoint_keys),
                "regime_initialized_fresh": len(model_regime_keys),
            }
        else:
            self.model.load_state_dict(model_state, strict=strict)
            self.model.eval()
            info = {
                "loaded_keys": len(model_state),
                "missing_keys": len(model_keys - checkpoint_keys),
                "regime_initialized_fresh": 0,
            }

        # Load temporal encoder if present
        if self.temporal_encoder is not None and "temporal_encoder" in state:
            self.temporal_encoder.load_state_dict(state["temporal_encoder"])
            self.temporal_encoder.eval()
            info["temporal_encoder_loaded"] = True
        else:
            info["temporal_encoder_loaded"] = False

        # Load regime embedder if present
        if "regime_embedder" in state:
            self.regime_embedder.load_state_dict(state["regime_embedder"])
            self.regime_embedder.eval()
            info["regime_embedder_loaded"] = True
        else:
            info["regime_embedder_loaded"] = False

        return info

    def _load_norm_stats(self, path: str) -> None:
        import json

        with open(path, "r") as f:
            stats = json.load(f)
        self._norm_means = np.array(stats["means"], dtype=np.float32)
        self._norm_stds = np.array(stats["stds"], dtype=np.float32)
        self._norm_stds = np.where(self._norm_stds < 1e-8, 1.0, self._norm_stds)

    def denormalize(self, synthetic_normed: np.ndarray) -> np.ndarray:
        """Convert normalized-space outputs back to original scale."""
        if self._norm_means is None:
            raise ValueError("norm_stats_path not set — cannot denormalize")
        means = self._norm_means
        stds = self._norm_stds
        if synthetic_normed.ndim == 2:
            return synthetic_normed * stds[None, :] + means[None, :]
        return synthetic_normed * stds[None, None, :] + means[None, None, :]

    @torch.no_grad()
    def _encode_temporal(self, past_context: Tensor) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Run temporal encoder on past context."""
        if self.temporal_encoder is None:
            return None, None
        temporal_seq, temporal_emb, _ = self.temporal_encoder(past_context)
        return temporal_seq, temporal_emb

    @torch.no_grad()
    def _encode_regime(self, regime_probs: Tensor) -> Tensor:
        """Convert regime probabilities to embedding."""
        return self.regime_embedder(regime_probs)

    @torch.no_grad()
    def _cfg_model_forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor,
        temporal_seq: Optional[Tensor] = None,
        temporal_emb: Optional[Tensor] = None,
        regime_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """Classifier-free guidance with temporal and regime conditioning."""
        eps_cond = self.model(
            x,
            t,
            context,
            temporal_seq=temporal_seq,
            temporal_emb=temporal_emb,
            regime_emb=regime_emb,
        )
        zero_ctx = torch.zeros_like(context)
        zero_temp_emb = torch.zeros_like(temporal_emb) if temporal_emb is not None else None
        zero_temp_seq = torch.zeros_like(temporal_seq) if temporal_seq is not None else None
        zero_regime_emb = torch.zeros_like(regime_emb) if regime_emb is not None else None

        eps_uncond = self.model(
            x,
            t,
            zero_ctx,
            temporal_seq=zero_temp_seq,
            temporal_emb=zero_temp_emb,
            regime_emb=zero_regime_emb,
        )
        w = self.config.guidance_scale
        return eps_uncond + w * (eps_cond - eps_uncond)

    def generate_paths(
        self,
        world_state: WorldState | Mapping[str, Any],
        regime_probs: Optional[Tensor] = None,
        num_paths: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        past_context: Optional[Tensor] = None,
    ) -> list[dict[str, Any]]:
        """Generate future paths conditioned on world state and regime.

        Args:
            world_state: Current market state for conditioning.
            regime_probs: Optional (B, 9) or (9,) regime probability vector.
                If None, uses uniform distribution over regimes.
            num_paths: Number of paths (default from config).
            steps: DDIM sampling steps (default from config).
            guidance_scale: CFG strength (default from config).
            past_context: Optional (B, context_len, C) past bars for temporal encoding.

        Returns:
            List of path dicts with data, confidence, regime info, and metadata.
        """
        n = num_paths or self.config.num_paths
        s = steps or self.config.sampling_steps
        if guidance_scale is not None:
            self.config.guidance_scale = guidance_scale

        context = self._world_state_to_context(world_state)
        context_batch = context.unsqueeze(0).expand(n, -1).to(self.device)

        # Encode temporal conditioning
        temporal_seq = None
        temporal_emb = None
        if self.temporal_encoder is not None and past_context is not None:
            pc = past_context.to(self.device)
            if pc.dim() == 2:
                pc = pc.unsqueeze(0).expand(n, -1, -1)
            elif pc.shape[0] == 1 and n > 1:
                pc = pc.expand(n, -1, -1)
            temporal_seq, temporal_emb = self._encode_temporal(pc)

        # Encode regime conditioning
        regime_emb = None
        if regime_probs is not None:
            if regime_probs.dim() == 1:
                regime_probs = regime_probs.unsqueeze(0).expand(n, -1)
            elif regime_probs.shape[0] == 1 and n > 1:
                regime_probs = regime_probs.expand(n, -1)
            regime_emb = self._encode_regime(regime_probs.to(self.device))
        else:
            # Use uniform regime distribution
            uniform_probs = torch.ones(n, self.config.num_regimes, device=self.device)
            uniform_probs = uniform_probs / self.config.num_regimes
            regime_emb = self._encode_regime(uniform_probs)

        self.model.eval()
        if self.temporal_encoder is not None:
            self.temporal_encoder.eval()
        if self.regime_embedder is not None:
            self.regime_embedder.eval()

        shape = (n, self.config.in_channels, self.config.sequence_length)
        paths_tensor = self._sample_with_cfg(
            shape, context_batch, s, temporal_seq, temporal_emb, regime_emb
        )

        paths_tensor = paths_tensor.cpu().float()
        paths_np = paths_tensor.permute(0, 2, 1).numpy()

        confidence = self._compute_learned_confidence(paths_np)

        paths = []
        for i in range(n):
            regime_info = {
                "embedding": regime_emb[i].cpu().numpy().tolist() if regime_emb is not None else None,
                "conditioned": regime_probs is not None,
            }
            paths.append({
                "path_id": i,
                "data": paths_np[i].tolist(),
                "confidence": float(confidence[i]),
                "regime": regime_info,
                "metadata": {
                    "generator": "regime_diffusion_unet_v26",
                    "path_length": self.config.sequence_length,
                    "features": self.config.in_channels,
                    "guidance_scale": self.config.guidance_scale,
                    "sampling_steps": s,
                    "sampling_method": "ddim_regime_cfg",
                    "temporal_conditioning": self.config.temporal_gru_dim > 0,
                    "regime_conditioning": True,
                    "regime_embed_dim": self.config.regime_embed_dim,
                },
            })
        return paths

    @torch.no_grad()
    def _sample_with_cfg(
        self,
        shape: tuple,
        context: Tensor,
        num_steps: int,
        temporal_seq: Optional[Tensor] = None,
        temporal_emb: Optional[Tensor] = None,
        regime_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """DDIM sampling with classifier-free guidance and regime conditioning."""
        device = self.device
        step_size = self.scheduler.num_timesteps // num_steps
        timesteps = list(reversed(range(0, self.scheduler.num_timesteps, step_size)))

        x = torch.randn(shape, device=device)
        for i, t_val in enumerate(timesteps):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            eps_pred = self._cfg_model_forward(
                x, t, context, temporal_seq, temporal_emb, regime_emb
            )
            s1 = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t, x.shape)
            s2 = self.scheduler._extract(
                self.scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            x0_pred = (x - s2 * eps_pred) / s1
            x0_pred = torch.clamp(x0_pred, -2.0, 2.0)

            if i < len(timesteps) - 1:
                t_next_val = timesteps[i + 1]
                t_next = torch.full((shape[0],), t_next_val, device=device, dtype=torch.long)
                s1_n = self.scheduler._extract(
                    self.scheduler.sqrt_alphas_cumprod, t_next, x.shape
                )
                s2_n = self.scheduler._extract(
                    self.scheduler.sqrt_one_minus_alphas_cumprod, t_next, x.shape
                )
                direction = torch.sqrt(1 - s1_n ** 2) * eps_pred
                x = s1_n * x0_pred + direction
        return x

    def _world_state_to_context(self, world_state: WorldState | Mapping[str, Any]) -> Tensor:
        """Convert WorldState to a context vector for the diffusion model."""
        if isinstance(world_state, WorldState):
            flat = world_state.to_flat_features()
            values = list(flat.values())[: self.config.ctx_dim]
        elif isinstance(world_state, dict):
            values = []
            for k, v in sorted(world_state.items())[: self.config.ctx_dim]:
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
        values = values[: self.config.ctx_dim]
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _compute_learned_confidence(paths: np.ndarray) -> np.ndarray:
        """Compute per-path confidence from ensemble variance."""
        n = paths.shape[0]
        path_vars = np.var(paths, axis=(1, 2))
        max_var = path_vars.max() + 1e-8
        confidences = 1.0 - (path_vars / (2.0 * max_var))
        confidences = np.clip(confidences, 0.1, 0.99)
        return confidences.astype(np.float32)
