"""V24 Diffusion Sub-Package — 1D U-Net conditional diffusion model.

Phase 0.5: Adds TemporalEncoder (GRU + FiLM) for temporal conditioning,
tighter x0_pred clamp [-3,3], and 6M × 144 feature support.
"""

from __future__ import annotations

from src.v24.diffusion.unet_1d import DiffusionUNet1D, ResBlock1d, FiLMConditioning
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.dataset import DiffusionDataset
from src.v24.diffusion.generator import DiffusionPathGeneratorV2, GeneratorConfig
from src.v24.diffusion.temporal_encoder import TemporalEncoder

__all__ = [
    "DiffusionDataset",
    "DiffusionPathGeneratorV2",
    "DiffusionUNet1D",
    "FiLMConditioning",
    "GeneratorConfig",
    "NoiseScheduler",
    "ResBlock1d",
    "TemporalEncoder",
]
