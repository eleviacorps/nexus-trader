"""V24 Diffusion Sub-Package — 1D U-Net conditional diffusion model.

Replaces the broken flat-file diffusion_model.py with a proper
epsilon-prediction DDPM pipeline: 1D U-Net, noise scheduler,
memmap dataset, and classifier-free-guidance path generator.
"""

from __future__ import annotations

from src.v24.diffusion.unet_1d import DiffusionUNet1D, ResBlock1d, FiLMConditioning
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.dataset import DiffusionDataset
from src.v24.diffusion.generator import DiffusionPathGeneratorV2

__all__ = [
    "DiffusionDataset",
    "DiffusionPathGeneratorV2",
    "DiffusionUNet1D",
    "FiLMConditioning",
    "NoiseScheduler",
    "ResBlock1d",
]
