"""Full MMFPS generator - integrates temporal + regime + quant + diffusion.

ONLY the diffusion generator - 345M params.
No selector - train this first.
"""

from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve().parents[1]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple, Optional

from .regime.regime_detector import RegimeDetector, RegimeState
from .quant.quant_features import QuantFeatureExtractor
from .generator.diffusion_path_generator import DiffusionPathGenerator, DiffusionGeneratorConfig


class GeneratorOutput(NamedTuple):
    paths: Tensor
    regime_probs: tuple
    regime_embedding: Tensor
    regime_type: Tensor
    temporal_embedding: Tensor
    quant_embedding: Tensor
    diversity_loss: Tensor


class MMFPSGeneratorConfig:
    """Configuration for MMFPS generator only."""
    
    # Feature dimensions
    feature_dim: int = 144
    path_horizon: int = 20
    
    # Generator
    base_channels: int = 256
    channel_multipliers: tuple = (1, 2, 4, 8)
    time_dim: int = 512
    regime_dim: int = 64
    quant_dim: int = 64
    temporal_dim: int = 256
    
    # Generator settings
    num_paths: int = 128
    sampling_steps: int = 50


class TemporalEncoder(nn.Module):
    """GRU temporal encoder with FiLM output for conditioning."""
    
    def __init__(
        self,
        in_features: int = 144,
        d_gru: int = 256,
        num_layers: int = 2,
        film_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.d_gru = d_gru
        self.film_dim = film_dim
        
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=d_gru,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.film_proj = nn.Sequential(
            nn.Linear(d_gru, film_dim),
            nn.SiLU(),
            nn.Linear(film_dim, film_dim),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Process temporal sequence -> FiLM embedding."""
        outputs, hidden = self.gru(x)
        final_hidden = hidden[-1]
        return self.film_proj(final_hidden)


class MMFPSGenerator(nn.Module):
    """Full MMFPS diffusion generator - 345M params."""
    
    def __init__(self, config: Optional[MMFPSGeneratorConfig] = None):
        super().__init__()
        
        self.config = config or MMFPSGeneratorConfig()
        cfg = self.config
        
        # 1. Temporal Encoder
        self.temporal_encoder = TemporalEncoder(
            in_features=cfg.feature_dim,
            d_gru=cfg.temporal_dim,
            num_layers=2,
            film_dim=cfg.time_dim,
        )
        
        # 2. Regime Detector
        self.regime_detector = RegimeDetector(
            feature_dim=cfg.feature_dim,
            embed_dim=cfg.regime_dim,
        )
        
        # 3. Quant Feature Extractor
        self.quant_extractor = QuantFeatureExtractor(
            path_dim=cfg.path_horizon,
            embed_dim=cfg.quant_dim,
        )
        
        # 4. Diffusion Generator (345M params)
        generator_config = DiffusionGeneratorConfig(
            in_channels=cfg.feature_dim,
            horizon=cfg.path_horizon,
            base_channels=cfg.base_channels,
            channel_multipliers=cfg.channel_multipliers,
            time_dim=cfg.time_dim,
            regime_dim=cfg.regime_dim,
            quant_dim=cfg.quant_dim,
            num_paths=cfg.num_paths,
            sampling_steps=cfg.sampling_steps,
        )
        self.generator = DiffusionPathGenerator(generator_config)
    
    def forward(
        self,
        context: Tensor,
        past_sequence: Optional[Tensor] = None,
        target_paths: Optional[Tensor] = None,
    ) -> GeneratorOutput:
        """Generate paths.
        
        Args:
            context: (B, feature_dim) - current context
            past_sequence: (B, T, feature_dim) - historical sequence
            target_paths: (B, 1, horizon, channels) - for training
        """
        B = context.shape[0]
        device = context.device
        
        # 1. Temporal encoding
        if past_sequence is not None:
            temporal_emb = self.temporal_encoder(past_sequence)
        else:
            temporal_emb = torch.zeros(B, self.config.time_dim, device=device)
        
        # 2. Regime detection
        regime_state = self.regime_detector(context)
        regime_emb = regime_state.regime_embedding
        
        # 3. Quant features
        if target_paths is not None:
            quant_emb = self.quant_extractor(target_paths.squeeze(1))
        else:
            quant_emb = torch.zeros(B, self.config.quant_dim, device=device)
        
        # 4. Generate paths
        gen_output = self.generator(
            context=context,
            regime_emb=regime_emb,
            quant_emb=quant_emb,
            targets=target_paths,
        )
        
        return GeneratorOutput(
            paths=gen_output.paths,
            regime_probs=(
                regime_state.prob_trend_up,
                regime_state.prob_chop,
                regime_state.prob_reversal,
            ),
            regime_embedding=regime_emb,
            regime_type=regime_state.regime_type,
            temporal_embedding=temporal_emb,
            quant_embedding=quant_emb,
            diversity_loss=gen_output.diversity_loss,
        )
    
    @torch.no_grad()
    def generate(
        self,
        context: Tensor,
        past_sequence: Optional[Tensor] = None,
        num_paths: int = 64,
    ) -> Tensor:
        """Generate paths only."""
        B = context.shape[0]
        device = context.device
        
        if past_sequence is not None:
            temporal_emb = self.temporal_encoder(past_sequence)
        else:
            temporal_emb = torch.zeros(B, self.config.time_dim, device=device)
        
        regime_state = self.regime_detector(context)
        regime_emb = regime_state.regime_embedding
        quant_emb = torch.zeros(B, self.config.quant_dim, device=device)
        
        return self.generator.quick_generate(
            context=context,
            regime_emb=regime_emb,
            quant_emb=quant_emb,
            num_paths=num_paths,
        )


def count_parameters(model: nn.Module) -> dict:
    """Count parameters by component."""
    counts = {}
    for name, module in model.named_children():
        counts[name] = sum(p.numel() for p in module.parameters())
    counts["total"] = sum(p.numel() for p in model.parameters())
    return counts


if __name__ == "__main__":
    config = MMFPSGeneratorConfig()
    model = MMFPSGenerator(config)
    
    counts = count_parameters(model)
    for name, count in counts.items():
        print(f"{name}: {count:,}")
    print(f"Total: {counts['total']:,}")