from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from config.project_config import FEATURE_DIM_CROWD, FEATURE_DIM_NEWS, FEATURE_DIM_PRICE, FEATURE_DIM_TOTAL
from src.utils.alignment import concatenate_feature_blocks

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from src.models.nexus_tft import NexusTFT, NexusTFTConfig


@dataclass(frozen=True)
class FeatureSlices:
    price: slice = field(default_factory=lambda: slice(0, FEATURE_DIM_PRICE))
    news: slice = field(default_factory=lambda: slice(FEATURE_DIM_PRICE, FEATURE_DIM_PRICE + FEATURE_DIM_NEWS))
    crowd: slice = field(default_factory=lambda: slice(FEATURE_DIM_PRICE + FEATURE_DIM_NEWS, FEATURE_DIM_TOTAL))


def fuse_feature_rows(
    price_rows: Sequence[Sequence[float]],
    news_rows: Sequence[Sequence[float]],
    crowd_rows: Sequence[Sequence[float]],
) -> List[List[float]]:
    fused = concatenate_feature_blocks(price_rows, news_rows, crowd_rows)
    for row in fused:
        if len(row) != FEATURE_DIM_TOTAL:
            raise ValueError(f"Expected fused feature width {FEATURE_DIM_TOTAL}, got {len(row)}")
    return fused


if nn is not None:
    class FusedTFT(nn.Module):
        def __init__(self, config: NexusTFTConfig | None = None):
            super().__init__()
            self.feature_slices = FeatureSlices()
            self.news_adapter = nn.Sequential(
                nn.Linear(FEATURE_DIM_NEWS, FEATURE_DIM_NEWS),
                nn.GELU(),
                nn.LayerNorm(FEATURE_DIM_NEWS),
            )
            self.crowd_adapter = nn.Sequential(
                nn.Linear(FEATURE_DIM_CROWD, FEATURE_DIM_CROWD),
                nn.GELU(),
                nn.LayerNorm(FEATURE_DIM_CROWD),
            )
            self.core = NexusTFT(config or NexusTFTConfig(input_dim=FEATURE_DIM_TOTAL))

        def forward(self, price, news, crowd, return_feature_importance: bool = False):
            fused = torch.cat([price, self.news_adapter(news), self.crowd_adapter(crowd)], dim=-1)
            return self.core(fused, return_feature_importance=return_feature_importance)

        def optimizer_groups(self, old_layers_lr: float = 1e-4, new_layers_lr: float = 5e-4):
            return [
                {"params": list(self.core.parameters()), "lr": old_layers_lr},
                {"params": list(self.news_adapter.parameters()) + list(self.crowd_adapter.parameters()), "lr": new_layers_lr},
            ]
else:  # pragma: no cover
    class FusedTFT:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for FusedTFT.")
