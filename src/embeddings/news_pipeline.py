from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

from config.project_config import FEATURE_DIM_NEWS, FILL_LIMIT
from src.utils.alignment import forward_fill_embeddings, zero_embedding_matrix

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class NewsProjectionHeadSpec:
    input_dim: int = 768
    hidden_dim: int = 256
    output_dim: int = FEATURE_DIM_NEWS
    dropout: float = 0.1


if nn is not None:
    class NewsProjectionHead(nn.Module):
        def __init__(self, spec: NewsProjectionHeadSpec | None = None):
            super().__init__()
            spec = spec or NewsProjectionHeadSpec()
            self.network = nn.Sequential(
                nn.Linear(spec.input_dim, spec.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(spec.hidden_dim),
                nn.Dropout(spec.dropout),
                nn.Linear(spec.hidden_dim, spec.output_dim),
                nn.Tanh(),
            )

        def forward(self, values):
            return self.network(values)
else:  # pragma: no cover
    class NewsProjectionHead:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for NewsProjectionHead.")


def align_news_embeddings(
    bar_times: Sequence[datetime],
    event_times: Sequence[datetime],
    event_vectors: Sequence[Sequence[float]],
    fill_limit_minutes: int = FILL_LIMIT,
) -> List[List[float]]:
    return forward_fill_embeddings(bar_times, event_times, event_vectors, FEATURE_DIM_NEWS, fill_limit_minutes)


def fallback_news_embeddings(length: int) -> List[List[float]]:
    return zero_embedding_matrix(length, FEATURE_DIM_NEWS)
