from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from config.project_config import FUSED_FEATURE_MATRIX_PATH, PRICE_FEATURE_COLUMNS, TARGETS_MULTIHORIZON_PATH, TRAIN_YEARS


RETRIEVAL_FEATURES: tuple[str, ...] = (
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "macd_hist",
    "ema_cross",
    "atr_pct",
    "bb_pct",
    "body_pct",
    "volume_ratio",
)


@dataclass(frozen=True)
class HistoricalRetrievalResult:
    support: int
    similarity: float
    directional_prior: float
    hold_prior_15m: float
    hold_prior_30m: float


class HistoricalPathRetriever:
    def __init__(
        self,
        feature_path: Path = FUSED_FEATURE_MATRIX_PATH,
        target_bundle_path: Path = TARGETS_MULTIHORIZON_PATH,
        *,
        sample_stride: int = 60,
        max_samples: int = 50000,
        top_k: int = 64,
    ) -> None:
        if not feature_path.exists() or not target_bundle_path.exists():
            raise FileNotFoundError("Historical path retriever requires fused features and multi-horizon targets.")
        feature_memmap = np.load(feature_path, mmap_mode="r")
        bundle = np.load(target_bundle_path, mmap_mode="r")
        feature_lookup = {name: idx for idx, name in enumerate(PRICE_FEATURE_COLUMNS)}
        self.feature_indices = np.asarray([feature_lookup[name] for name in RETRIEVAL_FEATURES], dtype=np.int64)
        sampled_idx = np.arange(0, len(feature_memmap), max(1, int(sample_stride)), dtype=np.int64)
        sampled_idx = sampled_idx[:max_samples]
        self.features = np.asarray(feature_memmap[sampled_idx][:, self.feature_indices], dtype=np.float32)
        self.direction_15m = np.asarray(bundle["target_15m"][sampled_idx], dtype=np.float32)
        self.direction_30m = np.asarray(bundle["target_30m"][sampled_idx], dtype=np.float32)
        self.hold_15m = np.asarray(bundle["hold_15m"][sampled_idx], dtype=np.float32)
        self.hold_30m = np.asarray(bundle["hold_30m"][sampled_idx], dtype=np.float32)
        self.top_k = max(4, int(top_k))
        self.mean = self.features.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = self.features.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        self.features = ((self.features - self.mean) / self.std).astype(np.float32)

    def _row_vector(self, current_row: Mapping[str, float]) -> np.ndarray:
        values = np.asarray([float(current_row.get(name, 0.0) or 0.0) for name in RETRIEVAL_FEATURES], dtype=np.float32)
        return ((values - self.mean[0]) / self.std[0]).astype(np.float32)

    def retrieve(self, current_row: Mapping[str, float]) -> HistoricalRetrievalResult:
        if len(self.features) == 0:
            return HistoricalRetrievalResult(0, 0.0, 0.0, 0.0, 0.0)
        row = self._row_vector(current_row)
        distances = np.sqrt(np.mean((self.features - row) ** 2, axis=1))
        top_k = min(self.top_k, len(distances))
        nearest_idx = np.argpartition(distances, top_k - 1)[:top_k]
        nearest_distances = distances[nearest_idx]
        weights = np.exp(-nearest_distances / max(0.25, float(np.median(nearest_distances) + 1e-6)))
        weights = weights / max(float(weights.sum()), 1e-6)
        directional_prior = float(np.sum(weights * (((self.direction_15m[nearest_idx] + self.direction_30m[nearest_idx]) / 2.0) - 0.5) * 2.0))
        hold_prior_15m = float(np.sum(weights * self.hold_15m[nearest_idx]))
        hold_prior_30m = float(np.sum(weights * self.hold_30m[nearest_idx]))
        similarity = float(1.0 / (1.0 + np.mean(nearest_distances)))
        return HistoricalRetrievalResult(
            support=int(top_k),
            similarity=round(similarity, 6),
            directional_prior=round(directional_prior, 6),
            hold_prior_15m=round(hold_prior_15m, 6),
            hold_prior_30m=round(hold_prior_30m, 6),
        )


@lru_cache(maxsize=1)
def get_historical_path_retriever() -> HistoricalPathRetriever:
    return HistoricalPathRetriever()
