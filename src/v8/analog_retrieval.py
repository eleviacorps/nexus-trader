from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.project_config import FUSED_FEATURE_MATRIX_PATH, FUSED_TIMESTAMPS_PATH, TARGETS_MULTIHORIZON_PATH


@dataclass(frozen=True)
class AnalogRetrievalResult:
    support: int
    similarity: float
    disagreement: float
    directional_prior: float
    average_path: list[float]


@dataclass(frozen=True)
class AnalogRetrievalCache:
    features: np.ndarray
    future_paths: np.ndarray
    timestamps: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    window_size: int
    sample_stride: int


def _build_window_vectors(features: np.ndarray, window_size: int, sample_stride: int, path_15m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = []
    futures = []
    positions = []
    limit = min(len(features), len(path_15m))
    for end in range(window_size - 1, limit, max(1, sample_stride)):
        start = end - window_size + 1
        window = features[start : end + 1].reshape(-1)
        future = path_15m[end]
        if not np.isfinite(window).all() or not np.isfinite(future).all():
            continue
        windows.append(window.astype(np.float32))
        futures.append(future.astype(np.float32))
        positions.append(end)
    if not windows:
        return np.zeros((0, window_size * features.shape[1]), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.vstack(windows), np.vstack(futures), np.asarray(positions, dtype=np.int64)


def build_analog_cache(
    feature_path: Path = FUSED_FEATURE_MATRIX_PATH,
    targets_path: Path = TARGETS_MULTIHORIZON_PATH,
    timestamps_path: Path = FUSED_TIMESTAMPS_PATH,
    *,
    window_size: int = 24,
    sample_stride: int = 30,
) -> AnalogRetrievalCache:
    features = np.load(feature_path, mmap_mode="r")
    bundle = np.load(targets_path, mmap_mode="r")
    timestamps = np.load(timestamps_path, mmap_mode="r")
    future_path = np.column_stack(
        [
            bundle["forward_return_5m"][: len(features)],
            bundle["forward_return_10m"][: len(features)],
            bundle["forward_return_15m"][: len(features)],
        ]
    ).astype(np.float32)
    window_vectors, future_paths, positions = _build_window_vectors(
        np.asarray(features[:, :36], dtype=np.float32),
        window_size=window_size,
        sample_stride=sample_stride,
        path_15m=future_path,
    )
    mean = window_vectors.mean(axis=0, keepdims=True).astype(np.float32) if len(window_vectors) else np.zeros((1, 36 * window_size), dtype=np.float32)
    std = window_vectors.std(axis=0, keepdims=True).astype(np.float32) if len(window_vectors) else np.ones((1, 36 * window_size), dtype=np.float32)
    std[std < 1e-6] = 1.0
    normed = ((window_vectors - mean) / std).astype(np.float32) if len(window_vectors) else window_vectors
    timestamp_rows = np.asarray(timestamps[positions], dtype="datetime64[ns]") if len(positions) else np.zeros((0,), dtype="datetime64[ns]")
    return AnalogRetrievalCache(
        features=normed,
        future_paths=future_paths,
        timestamps=timestamp_rows,
        feature_mean=mean[0],
        feature_std=std[0],
        window_size=window_size,
        sample_stride=sample_stride,
    )


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    len_a = len(a)
    len_b = len(b)
    cost = np.full((len_a + 1, len_b + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            dist = abs(float(a[i - 1] - b[j - 1]))
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[len_a, len_b] / max(len_a, len_b))


def retrieve_analogs(cache: AnalogRetrievalCache, window_features: np.ndarray, *, top_k: int = 32) -> AnalogRetrievalResult:
    if cache.features.size == 0:
        return AnalogRetrievalResult(0, 0.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    vector = np.asarray(window_features, dtype=np.float32).reshape(-1)
    vector = ((vector - cache.feature_mean) / cache.feature_std).astype(np.float32)
    cosine = cache.features @ vector / (np.linalg.norm(cache.features, axis=1) * max(np.linalg.norm(vector), 1e-6))
    cosine = np.nan_to_num(cosine, nan=0.0)
    dtw_scores = np.asarray([_dtw_distance(vector[-cache.window_size :], row[-cache.window_size :]) for row in cache.features[: min(len(cache.features), 5000)]], dtype=np.float32)
    dtw_mean = float(dtw_scores.mean()) if len(dtw_scores) else 1.0
    combined = cosine.copy()
    combined[: len(dtw_scores)] -= (dtw_scores / max(dtw_mean, 1e-6)) * 0.15
    top_k = min(max(1, top_k), len(combined))
    idx = np.argpartition(-combined, top_k - 1)[:top_k]
    scores = np.clip(combined[idx], -1.0, 1.0)
    weights = np.exp(scores - scores.max())
    weights = weights / max(float(weights.sum()), 1e-6)
    future_paths = cache.future_paths[idx]
    average_path = np.average(future_paths, axis=0, weights=weights).astype(np.float32)
    disagreement = float(np.average(np.mean(np.abs(future_paths - average_path), axis=1), weights=weights))
    directional_prior = float(np.average(np.sign(future_paths[:, -1]), weights=weights))
    similarity = float(np.average(np.clip(scores, 0.0, 1.0), weights=weights))
    return AnalogRetrievalResult(
        support=int(top_k),
        similarity=round(similarity, 6),
        disagreement=round(disagreement, 6),
        directional_prior=round(directional_prior, 6),
        average_path=[round(float(value), 6) for value in average_path.tolist()],
    )
