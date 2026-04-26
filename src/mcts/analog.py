from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from config.project_config import (
    ANALOG_REPORT_PATH,
    FUSED_FEATURE_MATRIX_PATH,
    FUSED_TIMESTAMPS_PATH,
    PRICE_FEATURE_COLUMNS,
    TARGETS_PATH,
    TRAIN_YEARS,
)

ROW_ANALOG_FEATURE_NAMES: tuple[str, ...] = (
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "rsi_14",
    "rsi_7",
    "macd_hist",
    "macd",
    "macd_sig",
    "ema_cross",
    "atr_pct",
    "bb_pct",
    "body_pct",
    "upper_wick",
    "lower_wick",
    "dist_to_high",
    "dist_to_low",
    "hh",
    "ll",
    "volume_ratio",
    "session_asian",
    "session_london",
    "session_ny",
    "session_overlap",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

WINDOW_ANALOG_FEATURE_NAMES: tuple[str, ...] = (
    "return_1",
    "return_3",
    "return_6",
    "rsi_14",
    "macd_hist",
    "ema_cross",
    "atr_pct",
    "bb_pct",
    "body_pct",
    "upper_wick",
    "lower_wick",
    "dist_to_high",
    "dist_to_low",
    "volume_ratio",
)


@dataclass(frozen=True)
class AnalogScore:
    bullish_probability: float
    directional_bias: float
    confidence: float
    mean_distance: float
    support: int
    session: str
    sample_size: int
    mode: str = "regime"
    window_size: int = 24

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _resolve_feature_indices(feature_names: Sequence[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(PRICE_FEATURE_COLUMNS)}
    return np.asarray([lookup[name] for name in feature_names], dtype=np.int64)


def _session_name(row: Sequence[float]) -> str:
    asian, london, ny, overlap = [float(value) for value in row[:4]]
    if overlap >= 0.5:
        return "overlap"
    if london >= 0.5:
        return "london"
    if ny >= 0.5:
        return "new_york"
    if asian >= 0.5:
        return "asian"
    return "unknown"


class HistoricalAnalogScorer:
    def __init__(
        self,
        features_path: Path = FUSED_FEATURE_MATRIX_PATH,
        targets_path: Path = TARGETS_PATH,
        timestamps_path: Path = FUSED_TIMESTAMPS_PATH,
        train_years: Sequence[int] = TRAIN_YEARS,
        sample_stride: int = 60,
        max_samples: int = 60000,
        top_k: int = 64,
        row_feature_names: Sequence[str] = ROW_ANALOG_FEATURE_NAMES,
        window_feature_names: Sequence[str] = WINDOW_ANALOG_FEATURE_NAMES,
        window_size: int = 24,
    ) -> None:
        if not features_path.exists() or not targets_path.exists():
            raise FileNotFoundError("Historical analog scorer requires fused features and targets.")
        self.row_feature_names = tuple(row_feature_names)
        self.window_feature_names = tuple(window_feature_names)
        self.row_feature_indices = _resolve_feature_indices(self.row_feature_names)
        self.window_feature_indices = _resolve_feature_indices(self.window_feature_names)
        self.window_size = max(3, int(window_size))
        self.top_k = max(1, int(top_k))

        features_memmap = np.load(features_path, mmap_mode="r")
        targets_memmap = np.load(targets_path, mmap_mode="r")
        if len(features_memmap) != len(targets_memmap):
            raise ValueError("Analog scorer requires aligned features and targets.")

        sampled_idx = np.arange(self.window_size - 1, len(targets_memmap), max(1, int(sample_stride)), dtype=np.int64)
        if timestamps_path.exists():
            timestamps = np.asarray(np.load(timestamps_path, mmap_mode="r")[sampled_idx], dtype="datetime64[ns]")
            years = timestamps.astype("datetime64[Y]").astype(int) + 1970
            year_mask = np.isin(years, np.asarray([int(year) for year in train_years], dtype=np.int64))
            sampled_idx = sampled_idx[year_mask]
        if sampled_idx.size == 0:
            sampled_idx = np.arange(self.window_size - 1, min(len(targets_memmap), max_samples + self.window_size - 1), dtype=np.int64)
        if sampled_idx.size > max_samples:
            sampled_idx = sampled_idx[:max_samples]

        sampled_rows = np.asarray(features_memmap[sampled_idx][:, self.row_feature_indices], dtype=np.float32)
        sampled_windows = np.asarray(
            [features_memmap[end_idx - self.window_size + 1 : end_idx + 1, self.window_feature_indices] for end_idx in sampled_idx],
            dtype=np.float32,
        )
        sampled_targets = np.asarray(targets_memmap[sampled_idx], dtype=np.float32)
        finite_mask = np.isfinite(sampled_rows).all(axis=1) & np.isfinite(sampled_windows).all(axis=(1, 2)) & np.isfinite(sampled_targets)
        sampled_rows = sampled_rows[finite_mask]
        sampled_windows = sampled_windows[finite_mask]
        sampled_targets = sampled_targets[finite_mask]
        if sampled_rows.size == 0 or sampled_windows.size == 0:
            raise ValueError("Analog scorer could not build a valid historical regime sample.")

        self.row_mean = sampled_rows.mean(axis=0).astype(np.float32)
        self.row_std = sampled_rows.std(axis=0).astype(np.float32)
        self.row_std[self.row_std < 1e-6] = 1.0
        self.rows = ((sampled_rows - self.row_mean) / self.row_std).astype(np.float32)

        self.window_mean = sampled_windows.mean(axis=(0, 1)).astype(np.float32)
        self.window_std = sampled_windows.std(axis=(0, 1)).astype(np.float32)
        self.window_std[self.window_std < 1e-6] = 1.0
        self.windows = ((sampled_windows - self.window_mean) / self.window_std).astype(np.float32)

        self.targets = sampled_targets.astype(np.float32)
        session_slice = self.row_feature_names.index("session_asian")
        self.session_vectors = sampled_rows[:, session_slice : session_slice + 4].astype(np.float32)
        self.sample_size = int(len(self.targets))

    def _row_vector(self, current_row: Mapping[str, Any]) -> np.ndarray:
        values = np.asarray([float(current_row.get(name, 0.0) or 0.0) for name in self.row_feature_names], dtype=np.float32)
        return (values - self.row_mean) / self.row_std

    def _window_matrix(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        row_list = list(rows)
        if not row_list:
            row_list = [{name: 0.0 for name in self.window_feature_names}]
        if len(row_list) >= self.window_size:
            row_list = row_list[-self.window_size :]
        else:
            row_list = ([row_list[0]] * (self.window_size - len(row_list))) + row_list
        matrix = np.asarray(
            [[float(row.get(name, 0.0) or 0.0) for name in self.window_feature_names] for row in row_list],
            dtype=np.float32,
        )
        return (matrix - self.window_mean) / self.window_std

    def _candidate_mask(self, row_vector: np.ndarray) -> np.ndarray:
        session_start = self.row_feature_names.index("session_asian")
        session_vector = row_vector[session_start : session_start + 4] * self.row_std[session_start : session_start + 4] + self.row_mean[session_start : session_start + 4]
        if float(np.max(session_vector)) < 0.5:
            return np.ones(self.sample_size, dtype=bool)
        dominant = int(np.argmax(session_vector))
        sample_dominant = np.argmax(self.session_vectors, axis=1)
        mask = sample_dominant == dominant
        if int(mask.sum()) < self.top_k:
            return np.ones(self.sample_size, dtype=bool)
        return mask

    def score_history(self, rows: Sequence[Mapping[str, Any]]) -> AnalogScore:
        row_list = list(rows)
        if not row_list:
            return AnalogScore(0.5, 0.0, 0.0, 1.0, 0, "unknown", self.sample_size, "regime", self.window_size)
        latest_row = row_list[-1]
        row_vector = self._row_vector(latest_row)
        window_matrix = self._window_matrix(row_list)
        candidate_mask = self._candidate_mask(row_vector)
        rows_bank = self.rows[candidate_mask]
        windows_bank = self.windows[candidate_mask]
        targets = self.targets[candidate_mask]
        candidate_count = int(len(targets))
        if candidate_count == 0:
            return AnalogScore(0.5, 0.0, 0.0, 1.0, 0, "unknown", self.sample_size, "regime", self.window_size)

        row_distances = np.sqrt(np.mean((rows_bank - row_vector) ** 2, axis=1))
        window_distances = np.sqrt(np.mean((windows_bank - window_matrix) ** 2, axis=(1, 2)))
        distances = (0.72 * window_distances) + (0.28 * row_distances)

        top_k = min(self.top_k, candidate_count)
        nearest_idx = np.argpartition(distances, top_k - 1)[:top_k]
        nearest_distances = distances[nearest_idx]
        nearest_targets = targets[nearest_idx]
        weights = np.exp(-nearest_distances / max(0.30, float(np.median(nearest_distances) + 1e-6)))
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-9:
            weights = np.full_like(nearest_distances, 1.0 / max(1, len(nearest_distances)))
        else:
            weights = weights / weight_sum

        bullish_probability = float(np.sum(weights * nearest_targets))
        directional_bias = float((bullish_probability - 0.5) * 2.0)
        mean_distance = float(np.mean(nearest_distances))
        similarity = 1.0 / (1.0 + mean_distance)
        conviction = abs(directional_bias)
        confidence = _clamp(similarity * (0.30 + 0.70 * conviction), 0.0, 1.0)
        session_start = self.row_feature_names.index("session_asian")
        session = _session_name(
            row_vector[session_start : session_start + 4] * self.row_std[session_start : session_start + 4]
            + self.row_mean[session_start : session_start + 4]
        )
        return AnalogScore(
            bullish_probability=round(bullish_probability, 6),
            directional_bias=round(directional_bias, 6),
            confidence=round(confidence, 6),
            mean_distance=round(mean_distance, 6),
            support=int(top_k),
            session=session,
            sample_size=self.sample_size,
            mode="regime",
            window_size=self.window_size,
        )

    def score_row(self, current_row: Mapping[str, Any]) -> AnalogScore:
        return self.score_history([current_row])

    def score_window(self, rows: Sequence[Mapping[str, Any]]) -> AnalogScore:
        return self.score_history(rows)

    def snapshot(self) -> dict[str, Any]:
        return {
            "row_feature_names": list(self.row_feature_names),
            "window_feature_names": list(self.window_feature_names),
            "sample_size": self.sample_size,
            "top_k": self.top_k,
            "window_size": self.window_size,
            "mode": "regime",
        }


@lru_cache(maxsize=1)
def get_historical_analog_scorer() -> HistoricalAnalogScorer:
    scorer = HistoricalAnalogScorer()
    ANALOG_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALOG_REPORT_PATH.write_text(json.dumps({"status": "ready", **scorer.snapshot()}, indent=2), encoding="utf-8")
    return scorer

