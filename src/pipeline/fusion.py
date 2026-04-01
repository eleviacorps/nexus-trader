from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from config.project_config import FEATURE_DIM_TOTAL, PRICE_FEATURE_COLUMNS

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


@dataclass(frozen=True)
class FusionReport:
    rows: int
    feature_dim: int
    target_positive_rate: float
    source_price_path: str
    source_news_path: str
    source_crowd_path: str
    target_hold_rate: float = 0.0
    target_horizon: int = 0
    sequence_rows: int = 0
    sequence_len: int = 0
    source_persona_path: str = ""
    target_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class TargetArtifacts:
    primary_targets: np.ndarray
    primary_hold_mask: np.ndarray
    sample_weights: np.ndarray
    horizon_returns: dict[int, np.ndarray]
    horizon_hold_targets: dict[int, np.ndarray]
    horizon_confidence_targets: dict[int, np.ndarray]
    summary: dict[str, Any]


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("pandas is required for fusion operations.")
    return pd


def load_price_frame(price_path: Path):
    pandas = _require_pandas()
    if price_path.suffix.lower() == ".parquet":
        frame = pandas.read_parquet(price_path)
    else:
        frame = pandas.read_csv(price_path, index_col=0, parse_dates=True)
    return frame


def extract_price_block(frame) -> np.ndarray:
    missing = [column for column in PRICE_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {', '.join(missing)}")
    return frame[PRICE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)


def extract_target_vector(frame, target_column: str = "target_direction") -> np.ndarray:
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")
    return frame[target_column].to_numpy(dtype=np.float32, copy=True)


def normalize_binary_targets(values: np.ndarray) -> np.ndarray:
    return (values > 0).astype(np.float32)


def _require_price_column(frame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Missing required price column: {column}")
    return frame[column].to_numpy(dtype=np.float32, copy=True)


def _forward_return(close: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    future = np.roll(close, -horizon)
    output = np.zeros(len(close), dtype=np.float32)
    valid = np.arange(len(close)) < max(0, len(close) - horizon)
    output[valid] = (future[valid] / np.maximum(close[valid], 1e-6)) - 1.0
    return output


def _median_or_default(values: np.ndarray, default: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(default)
    return float(np.median(finite))


def _normalize_atr_pct(atr_pct: np.ndarray, fallback: float) -> np.ndarray:
    values = np.nan_to_num(atr_pct, nan=fallback, posinf=fallback, neginf=fallback).astype(np.float32, copy=False)
    median_value = _median_or_default(np.abs(values), fallback)
    if median_value > 0.02:
        values = values / 100.0
    return values.astype(np.float32, copy=False)


def build_trade_target_artifacts(
    frame,
    *,
    horizons: tuple[int, ...] = (5, 10, 15, 30),
    primary_horizon: int = 5,
    atr_multiplier: float = 0.35,
    min_abs_return: float = 4e-4,
    hold_weight: float = 0.35,
) -> TargetArtifacts:
    if primary_horizon not in horizons:
        raise ValueError("primary_horizon must be included in horizons")

    close = _require_price_column(frame, "close")
    atr_pct = np.abs(frame["atr_pct"].to_numpy(dtype=np.float32, copy=True)) if "atr_pct" in frame.columns else np.full(len(close), np.nan, dtype=np.float32)
    fallback_atr = _median_or_default(np.abs(_forward_return(close, primary_horizon)), max(min_abs_return, 1e-4))
    atr_pct = _normalize_atr_pct(atr_pct, fallback=fallback_atr)
    base_threshold = np.maximum(atr_pct * float(atr_multiplier), float(min_abs_return)).astype(np.float32)

    horizon_returns: dict[int, np.ndarray] = {int(h): _forward_return(close, int(h)) for h in horizons}
    threshold_by_horizon = {
        int(h): np.asarray(base_threshold * np.sqrt(max(1.0, float(h) / float(primary_horizon))), dtype=np.float32)
        for h in horizons
    }

    primary_returns = horizon_returns[int(primary_horizon)]
    primary_threshold = threshold_by_horizon[int(primary_horizon)]
    primary_targets = (primary_returns > 0.0).astype(np.float32)
    primary_hold_mask = (np.abs(primary_returns) <= primary_threshold).astype(np.float32)

    significant_signs = []
    horizon_summary: dict[str, Any] = {}
    horizon_hold_targets: dict[int, np.ndarray] = {}
    horizon_confidence_targets: dict[int, np.ndarray] = {}
    for horizon in horizons:
        returns = horizon_returns[int(horizon)]
        threshold = threshold_by_horizon[int(horizon)]
        hold_mask = np.abs(returns) <= threshold
        confidence = np.clip(np.abs(returns) / np.maximum(threshold, 1e-6), 0.0, 3.0) / 3.0
        sign = np.sign(returns)
        sign[hold_mask] = 0.0
        significant_signs.append(sign.astype(np.float32))
        horizon_hold_targets[int(horizon)] = hold_mask.astype(np.float32)
        horizon_confidence_targets[int(horizon)] = confidence.astype(np.float32)
        horizon_summary[f"{horizon}m"] = {
            "positive_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
            "negative_rate": float((returns < 0.0).mean()) if len(returns) else 0.0,
            "hold_rate": float(hold_mask.mean()) if len(returns) else 0.0,
            "avg_confidence_target": float(np.mean(confidence)) if len(confidence) else 0.0,
            "avg_abs_return": float(np.mean(np.abs(returns))) if len(returns) else 0.0,
            "avg_threshold": float(np.mean(threshold)) if len(threshold) else 0.0,
        }

    sign_matrix = np.stack(significant_signs, axis=1).astype(np.float32)
    primary_sign = np.sign(primary_returns).astype(np.float32)
    direction_match = (sign_matrix == primary_sign[:, None]) & (sign_matrix != 0.0) & (primary_sign[:, None] != 0.0)
    disagreement = (sign_matrix != primary_sign[:, None]) & (sign_matrix != 0.0) & (primary_sign[:, None] != 0.0)
    non_hold = sign_matrix != 0.0
    agreement_ratio = direction_match.sum(axis=1) / np.maximum(1.0, non_hold.sum(axis=1))
    disagreement_ratio = disagreement.sum(axis=1) / np.maximum(1.0, non_hold.sum(axis=1))

    move_strength = np.abs(primary_returns) / np.maximum(primary_threshold, 1e-6)
    weights = 0.65 + 0.55 * np.clip(move_strength, 0.0, 3.0) + 0.90 * agreement_ratio - 0.45 * disagreement_ratio
    weights = np.where(primary_hold_mask > 0.5, hold_weight + 0.15 * agreement_ratio, weights)
    weights = np.clip(weights, max(0.05, hold_weight), 4.0).astype(np.float32)

    summary = {
        "primary_horizon": int(primary_horizon),
        "horizons": [int(h) for h in horizons],
        "target_positive_rate": float(primary_targets.mean()) if len(primary_targets) else 0.0,
        "target_hold_rate": float(primary_hold_mask.mean()) if len(primary_hold_mask) else 0.0,
        "avg_primary_threshold": float(np.mean(primary_threshold)) if len(primary_threshold) else 0.0,
        "avg_primary_abs_return": float(np.mean(np.abs(primary_returns))) if len(primary_returns) else 0.0,
        "avg_primary_move_strength": float(np.mean(move_strength)) if len(move_strength) else 0.0,
        "avg_horizon_agreement": float(np.mean(agreement_ratio)) if len(agreement_ratio) else 0.0,
        "avg_horizon_disagreement": float(np.mean(disagreement_ratio)) if len(disagreement_ratio) else 0.0,
        "sample_weight_mean": float(weights.mean()) if len(weights) else 0.0,
        "sample_weight_max": float(weights.max()) if len(weights) else 0.0,
        "hold_weight": float(hold_weight),
        "horizon_summary": horizon_summary,
    }
    return TargetArtifacts(
        primary_targets=primary_targets.astype(np.float32, copy=False),
        primary_hold_mask=primary_hold_mask.astype(np.float32, copy=False),
        sample_weights=weights.astype(np.float32, copy=False),
        horizon_returns={int(key): value.astype(np.float32, copy=False) for key, value in horizon_returns.items()},
        horizon_hold_targets={int(key): value.astype(np.float32, copy=False) for key, value in horizon_hold_targets.items()},
        horizon_confidence_targets={int(key): value.astype(np.float32, copy=False) for key, value in horizon_confidence_targets.items()},
        summary=summary,
    )


def align_row_count(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    if not arrays:
        return ()
    row_count = min(len(array) for array in arrays)
    return tuple(np.asarray(array[:row_count], dtype=np.float32) for array in arrays)


def build_fused_feature_matrix(price_block: np.ndarray, news_block: np.ndarray, crowd_block: np.ndarray) -> np.ndarray:
    price_block, news_block, crowd_block = align_row_count(price_block, news_block, crowd_block)
    fused = np.concatenate([price_block, news_block, crowd_block], axis=1)
    if fused.shape[1] != FEATURE_DIM_TOTAL:
        raise ValueError(f"Expected fused width {FEATURE_DIM_TOTAL}, got {fused.shape[1]}")
    return fused.astype(np.float32, copy=False)


def build_sequence_tensor(feature_matrix: np.ndarray, target_vector: np.ndarray, sequence_len: int) -> tuple[np.ndarray, np.ndarray]:
    if sequence_len <= 0:
        raise ValueError("sequence_len must be positive")
    if len(feature_matrix) != len(target_vector):
        raise ValueError("Feature matrix and target vector must have the same row count")
    usable = len(feature_matrix) - sequence_len + 1
    if usable <= 0:
        raise ValueError("Not enough rows to build sequence tensor")

    tensor = np.stack([feature_matrix[index : index + sequence_len] for index in range(usable)], axis=0).astype(np.float32, copy=False)
    seq_targets = np.asarray(target_vector[sequence_len - 1 :], dtype=np.float32)
    return tensor, seq_targets


def save_numpy_artifact(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_fusion_report(path: Path, report: FusionReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")
