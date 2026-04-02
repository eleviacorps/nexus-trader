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


GATE_CONTEXT_COLUMNS = [
    "gate_ctx_atr",
    "gate_ctx_bb_width",
    "gate_ctx_volume",
    "gate_ctx_session_overlap",
    "gate_ctx_regime_strength",
    "gate_ctx_regime_persistence",
    "gate_ctx_transition_risk",
    "gate_ctx_state_entropy",
    "gate_ctx_tail_risk",
    "gate_ctx_vol_realism",
    "gate_ctx_fair_value_abs",
    "gate_ctx_kalman_dislocation_abs",
    "gate_ctx_trend_score",
    "gate_ctx_route_confidence",
    "gate_ctx_route_bias",
    "gate_ctx_state_imbalance",
    "gate_ctx_chop_risk",
    "gate_ctx_dynamics_confidence",
    "gate_ctx_dynamics_trend",
    "gate_ctx_dynamics_breakout",
    "gate_ctx_dynamics_range",
    "gate_ctx_dynamics_panic",
]


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


def merge_market_dynamics_features(price_frame, dynamics_frame):
    pandas = _require_pandas()
    if dynamics_frame is None or len(dynamics_frame) == 0:
        return price_frame
    merged = price_frame.copy()
    aligned = dynamics_frame.copy()
    aligned.index = pandas.to_datetime(aligned.index, errors="coerce")
    merged.index = pandas.to_datetime(merged.index, errors="coerce")
    for column in aligned.columns:
        if str(column).startswith("market_dynamics_"):
            merged[column] = aligned[column].reindex(merged.index).ffill().bfill()
    return merged


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

    quant_regime_strength = frame["quant_regime_strength"].to_numpy(dtype=np.float32, copy=True) if "quant_regime_strength" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_regime_persistence = frame["quant_regime_persistence"].to_numpy(dtype=np.float32, copy=True) if "quant_regime_persistence" in frame.columns else np.full(len(frame), 0.5, dtype=np.float32)
    quant_transition_risk = frame["quant_transition_risk"].to_numpy(dtype=np.float32, copy=True) if "quant_transition_risk" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_state_entropy = frame["quant_state_entropy"].to_numpy(dtype=np.float32, copy=True) if "quant_state_entropy" in frame.columns else np.full(len(frame), 0.5, dtype=np.float32)
    quant_tail_risk = frame["quant_tail_risk"].to_numpy(dtype=np.float32, copy=True) if "quant_tail_risk" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_vol_realism = frame["quant_vol_realism"].to_numpy(dtype=np.float32, copy=True) if "quant_vol_realism" in frame.columns else np.full(len(frame), 0.5, dtype=np.float32)
    quant_fair_value_z = frame["quant_fair_value_z"].to_numpy(dtype=np.float32, copy=True) if "quant_fair_value_z" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_kalman_dislocation = frame["quant_kalman_dislocation"].to_numpy(dtype=np.float32, copy=True) if "quant_kalman_dislocation" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_trend_score = frame["quant_trend_score"].to_numpy(dtype=np.float32, copy=True) if "quant_trend_score" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_route_prob_up = frame["quant_route_prob_up"].to_numpy(dtype=np.float32, copy=True) if "quant_route_prob_up" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_route_prob_down = frame["quant_route_prob_down"].to_numpy(dtype=np.float32, copy=True) if "quant_route_prob_down" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_route_confidence = frame["quant_route_confidence"].to_numpy(dtype=np.float32, copy=True) if "quant_route_confidence" in frame.columns else quant_regime_strength.copy()
    quant_route_bias = np.clip(quant_route_prob_up - quant_route_prob_down, -1.0, 1.0).astype(np.float32)
    dynamics_confidence = frame["market_dynamics_confidence"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_confidence" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_trend_up = frame["market_dynamics_prob_trend_up"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_trend_up" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_trend_down = frame["market_dynamics_prob_trend_down"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_trend_down" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_breakout = frame["market_dynamics_prob_breakout"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_breakout" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_range = frame["market_dynamics_prob_range"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_range" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_mean_reversion = frame["market_dynamics_prob_mean_reversion"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_mean_reversion" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_false_breakout = frame["market_dynamics_prob_false_breakout"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_false_breakout" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_panic = frame["market_dynamics_prob_panic_news_shock"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_panic_news_shock" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_high_vol = frame["market_dynamics_prob_high_volatility"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_high_volatility" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_low_vol = frame["market_dynamics_prob_low_volatility"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_low_volatility" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_quantized_hold = (
        (quant_transition_risk >= 0.68)
        | (quant_tail_risk >= 0.72)
        | (quant_state_entropy >= 0.78)
        | (quant_vol_realism <= 0.18)
        | (np.abs(quant_fair_value_z) >= 2.75)
        | (np.abs(quant_kalman_dislocation) >= 0.010)
    ).astype(np.float32)
    primary_hold_mask = np.maximum(primary_hold_mask, quant_quantized_hold).astype(np.float32)

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
    directional_trend_support = np.where(primary_returns >= 0.0, dynamics_trend_up, dynamics_trend_down).astype(np.float32)
    range_risk = np.clip(dynamics_range + dynamics_mean_reversion + 0.6 * dynamics_false_breakout, 0.0, 1.0).astype(np.float32)
    panic_risk = np.clip(dynamics_panic + 0.5 * dynamics_high_vol, 0.0, 1.0).astype(np.float32)
    dynamics_hold = (
        (dynamics_confidence >= 0.55)
        & (range_risk >= 0.55)
        & (move_strength <= 1.10)
    ) | (
        (dynamics_confidence >= 0.60)
        & (panic_risk >= 0.65)
        & (agreement_ratio <= 0.45)
    ) | (
        (dynamics_low_vol >= 0.65)
        & (move_strength <= 0.75)
    )
    primary_hold_mask = np.maximum(primary_hold_mask, dynamics_hold.astype(np.float32)).astype(np.float32)
    weights = 0.65 + 0.55 * np.clip(move_strength, 0.0, 3.0) + 0.90 * agreement_ratio - 0.45 * disagreement_ratio
    weights = np.where(primary_hold_mask > 0.5, hold_weight + 0.15 * agreement_ratio, weights)
    trend_alignment = ((np.sign(primary_returns) == np.sign(quant_trend_score)) & (np.abs(primary_returns) > primary_threshold)).astype(np.float32)
    dynamics_alignment = (
        0.32 * directional_trend_support
        + 0.22 * dynamics_breakout
        + 0.08 * dynamics_confidence
        - 0.24 * range_risk
        - 0.12 * panic_risk * (move_strength < 1.0).astype(np.float32)
    )
    quant_bonus = (
        0.30 * quant_regime_strength
        + 0.12 * quant_regime_persistence
        + 0.22 * quant_vol_realism
        + 0.10 * quant_route_confidence
        + 0.18 * trend_alignment
        - 0.35 * quant_transition_risk
        - 0.12 * quant_state_entropy
        - 0.18 * quant_tail_risk
        - 0.10 * np.clip(np.abs(quant_fair_value_z) - 1.5, 0.0, 2.0)
        - 0.12 * np.clip(np.abs(quant_kalman_dislocation) * 80.0, 0.0, 1.0)
    )
    weights = weights * np.clip(0.85 + quant_bonus + dynamics_alignment, 0.30, 1.75)
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
        "avg_quant_regime_strength": float(np.mean(quant_regime_strength)) if len(quant_regime_strength) else 0.0,
        "avg_quant_regime_persistence": float(np.mean(quant_regime_persistence)) if len(quant_regime_persistence) else 0.0,
        "avg_quant_transition_risk": float(np.mean(quant_transition_risk)) if len(quant_transition_risk) else 0.0,
        "avg_quant_state_entropy": float(np.mean(quant_state_entropy)) if len(quant_state_entropy) else 0.0,
        "avg_quant_tail_risk": float(np.mean(quant_tail_risk)) if len(quant_tail_risk) else 0.0,
        "avg_quant_vol_realism": float(np.mean(quant_vol_realism)) if len(quant_vol_realism) else 0.0,
        "avg_quant_fair_value_z": float(np.mean(np.abs(quant_fair_value_z))) if len(quant_fair_value_z) else 0.0,
        "avg_quant_route_confidence": float(np.mean(quant_route_confidence)) if len(quant_route_confidence) else 0.0,
        "avg_quant_kalman_dislocation": float(np.mean(np.abs(quant_kalman_dislocation))) if len(quant_kalman_dislocation) else 0.0,
        "avg_dynamics_confidence": float(np.mean(dynamics_confidence)) if len(dynamics_confidence) else 0.0,
        "avg_dynamics_trend_support": float(np.mean(directional_trend_support)) if len(directional_trend_support) else 0.0,
        "avg_dynamics_breakout": float(np.mean(dynamics_breakout)) if len(dynamics_breakout) else 0.0,
        "avg_dynamics_range_risk": float(np.mean(range_risk)) if len(range_risk) else 0.0,
        "avg_dynamics_panic_risk": float(np.mean(panic_risk)) if len(panic_risk) else 0.0,
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


def build_gate_context_matrix(frame) -> np.ndarray:
    atr_pct = np.abs(frame["atr_pct"].to_numpy(dtype=np.float32, copy=True)) if "atr_pct" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    bb_width = np.abs(frame["bb_width"].to_numpy(dtype=np.float32, copy=True)) if "bb_width" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    volume_ratio = frame["volume_ratio"].to_numpy(dtype=np.float32, copy=True) if "volume_ratio" in frame.columns else np.ones(len(frame), dtype=np.float32)
    session_overlap = frame["session_overlap"].to_numpy(dtype=np.float32, copy=True) if "session_overlap" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_regime_strength = frame["quant_regime_strength"].to_numpy(dtype=np.float32, copy=True) if "quant_regime_strength" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_regime_persistence = frame["quant_regime_persistence"].to_numpy(dtype=np.float32, copy=True) if "quant_regime_persistence" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_transition_risk = frame["quant_transition_risk"].to_numpy(dtype=np.float32, copy=True) if "quant_transition_risk" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_state_entropy = frame["quant_state_entropy"].to_numpy(dtype=np.float32, copy=True) if "quant_state_entropy" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_tail_risk = frame["quant_tail_risk"].to_numpy(dtype=np.float32, copy=True) if "quant_tail_risk" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_vol_realism = frame["quant_vol_realism"].to_numpy(dtype=np.float32, copy=True) if "quant_vol_realism" in frame.columns else np.ones(len(frame), dtype=np.float32)
    quant_fair_value_z = frame["quant_fair_value_z"].to_numpy(dtype=np.float32, copy=True) if "quant_fair_value_z" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_kalman_dislocation = frame["quant_kalman_dislocation"].to_numpy(dtype=np.float32, copy=True) if "quant_kalman_dislocation" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    quant_trend_score = frame["quant_trend_score"].to_numpy(dtype=np.float32, copy=True) if "quant_trend_score" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    state_up = frame["quant_state_prob_up"].to_numpy(dtype=np.float32, copy=True) if "quant_state_prob_up" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    state_down = frame["quant_state_prob_down"].to_numpy(dtype=np.float32, copy=True) if "quant_state_prob_down" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    state_chop = frame["quant_state_prob_chop"].to_numpy(dtype=np.float32, copy=True) if "quant_state_prob_chop" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    state_range = frame["quant_state_prob_range"].to_numpy(dtype=np.float32, copy=True) if "quant_state_prob_range" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    route_up = frame["quant_route_prob_up"].to_numpy(dtype=np.float32, copy=True) if "quant_route_prob_up" in frame.columns else state_up.copy()
    route_down = frame["quant_route_prob_down"].to_numpy(dtype=np.float32, copy=True) if "quant_route_prob_down" in frame.columns else state_down.copy()
    route_confidence = frame["quant_route_confidence"].to_numpy(dtype=np.float32, copy=True) if "quant_route_confidence" in frame.columns else quant_regime_strength.copy()

    atr_scale = _normalize_atr_pct(atr_pct, fallback=1e-4)
    atr_feature = np.tanh(atr_scale / max(1e-6, _median_or_default(atr_scale, 1e-4)))
    bb_scale = np.nan_to_num(bb_width, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    bb_feature = np.tanh(bb_scale / max(1e-6, _median_or_default(bb_scale, 1e-3)))
    volume_feature = np.tanh(np.nan_to_num(volume_ratio - 1.0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    fair_value_abs = np.clip(np.abs(quant_fair_value_z) / 3.0, 0.0, 1.0).astype(np.float32)
    kalman_dislocation_abs = np.clip(np.abs(quant_kalman_dislocation) * 80.0, 0.0, 1.0).astype(np.float32)
    state_imbalance = np.clip(state_up - state_down, -1.0, 1.0).astype(np.float32)
    route_bias = np.clip(route_up - route_down, -1.0, 1.0).astype(np.float32)
    chop_risk = np.clip(state_chop + state_range, 0.0, 1.0).astype(np.float32)
    dynamics_confidence = frame["market_dynamics_confidence"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_confidence" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_trend = np.clip(
        (frame["market_dynamics_prob_trend_up"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_trend_up" in frame.columns else np.zeros(len(frame), dtype=np.float32))
        + (frame["market_dynamics_prob_trend_down"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_trend_down" in frame.columns else np.zeros(len(frame), dtype=np.float32)),
        0.0,
        1.0,
    ).astype(np.float32)
    dynamics_breakout = np.clip(frame["market_dynamics_prob_breakout"].to_numpy(dtype=np.float32, copy=True), 0.0, 1.0) if "market_dynamics_prob_breakout" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    dynamics_range = np.clip(
        (frame["market_dynamics_prob_range"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_range" in frame.columns else np.zeros(len(frame), dtype=np.float32))
        + (frame["market_dynamics_prob_mean_reversion"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_mean_reversion" in frame.columns else np.zeros(len(frame), dtype=np.float32)),
        0.0,
        1.0,
    ).astype(np.float32)
    dynamics_panic = np.clip(
        (frame["market_dynamics_prob_panic_news_shock"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_panic_news_shock" in frame.columns else np.zeros(len(frame), dtype=np.float32))
        + 0.5 * (frame["market_dynamics_prob_high_volatility"].to_numpy(dtype=np.float32, copy=True) if "market_dynamics_prob_high_volatility" in frame.columns else np.zeros(len(frame), dtype=np.float32)),
        0.0,
        1.0,
    ).astype(np.float32)

    context = np.column_stack(
        [
            atr_feature,
            bb_feature,
            volume_feature,
            np.clip(session_overlap, 0.0, 1.0).astype(np.float32),
            np.clip(quant_regime_strength, 0.0, 1.0).astype(np.float32),
            np.clip(quant_regime_persistence, 0.0, 1.0).astype(np.float32),
            np.clip(quant_transition_risk, 0.0, 1.0).astype(np.float32),
            np.clip(quant_state_entropy, 0.0, 1.0).astype(np.float32),
            np.clip(quant_tail_risk, 0.0, 1.0).astype(np.float32),
            np.clip(quant_vol_realism, 0.0, 1.0).astype(np.float32),
            fair_value_abs,
            kalman_dislocation_abs,
            np.clip(quant_trend_score, -1.0, 1.0).astype(np.float32),
            np.clip(route_confidence, 0.0, 1.0).astype(np.float32),
            route_bias,
            state_imbalance,
            chop_risk,
            np.clip(dynamics_confidence, 0.0, 1.0).astype(np.float32),
            dynamics_trend,
            dynamics_breakout,
            dynamics_range,
            dynamics_panic,
        ]
    ).astype(np.float32, copy=False)
    return context


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
