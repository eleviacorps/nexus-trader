from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


BRANCH_FEATURES_V9: tuple[str, ...] = (
    "path_curvature",
    "path_acceleration",
    "path_entropy",
    "path_smoothness",
    "path_convexity",
    "reversal_likelihood",
    "breakout_likelihood",
    "mean_reversion_likelihood",
    "news_consistency_v9",
    "macro_consistency_v9",
    "crowd_consistency_v9",
    "order_flow_plausibility",
    "analog_density",
    "analog_disagreement_v9",
    "analog_weighted_accuracy",
    "hmm_regime_probability",
    "regime_persistence",
    "regime_transition_risk_v9",
    "garch_zscore",
    "fair_value_distance",
    "fair_value_mean_reversion_prob",
    "atr_normalised_move_v9",
    "historical_move_percentile",
    "branch_disagreement",
    "consensus_direction",
    "consensus_strength",
    "regime_match_x_analog",
    "volatility_realism_x_fair_value",
    "news_x_crowd",
    "analog_density_x_regime_persistence",
)


@dataclass(frozen=True)
class BranchFeatureSummary:
    sample_count: int
    branch_rows: int
    mean_branch_disagreement: float
    mean_consensus_strength: float


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V9 branch features.")


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float32, copy=False), 0.0, 1.0)


def _row_entropy(sign_steps: np.ndarray) -> np.ndarray:
    result = np.zeros(sign_steps.shape[0], dtype=np.float32)
    for index, row in enumerate(sign_steps):
        counts = np.bincount((row + 1).astype(np.int64), minlength=3).astype(np.float32)
        probabilities = counts / max(float(counts.sum()), 1.0)
        mask = probabilities > 0.0
        result[index] = float(-(probabilities[mask] * np.log(probabilities[mask])).sum())
    return result


def _historical_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    rank = np.empty_like(order, dtype=np.float32)
    rank[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return rank


def build_branch_features(frame) -> Any:
    _require_pandas()
    features = frame.copy()
    timestamp = pd.to_datetime(features["timestamp"], utc=True, errors="coerce")
    features["timestamp"] = timestamp.astype(str)

    anchor = features["anchor_price"].to_numpy(dtype=np.float32)
    price_path = features[["predicted_price_5m", "predicted_price_10m", "predicted_price_15m"]].to_numpy(dtype=np.float32)
    full_path = np.concatenate([anchor.reshape(-1, 1), price_path], axis=1)
    step_returns = np.diff(full_path, axis=1) / np.maximum(anchor.reshape(-1, 1), 1e-4)
    velocity_delta = np.diff(step_returns, axis=1)
    accel_delta = np.diff(velocity_delta, axis=1) if velocity_delta.shape[1] > 1 else np.zeros((len(features), 1), dtype=np.float32)
    sign_steps = np.sign(step_returns).astype(np.int8)

    atr_proxy = np.maximum(anchor * 0.0005 * np.maximum(features["volatility_scale"].to_numpy(dtype=np.float32), 0.25), 0.05)
    distance_from_open = np.abs(price_path - anchor.reshape(-1, 1))
    distance_progress = np.abs(full_path[:, 1:] - anchor.reshape(-1, 1))

    features["path_curvature"] = np.mean(np.abs(velocity_delta), axis=1).astype(np.float32)
    features["path_acceleration"] = np.mean(np.abs(accel_delta), axis=1).astype(np.float32)
    features["path_entropy"] = _row_entropy(sign_steps)
    features["path_smoothness"] = (1.0 / (1.0 + np.mean(np.abs(velocity_delta), axis=1))).astype(np.float32)
    features["path_convexity"] = np.tanh(np.mean(velocity_delta, axis=1) * 6.0).astype(np.float32)
    features["reversal_likelihood"] = (
        np.mean(sign_steps[:, 1:] != sign_steps[:, :-1], axis=1) if sign_steps.shape[1] > 1 else np.zeros(len(features), dtype=np.float32)
    ).astype(np.float32)
    features["breakout_likelihood"] = np.mean(distance_from_open > atr_proxy.reshape(-1, 1), axis=1).astype(np.float32)
    features["mean_reversion_likelihood"] = np.mean(
        distance_progress[:, 1:] <= distance_progress[:, :-1],
        axis=1,
    ).astype(np.float32)

    features["news_consistency_v9"] = _clip01(features["news_consistency"].to_numpy(dtype=np.float32))
    features["macro_consistency_v9"] = _clip01(features["macro_alignment"].to_numpy(dtype=np.float32))
    features["crowd_consistency_v9"] = _clip01(features["crowd_consistency"].to_numpy(dtype=np.float32))

    hours = pd.to_datetime(features["timestamp"], utc=True, errors="coerce").dt.hour.fillna(0).astype(int)
    hourly_median = features.groupby(hours)["branch_volatility"].transform("median").to_numpy(dtype=np.float32)
    hourly_std = features.groupby(hours)["branch_volatility"].transform("std").fillna(0.0).to_numpy(dtype=np.float32)
    volatility_gap = np.abs(features["branch_volatility"].to_numpy(dtype=np.float32) - hourly_median)
    features["order_flow_plausibility"] = np.exp(-(volatility_gap / np.maximum(hourly_std, 1e-4))).astype(np.float32)

    analog_similarity = _clip01(features["analog_similarity"].to_numpy(dtype=np.float32))
    analog_disagreement = _clip01(features["analog_disagreement"].to_numpy(dtype=np.float32))
    features["analog_density"] = (24.0 * analog_similarity * (1.0 - 0.5 * analog_disagreement)).astype(np.float32)
    features["analog_disagreement_v9"] = analog_disagreement
    features["analog_weighted_accuracy"] = (analog_similarity * (1.0 - analog_disagreement)).astype(np.float32)

    regime_persistence = _clip01(features["hmm_persistence"].to_numpy(dtype=np.float32))
    transition_risk = _clip01(features["hmm_transition_risk"].to_numpy(dtype=np.float32))
    features["hmm_regime_probability"] = _clip01(0.65 * regime_persistence + 0.35 * (1.0 - transition_risk))
    features["regime_persistence"] = regime_persistence
    features["regime_transition_risk_v9"] = transition_risk

    fair_value_distance = np.abs(features["fair_value_dislocation"].to_numpy(dtype=np.float32))
    mean_reversion_pressure = features["mean_reversion_pressure"].to_numpy(dtype=np.float32)
    features["garch_zscore"] = features["branch_move_zscore"].to_numpy(dtype=np.float32)
    features["fair_value_distance"] = fair_value_distance.astype(np.float32)
    features["fair_value_mean_reversion_prob"] = _clip01(
        1.0 / (1.0 + np.exp(-(1.75 * mean_reversion_pressure - 8.0 * fair_value_distance)))
    )
    features["atr_normalised_move_v9"] = features["atr_normalized_move"].to_numpy(dtype=np.float32)
    features["historical_move_percentile"] = _historical_percentile(
        np.abs(features["branch_move_size"].to_numpy(dtype=np.float32))
    )

    features["branch_disagreement"] = (
        features.groupby("sample_id")["predicted_price_15m"].transform("std").fillna(0.0).to_numpy(dtype=np.float32)
    )

    weighted_direction = features["generator_probability"].to_numpy(dtype=np.float32) * features["branch_direction"].to_numpy(dtype=np.float32)
    features["_weighted_direction"] = weighted_direction
    consensus_raw = features.groupby("sample_id")["_weighted_direction"].sum()
    consensus_direction_map = {int(sample_id): (1.0 if score >= 0.0 else -1.0) for sample_id, score in consensus_raw.items()}
    features["consensus_direction"] = features["sample_id"].map(consensus_direction_map).astype(np.float32)
    matching_weight = np.where(
        features["branch_direction"].to_numpy(dtype=np.float32) == features["consensus_direction"].to_numpy(dtype=np.float32),
        features["generator_probability"].to_numpy(dtype=np.float32),
        0.0,
    )
    features["_matching_weight"] = matching_weight.astype(np.float32)
    matching_sum = features.groupby("sample_id")["_matching_weight"].sum()
    total_sum = features.groupby("sample_id")["generator_probability"].sum().clip(lower=1e-6)
    consensus_strength_map = (matching_sum / total_sum).astype(np.float32).to_dict()
    features["consensus_strength"] = features["sample_id"].map(consensus_strength_map).astype(np.float32)

    regime_consistency = (
        features["regime_consistency"].to_numpy(dtype=np.float32)
        if "regime_consistency" in features.columns
        else _clip01(features["hmm_regime_match"].to_numpy(dtype=np.float32))
    )
    volatility_realism = (
        features["volatility_realism_v9"].to_numpy(dtype=np.float32)
        if "volatility_realism_v9" in features.columns
        else _clip01(features["volatility_realism"].to_numpy(dtype=np.float32))
    )
    features["regime_match_x_analog"] = (regime_consistency * features["analog_density"].to_numpy(dtype=np.float32)).astype(np.float32)
    features["volatility_realism_x_fair_value"] = (
        volatility_realism * (1.0 / (fair_value_distance + 1e-6))
    ).astype(np.float32)
    features["news_x_crowd"] = (
        features["news_consistency_v9"].to_numpy(dtype=np.float32)
        * features["crowd_consistency_v9"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    features["analog_density_x_regime_persistence"] = (
        features["analog_density"].to_numpy(dtype=np.float32) * regime_persistence
    ).astype(np.float32)
    features = features.drop(columns=["_weighted_direction", "_matching_weight"])
    return features


def summarize_branch_features(frame) -> BranchFeatureSummary:
    _require_pandas()
    return BranchFeatureSummary(
        sample_count=int(frame["sample_id"].nunique()) if len(frame) else 0,
        branch_rows=int(len(frame)),
        mean_branch_disagreement=float(frame["branch_disagreement"].mean()) if len(frame) else 0.0,
        mean_consensus_strength=float(frame["consensus_strength"].mean()) if len(frame) else 0.0,
    )
