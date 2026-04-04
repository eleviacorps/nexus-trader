from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore
except ImportError:  # pragma: no cover
    HistGradientBoostingRegressor = None


SETL_FEATURES: tuple[str, ...] = (
    "setl_stage_id",
    "selector_score",
    "branch_confidence",
    "generator_probability",
    "predicted_edge",
    "remaining_move_abs",
    "consensus_strength",
    "branch_disagreement",
    "cone_width_15m",
    "minority_share",
    "volatility_scale",
    "analog_similarity",
    "analog_disagreement_v9",
    "cesm_state_id",
    "cesm_confidence",
    "cesm_transition_score",
    "pmwm_institutional_positioning",
    "pmwm_retail_sentiment_momentum",
    "pmwm_structural_memory_strength",
    "pmwm_regime_persistence",
    "pmwm_smart_money_fingerprint",
    "v10_diversity_score",
    "pcop_survival_score",
)


@dataclass(frozen=True)
class SetlThreshold:
    threshold: float
    participation_rate: float
    avg_unit_pnl: float


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V11 SETL.")


def build_setl_features(frame, *, stage_bars: int):
    _require_pandas()
    output = frame.copy()
    stage_bars = int(stage_bars)
    if stage_bars == 0:
        current_price = output["entry_open_price"].to_numpy(dtype=np.float32)
    elif stage_bars == 5:
        current_price = output["actual_price_5m"].to_numpy(dtype=np.float32)
    elif stage_bars == 10:
        current_price = output["actual_price_10m"].to_numpy(dtype=np.float32)
    else:
        raise ValueError("stage_bars must be one of 0, 5, 10.")
    predicted_final = output["predicted_price_15m"].to_numpy(dtype=np.float32)
    direction = np.where(predicted_final >= current_price, 1.0, -1.0).astype(np.float32)
    remaining_move = (predicted_final - current_price) / np.maximum(current_price, 1e-6)
    output["setl_stage_id"] = np.float32(stage_bars / 5.0)
    output["setl_trade_direction"] = direction
    output["predicted_edge"] = (direction * remaining_move).astype(np.float32)
    output["remaining_move_abs"] = np.abs(remaining_move).astype(np.float32)
    output["pcop_survival_score"] = output.get("pcop_survival_score", 0.5)
    return output


def train_setl_model(frame, *, feature_names: Sequence[str] = SETL_FEATURES, target_col: str = "setl_target_net_unit_pnl") -> tuple[Any, tuple[str, ...]]:
    _require_pandas()
    usable = [name for name in feature_names if name in frame.columns]
    if HistGradientBoostingRegressor is None or not usable:
        return None, tuple(usable)
    targets = frame[target_col].to_numpy(dtype=np.float32)
    model = HistGradientBoostingRegressor(max_depth=6, max_iter=260, learning_rate=0.045, random_state=42)
    model.fit(frame[usable].fillna(0.0).to_numpy(dtype=np.float32), targets)
    return model, tuple(usable)


def score_setl_model(model, frame, *, feature_names: Sequence[str]) -> np.ndarray:
    if model is None or not feature_names:
        return np.zeros(len(frame), dtype=np.float32)
    values = frame[list(feature_names)].fillna(0.0).to_numpy(dtype=np.float32)
    return np.asarray(model.predict(values), dtype=np.float32)


def optimize_setl_threshold(predicted_pnl: np.ndarray, actual_pnl: np.ndarray) -> SetlThreshold:
    predicted = np.asarray(predicted_pnl, dtype=np.float32)
    actual = np.asarray(actual_pnl, dtype=np.float32)
    best = SetlThreshold(threshold=0.0, participation_rate=0.0, avg_unit_pnl=0.0)
    if predicted.size == 0:
        return best
    candidates = sorted(set(np.quantile(predicted, [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]).astype(np.float32).tolist() + [0.0]))
    for threshold in candidates:
        active = predicted >= float(threshold)
        participation = float(np.mean(active)) if len(active) else 0.0
        if participation <= 0.02:
            continue
        pnl = float(np.mean(actual[active])) if np.any(active) else 0.0
        score = pnl * np.sqrt(max(participation, 1e-6))
        incumbent = best.avg_unit_pnl * np.sqrt(max(best.participation_rate, 1e-6))
        if score > incumbent:
            best = SetlThreshold(threshold=float(threshold), participation_rate=participation, avg_unit_pnl=pnl)
    return best
