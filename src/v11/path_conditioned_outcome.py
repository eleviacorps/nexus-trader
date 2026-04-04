from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
except ImportError:  # pragma: no cover
    HistGradientBoostingClassifier = None


PCOP_STAGE5_FEATURES: tuple[str, ...] = (
    "pcop_stage_id",
    "pcop_error_now",
    "pcop_direction_match_now",
    "pcop_remaining_move",
    "pcop_projected_followthrough",
    "selector_score",
    "branch_confidence",
    "generator_probability",
    "branch_disagreement",
    "consensus_strength",
    "cesm_state_id",
    "cesm_confidence",
    "pmwm_institutional_positioning",
    "pmwm_structural_memory_strength",
    "pmwm_regime_persistence",
    "v10_diversity_score",
)

PCOP_STAGE10_FEATURES: tuple[str, ...] = PCOP_STAGE5_FEATURES + (
    "pcop_error_prev",
    "pcop_error_mean",
)


@dataclass(frozen=True)
class PathConditionedResult:
    sample_count: int
    mean_survival_score: float
    mean_selected_remaining_move: float


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V11 path-conditioned outcome modeling.")


def _safe_sign(values: np.ndarray) -> np.ndarray:
    output = np.sign(values.astype(np.float32, copy=False))
    output[output == 0.0] = 1.0
    return output


def _scale(frame) -> np.ndarray:
    anchor = np.maximum(frame["anchor_price"].to_numpy(dtype=np.float32), 1e-6)
    vol = np.maximum(frame["volatility_scale"].to_numpy(dtype=np.float32), 0.25)
    return np.maximum(anchor * vol * 0.0005, 0.05)


def build_path_conditioned_features(frame, *, stage_bars: int):
    _require_pandas()
    output = frame.copy()
    stage_bars = int(stage_bars)
    if stage_bars not in {5, 10}:
        raise ValueError("stage_bars must be 5 or 10.")
    scale = _scale(output)
    if stage_bars == 5:
        actual_now = output["actual_price_5m"].to_numpy(dtype=np.float32)
        predicted_now = output["predicted_price_5m"].to_numpy(dtype=np.float32)
        previous_error = np.zeros(len(output), dtype=np.float32)
        current_move = actual_now - output["anchor_price"].to_numpy(dtype=np.float32)
    else:
        actual_now = output["actual_price_10m"].to_numpy(dtype=np.float32)
        predicted_now = output["predicted_price_10m"].to_numpy(dtype=np.float32)
        previous_error = np.abs(output["predicted_price_5m"].to_numpy(dtype=np.float32) - output["actual_price_5m"].to_numpy(dtype=np.float32)) / scale
        current_move = actual_now - output["actual_price_5m"].to_numpy(dtype=np.float32)
    predicted_final = output["predicted_price_15m"].to_numpy(dtype=np.float32)
    current_error = np.abs(predicted_now - actual_now) / scale
    remaining_move = (predicted_final - actual_now) / np.maximum(output["anchor_price"].to_numpy(dtype=np.float32), 1e-6)
    projected_followthrough = (predicted_final - predicted_now) / np.maximum(output["anchor_price"].to_numpy(dtype=np.float32), 1e-6)
    direction_match = (_safe_sign(predicted_now - output["anchor_price"].to_numpy(dtype=np.float32)) == _safe_sign(current_move)).astype(np.float32)
    output["pcop_stage_id"] = np.float32(stage_bars / 5.0)
    output["pcop_error_now"] = current_error.astype(np.float32)
    output["pcop_error_prev"] = previous_error.astype(np.float32)
    output["pcop_error_mean"] = ((current_error + previous_error) / np.where(stage_bars == 10, 2.0, 1.0)).astype(np.float32)
    output["pcop_direction_match_now"] = direction_match.astype(np.float32)
    output["pcop_remaining_move"] = remaining_move.astype(np.float32)
    output["pcop_projected_followthrough"] = projected_followthrough.astype(np.float32)
    return output


def train_pcop_model(frame, *, stage_bars: int, target_col: str = "is_top_3_branch") -> tuple[Any, tuple[str, ...]]:
    _require_pandas()
    usable_frame = build_path_conditioned_features(frame, stage_bars=stage_bars)
    feature_names = PCOP_STAGE5_FEATURES if int(stage_bars) == 5 else PCOP_STAGE10_FEATURES
    usable = [name for name in feature_names if name in usable_frame.columns]
    if HistGradientBoostingClassifier is None or not usable:
        return None, tuple(usable)
    labels = usable_frame[target_col].to_numpy(dtype=np.float32)
    if len(np.unique(labels)) < 2:
        return None, tuple(usable)
    model = HistGradientBoostingClassifier(max_depth=6, max_iter=220, learning_rate=0.05, random_state=42)
    model.fit(usable_frame[usable].fillna(0.0).to_numpy(dtype=np.float32), labels)
    return model, tuple(usable)


def apply_pcop_model(model, frame, *, stage_bars: int, feature_names: Sequence[str]) -> np.ndarray:
    conditioned = build_path_conditioned_features(frame, stage_bars=stage_bars)
    if model is None or not feature_names:
        return np.ones(len(conditioned), dtype=np.float32) * 0.5
    values = conditioned[list(feature_names)].fillna(0.0).to_numpy(dtype=np.float32)
    return np.asarray(model.predict_proba(values)[:, 1], dtype=np.float32)


def reweight_branches(frame, *, survival_scores: np.ndarray, blend_weight: float = 0.45):
    _require_pandas()
    output = frame.copy()
    base = output["selector_score"].to_numpy(dtype=np.float32)
    output["pcop_survival_score"] = np.asarray(survival_scores, dtype=np.float32)
    output["pcop_conditioned_score"] = ((1.0 - blend_weight) * base) + (blend_weight * output["pcop_survival_score"].to_numpy(dtype=np.float32))
    return output
