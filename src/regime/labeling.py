from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


DYNAMICS_LABELS = (
    "trend_up",
    "trend_down",
    "mean_reversion",
    "range",
    "breakout",
    "false_breakout",
    "panic_news_shock",
    "high_volatility",
    "low_volatility",
)


@dataclass(frozen=True)
class MarketDynamicsReport:
    rows: int
    dominant_counts: dict[str, int]
    dominant_distribution: dict[str, float]
    avg_scores: dict[str, float]
    avg_primary_move: float
    avg_volatility: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("pandas is required for market-dynamics labeling.")
    return pd


def _safe_series(frame: Any, column: str, default: float = 0.0) -> Any:
    pandas = _require_pandas()
    if column not in frame.columns:
        return pandas.Series(np.full(len(frame), default, dtype=np.float32), index=frame.index)
    return frame[column].astype(float)


def _forward_return(close: Any, horizon: int) -> Any:
    return (close.shift(-horizon) / close.replace(0.0, np.nan) - 1.0).fillna(0.0)


def _rolling_zscore(series: Any, window: int) -> Any:
    rolling_mean = series.rolling(window, min_periods=max(3, window // 4)).mean()
    rolling_std = series.rolling(window, min_periods=max(3, window // 4)).std(ddof=0).replace(0.0, np.nan)
    return ((series - rolling_mean) / rolling_std).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _normalize_atr(atr_pct: Any, fallback: float = 1e-4) -> Any:
    values = atr_pct.abs().fillna(fallback)
    median_value = float(values.median()) if len(values) else fallback
    if median_value > 0.02:
        values = values / 100.0
    return values.fillna(fallback)


def _softmax_scores(score_frame: Any) -> Any:
    pandas = _require_pandas()
    clipped = score_frame.clip(lower=-8.0, upper=8.0)
    shifted = clipped.sub(clipped.max(axis=1), axis=0)
    exp_scores = np.exp(shifted)
    exp_frame = pandas.DataFrame(exp_scores, index=score_frame.index, columns=score_frame.columns)
    return exp_frame.div(exp_frame.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


def build_market_dynamics_labels(
    frame: Any,
    *,
    primary_horizon: int = 15,
    secondary_horizon: int = 30,
) -> Any:
    pandas = _require_pandas()
    working = frame.copy()

    open_ = _safe_series(working, "open")
    high = _safe_series(working, "high")
    low = _safe_series(working, "low")
    close = _safe_series(working, "close")
    atr_pct = _normalize_atr(_safe_series(working, "atr_pct", 0.001))
    bb_width = _safe_series(working, "bb_width", 0.0).abs().fillna(0.0)
    bb_pct = _safe_series(working, "bb_pct", 0.5).clip(0.0, 1.0).fillna(0.5)
    volume_ratio = _safe_series(working, "volume_ratio", 1.0).fillna(1.0)
    displacement = _safe_series(working, "displacement", 0.0).fillna(0.0)
    ema_cross = _safe_series(working, "ema_cross", 0.0).fillna(0.0)
    dist_to_high = _safe_series(working, "dist_to_high", 2.0).abs().fillna(2.0)
    dist_to_low = _safe_series(working, "dist_to_low", 2.0).abs().fillna(2.0)
    quant_transition_risk = _safe_series(working, "quant_transition_risk", 0.0).clip(0.0, 1.0)
    quant_state_entropy = _safe_series(working, "quant_state_entropy", 0.5).clip(0.0, 1.0)
    quant_tail_risk = _safe_series(working, "quant_tail_risk", 0.0).clip(0.0, 1.0)
    quant_regime_strength = _safe_series(working, "quant_regime_strength", 0.5).clip(0.0, 1.0)
    quant_trend_score = _safe_series(working, "quant_trend_score", 0.0).clip(-1.0, 1.0)

    ret_1 = _safe_series(working, "return_1", 0.0).fillna(0.0)
    ret_primary = _forward_return(close, primary_horizon)
    ret_secondary = _forward_return(close, secondary_horizon)

    realized_vol = ret_1.rolling(primary_horizon, min_periods=max(3, primary_horizon // 3)).std(ddof=0).fillna(0.0)
    vol_z = _rolling_zscore(realized_vol + atr_pct, 96)
    volume_z = _rolling_zscore(volume_ratio, 96)
    move_strength = (ret_primary.abs() / np.maximum(atr_pct, 1e-6)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    breakout_trigger = ((dist_to_high < 0.35) | (dist_to_low < 0.35)).astype(float)
    directional_break = np.sign(ret_primary).fillna(0.0)
    secondary_sign = np.sign(ret_secondary).fillna(0.0)
    directional_break = directional_break.where(directional_break != 0.0, secondary_sign).fillna(0.0)
    reversal_pressure = (bb_pct.sub(0.5).abs() * 2.0).clip(0.0, 1.0)

    score_frame = pandas.DataFrame(index=working.index)
    score_frame["trend_up"] = (
        2.1 * ret_primary
        + 1.1 * ret_secondary
        + 0.7 * ema_cross
        + 0.8 * quant_trend_score
        + 0.3 * quant_regime_strength
        - 0.5 * quant_transition_risk
    )
    score_frame["trend_down"] = (
        -2.1 * ret_primary
        - 1.1 * ret_secondary
        - 0.7 * ema_cross
        - 0.8 * quant_trend_score
        + 0.3 * quant_regime_strength
        - 0.5 * quant_transition_risk
    )
    score_frame["mean_reversion"] = (
        1.5 * reversal_pressure
        + 0.8 * (1.0 - bb_width.clip(0.0, 1.5))
        + 0.5 * (1.0 - quant_regime_strength)
        + 0.7 * (np.sign(-displacement) == np.sign(ret_primary)).astype(float)
    )
    score_frame["range"] = (
        1.4 * (1.0 - bb_width.clip(0.0, 1.5))
        + 1.0 * (1.0 - vol_z.clip(-1.0, 1.0).abs())
        + 0.6 * (1.0 - quant_regime_strength)
        + 0.6 * (1.0 - breakout_trigger)
        + 0.4 * (1.0 - move_strength.clip(0.0, 2.0) / 2.0)
    )
    score_frame["breakout"] = (
        1.7 * breakout_trigger
        + 1.2 * move_strength.clip(0.0, 3.0)
        + 0.9 * vol_z.clip(lower=0.0)
        + 0.6 * volume_z.clip(lower=0.0)
        + 0.5 * quant_regime_strength
    )
    score_frame["false_breakout"] = (
        1.4 * breakout_trigger
        + 0.8 * reversal_pressure
        + 0.8 * (np.sign(directional_break) != np.sign(ret_secondary)).astype(float)
        + 0.5 * quant_transition_risk
        + 0.5 * quant_state_entropy
    )
    score_frame["panic_news_shock"] = (
        1.2 * vol_z.clip(lower=0.0)
        + 0.9 * volume_z.clip(lower=0.0)
        + 0.9 * quant_tail_risk
        + 0.7 * quant_transition_risk
        + 0.4 * move_strength.clip(0.0, 3.0)
    )
    score_frame["high_volatility"] = (
        1.5 * vol_z.clip(lower=0.0)
        + 0.8 * bb_width.clip(0.0, 2.0)
        + 0.8 * quant_tail_risk
        + 0.4 * quant_transition_risk
    )
    score_frame["low_volatility"] = (
        1.4 * (-vol_z).clip(lower=0.0)
        + 0.9 * (1.0 - bb_width.clip(0.0, 1.0))
        + 0.5 * (1.0 - quant_tail_risk)
        + 0.4 * (1.0 - quant_transition_risk)
    )

    probabilities = _softmax_scores(score_frame)
    dominant_label = probabilities.idxmax(axis=1)
    dominant_confidence = probabilities.max(axis=1).astype(np.float32)

    output = pandas.DataFrame(index=working.index)
    output["market_dynamics_label"] = dominant_label.astype(str)
    output["market_dynamics_confidence"] = dominant_confidence
    output["market_dynamics_primary_return"] = ret_primary.astype(np.float32)
    output["market_dynamics_secondary_return"] = ret_secondary.astype(np.float32)
    output["market_dynamics_vol_z"] = vol_z.astype(np.float32)
    output["market_dynamics_move_strength"] = move_strength.astype(np.float32)
    for label in DYNAMICS_LABELS:
        output[f"market_dynamics_score_{label}"] = score_frame[label].astype(np.float32)
        output[f"market_dynamics_prob_{label}"] = probabilities[label].astype(np.float32)
    return output


def summarize_market_dynamics(labels_frame: Any) -> MarketDynamicsReport:
    dominant = labels_frame["market_dynamics_label"].astype(str)
    counts = {label: int((dominant == label).sum()) for label in DYNAMICS_LABELS}
    total = max(1, int(len(labels_frame)))
    dominant_distribution = {label: round(count / total, 6) for label, count in counts.items()}
    avg_scores = {
        label: round(float(labels_frame[f"market_dynamics_prob_{label}"].mean()), 6)
        for label in DYNAMICS_LABELS
        if f"market_dynamics_prob_{label}" in labels_frame.columns
    }
    return MarketDynamicsReport(
        rows=int(len(labels_frame)),
        dominant_counts=counts,
        dominant_distribution=dominant_distribution,
        avg_scores=avg_scores,
        avg_primary_move=round(float(labels_frame["market_dynamics_primary_return"].abs().mean()), 6),
        avg_volatility=round(float(labels_frame["market_dynamics_vol_z"].abs().mean()), 6),
    )
