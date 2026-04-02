from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


REGIME_LABELS: tuple[str, ...] = (
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
class RegimeDetectionResult:
    probabilities: dict[str, float]
    dominant_regime: str
    dominant_confidence: float
    trend_bias: float
    volatility_bias: float
    shock_bias: float


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _safe(row: Mapping[str, float], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def detect_regime(current_row: Mapping[str, float]) -> RegimeDetectionResult:
    trend_score = np.tanh(
        (0.85 * _safe(current_row, "quant_trend_score"))
        + (0.55 * _safe(current_row, "ema_cross"))
        + (0.45 * _safe(current_row, "macd_hist"))
        + (0.20 * _safe(current_row, "analog_bias"))
    )
    transition_risk = _clamp(_safe(current_row, "quant_transition_risk"))
    regime_strength = _clamp(_safe(current_row, "quant_regime_strength"))
    vol_forecast = max(1e-6, _safe(current_row, "quant_vol_forecast", _safe(current_row, "atr_pct", 0.001)))
    atr_pct = max(1e-6, _safe(current_row, "atr_pct", 0.001))
    vol_ratio = np.tanh(vol_forecast / max(atr_pct, 1e-6))
    tail_risk = _clamp(_safe(current_row, "quant_tail_risk"))
    news_shock = _clamp(abs(_safe(current_row, "news_shock", _safe(current_row, "llm_event_severity", 0.0))))
    crowd_panic = _clamp(abs(_safe(current_row, "crowd_panic_index", abs(_safe(current_row, "crowd_bias") * _safe(current_row, "crowd_extreme")))))
    macro_bias = _safe(current_row, "macro_bias")
    route_up = _clamp(_safe(current_row, "quant_route_prob_up", 0.25))
    route_down = _clamp(_safe(current_row, "quant_route_prob_down", 0.25))
    route_range = _clamp(_safe(current_row, "quant_route_prob_range", 0.25))
    route_chop = _clamp(_safe(current_row, "quant_route_prob_chop", 0.25))

    scores = {
        "trend_up": max(0.0, 0.55 * trend_score + 0.25 * route_up + 0.20 * regime_strength),
        "trend_down": max(0.0, 0.55 * -trend_score + 0.25 * route_down + 0.20 * regime_strength),
        "mean_reversion": max(0.0, 0.40 * transition_risk + 0.30 * route_range + 0.30 * (1.0 - abs(trend_score))),
        "range": max(0.0, 0.45 * route_range + 0.30 * (1.0 - regime_strength) + 0.25 * (1.0 - vol_ratio)),
        "breakout": max(0.0, 0.35 * abs(trend_score) + 0.30 * vol_ratio + 0.20 * regime_strength + 0.15 * max(route_up, route_down)),
        "false_breakout": max(0.0, 0.35 * transition_risk + 0.25 * route_chop + 0.20 * crowd_panic + 0.20 * (1.0 - regime_strength)),
        "panic_news_shock": max(0.0, 0.45 * news_shock + 0.25 * tail_risk + 0.15 * crowd_panic + 0.15 * abs(macro_bias)),
        "high_volatility": max(0.0, 0.55 * vol_ratio + 0.25 * tail_risk + 0.20 * news_shock),
        "low_volatility": max(0.0, 0.50 * (1.0 - vol_ratio) + 0.30 * route_range + 0.20 * (1.0 - tail_risk)),
    }
    total = float(sum(scores.values()))
    if total <= 0.0:
        probabilities = {label: round(1.0 / len(REGIME_LABELS), 6) for label in REGIME_LABELS}
    else:
        probabilities = {label: round(float(value / total), 6) for label, value in scores.items()}
    dominant_regime = max(probabilities, key=probabilities.get)
    return RegimeDetectionResult(
        probabilities=probabilities,
        dominant_regime=dominant_regime,
        dominant_confidence=float(probabilities[dominant_regime]),
        trend_bias=float(trend_score),
        volatility_bias=float(vol_ratio),
        shock_bias=float(_clamp(0.65 * news_shock + 0.35 * crowd_panic)),
    )
