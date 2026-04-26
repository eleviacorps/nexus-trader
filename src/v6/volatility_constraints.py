from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.v6.regime_detection import RegimeDetectionResult


@dataclass(frozen=True)
class VolatilityEnvelope:
    horizon_minutes: int
    expected_move_abs: float
    expected_move_pct: float
    lower_bound: float
    upper_bound: float
    large_move_probability: float
    realized_volatility: float


def _safe(row: Mapping[str, float], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def build_volatility_envelopes(
    current_price: float,
    current_row: Mapping[str, float],
    regime: RegimeDetectionResult,
    horizons: tuple[int, ...] = (5, 15, 30),
) -> dict[int, VolatilityEnvelope]:
    atr_pct = max(1e-6, _safe(current_row, "atr_pct", 0.001))
    realized_vol = max(1e-6, _safe(current_row, "quant_vol_forecast", atr_pct))
    news_shock = abs(_safe(current_row, "news_shock", _safe(current_row, "llm_event_severity", 0.0)))
    tail_risk = max(0.0, min(1.0, _safe(current_row, "quant_tail_risk", 0.0)))
    high_vol_weight = float(regime.probabilities.get("high_volatility", 0.0))
    shock_weight = float(regime.probabilities.get("panic_news_shock", 0.0))
    breakout_weight = float(regime.probabilities.get("breakout", 0.0))
    multiplier = 1.0 + (0.35 * high_vol_weight) + (0.35 * shock_weight) + (0.20 * breakout_weight) + (0.15 * news_shock)
    envelopes: dict[int, VolatilityEnvelope] = {}
    for horizon in horizons:
        scale = np.sqrt(max(horizon, 1) / 5.0)
        expected_move_pct = float(max(1e-6, atr_pct * scale * multiplier))
        expected_move_abs = float(max(0.05, current_price * expected_move_pct))
        lower_bound = float(max(0.0, current_price - expected_move_abs))
        upper_bound = float(current_price + expected_move_abs)
        large_move_probability = float(np.clip((0.45 * high_vol_weight) + (0.30 * shock_weight) + (0.15 * tail_risk) + (0.10 * news_shock), 0.0, 1.0))
        envelopes[horizon] = VolatilityEnvelope(
            horizon_minutes=int(horizon),
            expected_move_abs=expected_move_abs,
            expected_move_pct=expected_move_pct,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            large_move_probability=large_move_probability,
            realized_volatility=float(realized_vol),
        )
    return envelopes
