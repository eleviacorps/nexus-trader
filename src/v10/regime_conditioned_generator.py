from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if not np.isfinite(number):
        return default
    return number


@dataclass(frozen=True)
class GenerationRegimeProfile:
    label: str
    volatility_scale: float
    transition_risk: float
    trend_strength: float
    temperature_floor: float
    temperature_ceiling: float
    target_branch_count: int
    minority_target_share: float
    cone_width_target: float


def infer_generation_regime(row: Mapping[str, object]) -> GenerationRegimeProfile:
    label = str(
        row.get("dominant_regime")
        or row.get("hmm_dominant_regime")
        or row.get("detected_regime")
        or "range"
    )
    transition_risk = np.clip(
        _safe_float(row.get("hmm_transition_risk"), _safe_float(row.get("quant_transition_risk"), 0.2)),
        0.0,
        1.0,
    )
    volatility_scale = float(
        np.clip(
            _safe_float(row.get("volatility_scale"), max(_safe_float(row.get("atr_pct"), 0.003) * 100.0, 0.25)),
            0.25,
            4.0,
        )
    )
    trend_strength = float(
        np.clip(
            abs(_safe_float(row.get("quant_trend_score"), _safe_float(row.get("model_direction_prob_15m"), 0.5) - 0.5) * 2.0),
            0.0,
            1.0,
        )
    )
    cone_width_target = float(np.clip((volatility_scale / 100.0) * (1.45 + 0.65 * transition_risk), 0.004, 0.035))
    if label in {"bullish_trend", "bearish_trend", "breakout"}:
        floor, ceiling = 0.85, 1.55
        target_branch_count = 10
        minority_target_share = 0.18
    elif label in {"panic_news_shock", "false_breakout"}:
        floor, ceiling = 1.00, 1.85
        target_branch_count = 12
        minority_target_share = 0.28
    elif label in {"mean_reversion", "range", "low_volatility_drift"}:
        floor, ceiling = 0.70, 1.20
        target_branch_count = 8
        minority_target_share = 0.22
    else:
        floor, ceiling = 0.80, 1.45
        target_branch_count = 10
        minority_target_share = 0.20
    ceiling = float(np.clip(ceiling + 0.25 * transition_risk + 0.12 * trend_strength, floor + 0.1, 2.1))
    return GenerationRegimeProfile(
        label=label,
        volatility_scale=volatility_scale,
        transition_risk=transition_risk,
        trend_strength=trend_strength,
        temperature_floor=float(floor),
        temperature_ceiling=ceiling,
        target_branch_count=int(target_branch_count),
        minority_target_share=float(minority_target_share),
        cone_width_target=cone_width_target,
    )


def temperature_schedule(profile: GenerationRegimeProfile) -> list[float]:
    steps = 4 if profile.target_branch_count >= 10 and profile.transition_risk >= 0.55 else 3
    return np.linspace(profile.temperature_floor, profile.temperature_ceiling, steps, dtype=np.float32).round(4).tolist()
