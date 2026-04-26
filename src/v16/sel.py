from __future__ import annotations

import numpy as np

from src.v16.confidence_tier import ConfidenceTier
from src.v16.sqt import SimulationQualityTracker


BASE_RISK_PCT = {
    "frequency": {
        "very_high": 0.020,
        "high": 0.015,
        "moderate": 0.010,
        "low": 0.005,
        "uncertain": 0.000,
    },
    "precision": {
        "very_high": 0.040,
        "high": 0.030,
        "moderate": 0.010,
        "low": 0.000,
        "uncertain": 0.000,
    },
}

SQT_MULTIPLIERS = {
    "HOT": 1.30,
    "GOOD": 1.00,
    "NEUTRAL": 0.80,
    "COLD": 0.50,
}


def should_execute(
    confidence_tier: ConfidenceTier,
    mode: str,
    sqt: SimulationQualityTracker,
) -> tuple[bool, str]:
    if sqt.should_pause():
        return False, "sqt_cold_streak"
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "precision":
        if confidence_tier in {ConfidenceTier.VERY_HIGH, ConfidenceTier.HIGH}:
            return True, "precision_mode_high_confidence"
        return False, f"precision_mode_below_threshold_{confidence_tier.value}"
    if normalized_mode == "frequency":
        if confidence_tier == ConfidenceTier.UNCERTAIN:
            return False, "frequency_mode_uncertain"
        return True, f"frequency_mode_{confidence_tier.value}"
    return False, "unknown_mode"


def sel_lot_size(
    equity: float,
    confidence_tier: ConfidenceTier,
    sqt_label: str,
    mode: str = "frequency",
    stop_pips: float = 20.0,
    pip_value_per_lot: float = 10.0,
    min_lot: float = 0.05,
    max_lot: float = 2.0,
) -> float:
    base_pct = BASE_RISK_PCT.get(str(mode).strip().lower(), BASE_RISK_PCT["frequency"])
    risk_pct = float(base_pct.get(confidence_tier.value, 0.010))
    sqt_multiplier = float(SQT_MULTIPLIERS.get(str(sqt_label).upper(), 1.0))
    final_pct = min(risk_pct * sqt_multiplier, 0.030)
    risk_amount = float(equity) * final_pct
    lot = risk_amount / max(float(stop_pips) * float(pip_value_per_lot), 1e-6)
    return round(float(np.clip(lot, float(min_lot), float(max_lot))), 2) if final_pct > 0.0 else 0.0
