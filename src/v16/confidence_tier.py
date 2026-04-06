from __future__ import annotations

from enum import Enum


class ConfidenceTier(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNCERTAIN = "uncertain"


TIER_COLORS = {
    ConfidenceTier.VERY_HIGH: "#00c853",
    ConfidenceTier.HIGH: "#64dd17",
    ConfidenceTier.MODERATE: "#ffd600",
    ConfidenceTier.LOW: "#ff6d00",
    ConfidenceTier.UNCERTAIN: "#d50000",
}

TIER_LABELS = {
    ConfidenceTier.VERY_HIGH: "VERY HIGH - strong 15m signal",
    ConfidenceTier.HIGH: "HIGH - good 15m signal",
    ConfidenceTier.MODERATE: "MODERATE - normal size only",
    ConfidenceTier.LOW: "LOW - reduce size or observe",
    ConfidenceTier.UNCERTAIN: "UNCERTAIN - wide cone, observe only",
}


def classify_confidence(
    cabr_score: float,
    bst_score: float,
    cone_width_pips: float,
    cpm_score: float = 0.5,
) -> ConfidenceTier:
    effective_width = float(cone_width_pips)
    predictability_bonus = 1.0 if float(cpm_score) >= 0.60 else 0.0
    if float(cabr_score) > 0.70 and float(bst_score) > 0.85 and effective_width < 6.0 - predictability_bonus:
        return ConfidenceTier.VERY_HIGH
    if float(cabr_score) > 0.62 and float(bst_score) > 0.75 and effective_width < 12.0 - predictability_bonus:
        return ConfidenceTier.HIGH
    if float(cabr_score) > 0.54 and float(bst_score) > 0.65:
        return ConfidenceTier.MODERATE
    if float(cabr_score) > 0.50:
        return ConfidenceTier.LOW
    return ConfidenceTier.UNCERTAIN
