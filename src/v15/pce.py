from __future__ import annotations

import numpy as np


class PredictabilityConditionedExecution:
    def __init__(
        self,
        cpm_threshold: float = 0.60,
        min_agreement: float = 0.55,
        min_regime_stability_bars: int = 8,
        atr_percentile_low: float = 30.0,
        atr_percentile_high: float = 88.0,
    ):
        self.cpm_threshold = float(cpm_threshold)
        self.min_agreement = float(min_agreement)
        self.min_regime_stability = int(min_regime_stability_bars)
        self.atr_low = float(atr_percentile_low)
        self.atr_high = float(atr_percentile_high)

    def adjusted_cpm_score(self, cpm_score: float, *, eci_boost: float = 0.0, avoid_window: bool = False) -> float:
        if avoid_window:
            return 0.0
        return float(np.clip(float(cpm_score) + float(eci_boost), 0.0, 1.0))

    def is_predictable_window(
        self,
        cpm_score: float,
        cpm_agreement: float,
        regime_stability_bars: int,
        atr_percentile: float,
    ) -> tuple[bool, str]:
        if float(cpm_score) < self.cpm_threshold:
            return False, f"cpm_below_threshold_{float(cpm_score):.3f}"
        if float(cpm_agreement) < self.min_agreement:
            return False, f"low_agreement_{float(cpm_agreement):.3f}"
        if int(regime_stability_bars) < self.min_regime_stability:
            return False, f"regime_unstable_{int(regime_stability_bars)}bars"
        if not (self.atr_low <= float(atr_percentile) <= self.atr_high):
            return False, f"atr_outside_range_{float(atr_percentile):.0f}pct"
        return True, "predictable_window"

    def tune_threshold_for_participation(self, cpm_scores: np.ndarray, target_rate: float = 0.20) -> float:
        scores = np.asarray(cpm_scores, dtype=np.float32)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            return self.cpm_threshold
        clipped_target = float(np.clip(target_rate, 0.0, 1.0))
        if clipped_target <= 0.0:
            return float(scores.max())
        if clipped_target >= 1.0:
            return 0.0
        return float(np.quantile(scores, max(0.0, 1.0 - clipped_target)))
