from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


REGIME_THRESHOLDS_DEFAULT: dict[str, float] = {
    "trend_up": 0.54,
    "trend_down": 0.64,
    "breakout": 0.58,
    "range": 0.60,
}


@dataclass(frozen=True)
class RegimeContext:
    regime: str
    regime_confidence: float
    rolling_expectancy: float
    rolling_win_rate: float


@dataclass(frozen=True)
class RegimeThresholdDecision:
    threshold: float
    enabled: bool
    strategic_only: bool
    reason: str


@dataclass
class RegimeThresholdRouter:
    thresholds: dict[str, float] = field(default_factory=lambda: dict(REGIME_THRESHOLDS_DEFAULT))
    min_threshold: float = 0.45
    max_threshold: float = 0.80

    def with_threshold_overrides(self, overrides: Mapping[str, float] | None) -> "RegimeThresholdRouter":
        if not overrides:
            return self
        for key, value in overrides.items():
            normalized = self._normalize_regime(str(key))
            self.thresholds[normalized] = float(value)
        return self

    def route(self, context: RegimeContext) -> RegimeThresholdDecision:
        regime = self._normalize_regime(context.regime)
        if regime == "chop":
            return RegimeThresholdDecision(
                threshold=1.0,
                enabled=False,
                strategic_only=False,
                reason="regime=chop disabled by policy",
            )
        if regime == "unknown":
            return RegimeThresholdDecision(
                threshold=0.62,
                enabled=True,
                strategic_only=True,
                reason="regime=unknown strategic-only policy",
            )
        base = float(self.thresholds.get(regime, self.thresholds.get("range", 0.60)))
        confidence_adj = self._confidence_adjustment(context.regime_confidence)
        expectancy_adj = self._expectancy_adjustment(context.rolling_expectancy)
        winrate_adj = self._winrate_adjustment(context.rolling_win_rate)
        threshold = base - confidence_adj - expectancy_adj - winrate_adj
        threshold = max(self.min_threshold, min(self.max_threshold, threshold))
        return RegimeThresholdDecision(
            threshold=float(threshold),
            enabled=True,
            strategic_only=False,
            reason=(
                f"regime={regime} base={base:.3f} "
                f"adj_conf={confidence_adj:+.3f} adj_exp={expectancy_adj:+.3f} adj_wr={winrate_adj:+.3f}"
            ),
        )

    @staticmethod
    def _normalize_regime(regime: str) -> str:
        normalized = (regime or "unknown").strip().lower()
        aliases = {
            "trending_up": "trend_up",
            "trend": "trend_up",
            "trending_down": "trend_down",
            "downtrend": "trend_down",
            "mean_reversion": "range",
            "ranging": "range",
            "liquidity_sweep_reversal": "range",
            "trend_continuation": "trend_up",
            "breakout_shock": "breakout",
        }
        return aliases.get(normalized, normalized if normalized in {"trend_up", "trend_down", "breakout", "range", "chop", "unknown"} else "unknown")

    @staticmethod
    def _confidence_adjustment(regime_confidence: float) -> float:
        centered = float(regime_confidence) - 0.60
        return max(-0.03, min(0.03, centered * 0.10))

    @staticmethod
    def _expectancy_adjustment(rolling_expectancy: float) -> float:
        # Historical expectancy values are in R units and can be very small in this project;
        # scale to a bounded threshold shift.
        scaled = float(rolling_expectancy) * 25.0
        return max(-0.04, min(0.04, scaled))

    @staticmethod
    def _winrate_adjustment(rolling_win_rate: float) -> float:
        centered = float(rolling_win_rate) - 0.55
        return max(-0.03, min(0.03, centered * 0.25))

