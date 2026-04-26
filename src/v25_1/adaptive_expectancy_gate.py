from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import numpy as np

from src.v25_1.trade_cluster_filter import TradeClusterFilter


@dataclass(frozen=True)
class AdaptiveGateDecision:
    allow: bool
    threshold_used: float
    reason: str
    buy_threshold: float
    sell_threshold: float


class AdaptiveExpectancyGate:
    """
    Rules:
    - Separate BUY and SELL thresholds
    - Reject duplicate trades from same regime cluster
    - Reject repeated SELL trades if SELL regime expectancy is negative
    - Increase threshold after 2 losses
    - Decrease threshold slightly after 3 wins
    """

    def __init__(
        self,
        *,
        regime_thresholds: Mapping[str, float],
        sell_threshold_buffer: float = 0.02,
        cluster_filter: TradeClusterFilter | None = None,
    ):
        self.regime_thresholds = {str(key).lower(): float(value) for key, value in dict(regime_thresholds).items()}
        self.sell_threshold_buffer = float(sell_threshold_buffer)
        self.cluster_filter = cluster_filter or TradeClusterFilter(max_age_minutes=45, price_radius=0.35)
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    @staticmethod
    def _clamp_01(value: Any) -> float:
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except Exception:
            return 0.0

    def _base_threshold(self, regime: str) -> float:
        normalized = str(regime or "unknown").strip().lower()
        return float(self.regime_thresholds.get(normalized, self.regime_thresholds.get("unknown", 0.75)))

    def _dynamic_adjustment(self) -> float:
        adjustment = 0.0
        if self.consecutive_losses >= 2:
            adjustment += 0.03
        if self.consecutive_wins >= 3:
            adjustment -= 0.015
        return adjustment

    def evaluate(
        self,
        *,
        regime: str,
        direction: str,
        score: float,
        timestamp: datetime,
        entry_price: float,
        sell_regime_expectancy: float,
    ) -> AdaptiveGateDecision:
        normalized_direction = str(direction).upper()
        base = self._base_threshold(regime)
        dynamic = self._dynamic_adjustment()
        buy_threshold = float(np.clip(base + dynamic, 0.55, 0.95))
        sell_threshold = float(np.clip(base + self.sell_threshold_buffer + dynamic, 0.58, 0.97))
        threshold_used = sell_threshold if normalized_direction == "SELL" else buy_threshold

        cluster = self.cluster_filter.evaluate(
            regime=str(regime),
            direction=normalized_direction,
            timestamp=timestamp,
            entry_price=float(entry_price),
        )
        if not cluster.allow:
            return AdaptiveGateDecision(False, threshold_used, cluster.reason, buy_threshold, sell_threshold)

        if normalized_direction == "SELL" and float(sell_regime_expectancy) < 0.0:
            return AdaptiveGateDecision(False, threshold_used, "sell_expectancy_negative_block", buy_threshold, sell_threshold)

        if self._clamp_01(score) < threshold_used:
            return AdaptiveGateDecision(False, threshold_used, "score_below_adaptive_threshold", buy_threshold, sell_threshold)

        return AdaptiveGateDecision(True, threshold_used, "gate_pass", buy_threshold, sell_threshold)

    def record_outcome(self, realized_r: float) -> None:
        if float(realized_r) > 0.0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            return
        self.consecutive_losses += 1
        self.consecutive_wins = 0
