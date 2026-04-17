from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence


@dataclass(frozen=True)
class MarketExecutionContext:
    spread: float
    slippage_estimate: float
    buy_threshold: float
    sell_min_regime_confidence: float = 0.68
    sell_min_tactical_cabr: float = 0.62
    sell_max_spread: float = 0.30
    sell_max_slippage: float = 0.10


@dataclass(frozen=True)
class StreakContext:
    consecutive_sell_trades: int
    consecutive_sell_losses: int
    cooldown_bars: int
    recent_sell_entries: Sequence[tuple[datetime, float]] = field(default_factory=tuple)


@dataclass(frozen=True)
class GuardDecision:
    allow: bool
    reason: str
    required_threshold: float
    sell_score: float
    adjusted_cooldown_bars: int


class SellBiasGuard:
    def __init__(self, max_consecutive_sells: int = 3, duplicate_cluster_radius: float = 0.25, duplicate_cluster_minutes: int = 10):
        self.max_consecutive_sells = int(max_consecutive_sells)
        self.duplicate_cluster_radius = float(duplicate_cluster_radius)
        self.duplicate_cluster_minutes = int(duplicate_cluster_minutes)

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def evaluate_sell(self, candidate: dict, market_exec_ctx: MarketExecutionContext, streak_ctx: StreakContext) -> GuardDecision:
        tactical_cabr = self._clamp_01(float(candidate.get("tactical_cabr", 0.0)))
        regime_confidence = self._clamp_01(float(candidate.get("regime_confidence", 0.0)))
        admission_score = self._clamp_01(float(candidate.get("admission_score", 0.0)))
        signal_direction = str(candidate.get("direction", "HOLD")).upper()

        if signal_direction != "SELL":
            return GuardDecision(
                allow=True,
                reason="direction_is_not_sell",
                required_threshold=market_exec_ctx.buy_threshold,
                sell_score=admission_score,
                adjusted_cooldown_bars=streak_ctx.cooldown_bars,
            )

        if streak_ctx.consecutive_sell_trades >= self.max_consecutive_sells:
            return GuardDecision(
                allow=False,
                reason="max_consecutive_sell_trades_reached",
                required_threshold=market_exec_ctx.buy_threshold + 0.05,
                sell_score=admission_score,
                adjusted_cooldown_bars=max(streak_ctx.cooldown_bars, 6),
            )

        if self._is_duplicate_sell_cluster(candidate, streak_ctx):
            return GuardDecision(
                allow=False,
                reason="sell_cluster_duplicate_blocked",
                required_threshold=market_exec_ctx.buy_threshold + 0.05,
                sell_score=admission_score,
                adjusted_cooldown_bars=streak_ctx.cooldown_bars,
            )

        if regime_confidence < market_exec_ctx.sell_min_regime_confidence:
            return GuardDecision(
                allow=False,
                reason="sell_regime_confidence_below_minimum",
                required_threshold=market_exec_ctx.buy_threshold + 0.05,
                sell_score=admission_score,
                adjusted_cooldown_bars=streak_ctx.cooldown_bars,
            )
        if tactical_cabr < market_exec_ctx.sell_min_tactical_cabr:
            return GuardDecision(
                allow=False,
                reason="sell_tactical_cabr_below_minimum",
                required_threshold=market_exec_ctx.buy_threshold + 0.05,
                sell_score=admission_score,
                adjusted_cooldown_bars=streak_ctx.cooldown_bars,
            )
        if float(market_exec_ctx.spread) > market_exec_ctx.sell_max_spread:
            return GuardDecision(
                allow=False,
                reason="sell_spread_too_wide",
                required_threshold=market_exec_ctx.buy_threshold + 0.05,
                sell_score=admission_score,
                adjusted_cooldown_bars=streak_ctx.cooldown_bars,
            )
        if float(market_exec_ctx.slippage_estimate) > market_exec_ctx.sell_max_slippage:
            return GuardDecision(
                allow=False,
                reason="sell_slippage_too_high",
                required_threshold=market_exec_ctx.buy_threshold + 0.05,
                sell_score=admission_score,
                adjusted_cooldown_bars=streak_ctx.cooldown_bars,
            )

        sell_score = self._clamp_01((0.50 * admission_score) + (0.25 * tactical_cabr) + (0.25 * regime_confidence))
        required_threshold = float(market_exec_ctx.buy_threshold) + 0.05
        if sell_score < required_threshold:
            return GuardDecision(
                allow=False,
                reason="sell_score_below_buy_threshold_plus_buffer",
                required_threshold=required_threshold,
                sell_score=sell_score,
                adjusted_cooldown_bars=streak_ctx.cooldown_bars,
            )

        cooldown = int(streak_ctx.cooldown_bars)
        if streak_ctx.consecutive_sell_losses >= 2:
            cooldown = max(cooldown, cooldown + 3)
        return GuardDecision(
            allow=True,
            reason="sell_guard_pass",
            required_threshold=required_threshold,
            sell_score=sell_score,
            adjusted_cooldown_bars=cooldown,
        )

    def _is_duplicate_sell_cluster(self, candidate: dict, streak_ctx: StreakContext) -> bool:
        timestamp = candidate.get("timestamp")
        entry_price = candidate.get("entry_price")
        if timestamp is None or entry_price is None:
            return False
        try:
            entry_value = float(entry_price)
        except Exception:
            return False
        for previous_ts, previous_entry in streak_ctx.recent_sell_entries:
            minutes = abs((timestamp - previous_ts).total_seconds()) / 60.0
            if minutes > self.duplicate_cluster_minutes:
                continue
            if abs(entry_value - float(previous_entry)) <= self.duplicate_cluster_radius:
                return True
        return False

