from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return _utc_now()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class CircuitBreakerStatus:
    trading_allowed: bool
    state: str
    pause_until: str | None
    size_multiplier: float
    consecutive_losses: int
    rolling_win_rate_10: float
    daily_drawdown_pct: float
    low_regime_confidence_bars: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CircuitBreakerConfig:
    rolling_window: int = 10
    arm_after_losses: int = 2
    consecutive_loss_limit: int = 3
    consecutive_loss_pause_minutes: int = 30
    drawdown_limit_pct: float = -0.02
    drawdown_pause_hours: int = 24
    min_rolling_win_rate: float = 0.35
    rolling_pause_minutes: int = 60
    low_regime_confidence_threshold: float = 0.55
    low_regime_bars_limit: int = 5
    low_regime_pause_minutes: int = 15
    reduced_size_on_losses: float = 0.50
    reduced_size_on_drawdown: float = 0.25
    reduced_size_on_regime: float = 0.50


class DailyCircuitBreaker:
    """
    V22 execution guard for live and paper sessions.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self.current_day = _utc_now().date()
        self.recent_trades: deque[bool] = deque(maxlen=max(1, int(self.config.rolling_window)))
        self.consecutive_losses = 0
        self.daily_drawdown_pct = 0.0
        self.pause_until: datetime | None = None
        self.pause_reason = ""
        self.low_regime_confidence_bars = 0
        self.trigger_log: list[dict[str, Any]] = []
        self.size_multiplier = 1.0

    def _reset_if_new_day(self, timestamp: datetime) -> None:
        if timestamp.date() == self.current_day:
            return
        self.current_day = timestamp.date()
        self.recent_trades.clear()
        self.consecutive_losses = 0
        self.daily_drawdown_pct = 0.0
        self.pause_until = None
        self.pause_reason = ""
        self.low_regime_confidence_bars = 0
        self.size_multiplier = 1.0

    def _activate_pause(self, timestamp: datetime, *, duration: timedelta, reason: str, size_multiplier: float = 1.0) -> None:
        self.pause_until = timestamp + duration
        self.pause_reason = reason
        self.size_multiplier = min(self.size_multiplier, float(size_multiplier))
        self.trigger_log.append(
            {
                "triggered_at": timestamp.isoformat(),
                "pause_until": self.pause_until.isoformat(),
                "reason": reason,
                "size_multiplier": round(float(self.size_multiplier), 6),
            }
        )

    def record_trade(self, trade_result: Mapping[str, Any], *, timestamp: datetime | None = None) -> CircuitBreakerStatus:
        ts = _to_utc(timestamp or trade_result.get("exit_time") or trade_result.get("timestamp"))
        self._reset_if_new_day(ts)
        profitable = bool(trade_result.get("profitable", _safe_float(trade_result.get("pnl", trade_result.get("profit", 0.0))) > 0.0))
        pnl_pct = _safe_float(trade_result.get("pnl_pct", 0.0), 0.0)
        self.recent_trades.append(profitable)
        self.consecutive_losses = 0 if profitable else self.consecutive_losses + 1
        self.daily_drawdown_pct += pnl_pct
        if self.consecutive_losses >= int(self.config.consecutive_loss_limit):
            self._activate_pause(
                ts,
                duration=timedelta(minutes=int(self.config.consecutive_loss_pause_minutes)),
                reason="consecutive_losses",
                size_multiplier=float(self.config.reduced_size_on_losses),
            )
        elif self.daily_drawdown_pct <= float(self.config.drawdown_limit_pct):
            self._activate_pause(
                ts,
                duration=timedelta(hours=int(self.config.drawdown_pause_hours)),
                reason="daily_drawdown",
                size_multiplier=float(self.config.reduced_size_on_drawdown),
            )
        elif len(self.recent_trades) == int(self.config.rolling_window) and self.rolling_win_rate_10 < float(self.config.min_rolling_win_rate):
            self._activate_pause(
                ts,
                duration=timedelta(minutes=int(self.config.rolling_pause_minutes)),
                reason="rolling_win_rate",
                size_multiplier=float(self.config.reduced_size_on_losses),
            )
        return self.status(ts)

    def update_regime_confidence(self, regime_confidence: float, *, timestamp: datetime | None = None) -> CircuitBreakerStatus:
        ts = _to_utc(timestamp)
        self._reset_if_new_day(ts)
        self.low_regime_confidence_bars = (
            self.low_regime_confidence_bars + 1
            if float(regime_confidence) < float(self.config.low_regime_confidence_threshold)
            else 0
        )
        if self.low_regime_confidence_bars >= int(self.config.low_regime_bars_limit):
            self._activate_pause(
                ts,
                duration=timedelta(minutes=int(self.config.low_regime_pause_minutes)),
                reason="uncertain_regime",
                size_multiplier=float(self.config.reduced_size_on_regime),
            )
        return self.status(ts)

    def sync_live_performance(self, live_performance: Mapping[str, Any], *, timestamp: datetime | None = None) -> CircuitBreakerStatus:
        ts = _to_utc(timestamp)
        self._reset_if_new_day(ts)
        rolling_win_rate = _safe_float(live_performance.get("rolling_win_rate_10"), self.rolling_win_rate_10)
        rolling_window = max(1, int(self.config.rolling_window))
        wins = int(round(max(0.0, min(1.0, rolling_win_rate)) * rolling_window))
        losses = max(0, rolling_window - wins)
        if wins > 0 or losses > 0:
            self.recent_trades = deque(([True] * wins) + ([False] * losses), maxlen=rolling_window)
        self.consecutive_losses = max(0, int(_safe_float(live_performance.get("consecutive_losses"), self.consecutive_losses)))
        balance = max(_safe_float(live_performance.get("balance"), 0.0), 0.0)
        equity = _safe_float(live_performance.get("equity"), balance)
        daily_pnl = _safe_float(live_performance.get("daily_pnl"), 0.0)
        drawdown_from_pnl = (daily_pnl / balance) if balance > 0.0 else self.daily_drawdown_pct
        drawdown_from_equity = ((equity - balance) / balance) if balance > 0.0 else self.daily_drawdown_pct
        self.daily_drawdown_pct = min(drawdown_from_pnl, drawdown_from_equity, self.daily_drawdown_pct)
        return self.status(ts)

    @property
    def rolling_win_rate_10(self) -> float:
        if not self.recent_trades:
            return 1.0
        return float(sum(1 for item in self.recent_trades if item) / len(self.recent_trades))

    def status(self, timestamp: datetime | None = None) -> CircuitBreakerStatus:
        ts = _to_utc(timestamp)
        self._reset_if_new_day(ts)
        if self.pause_until is not None and ts >= self.pause_until:
            self.pause_until = None
            self.pause_reason = ""
            self.size_multiplier = 1.0 if self.daily_drawdown_pct > -0.02 else self.size_multiplier
        reasons: list[str] = []
        state = "CLEAR"
        allowed = True
        if self.pause_until is not None and ts < self.pause_until:
            state = "PAUSED"
            allowed = False
            if self.pause_reason:
                reasons.append(self.pause_reason)
        elif self.consecutive_losses >= int(self.config.arm_after_losses):
            state = "ARMED"
            reasons.append("armed_consecutive_losses")
        if self.low_regime_confidence_bars >= int(self.config.low_regime_bars_limit):
            reasons.append("uncertain_regime")
        if len(self.recent_trades) == int(self.config.rolling_window) and self.rolling_win_rate_10 < float(self.config.min_rolling_win_rate):
            reasons.append("rolling_win_rate")
        if self.daily_drawdown_pct <= float(self.config.drawdown_limit_pct):
            reasons.append("daily_drawdown")
        return CircuitBreakerStatus(
            trading_allowed=allowed,
            state=state,
            pause_until=self.pause_until.isoformat() if self.pause_until is not None else None,
            size_multiplier=round(float(self.size_multiplier), 6),
            consecutive_losses=int(self.consecutive_losses),
            rolling_win_rate_10=round(float(self.rolling_win_rate_10), 6),
            daily_drawdown_pct=round(float(self.daily_drawdown_pct), 6),
            low_regime_confidence_bars=int(self.low_regime_confidence_bars),
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def snapshot(self, timestamp: datetime | None = None) -> dict[str, Any]:
        status = self.status(timestamp)
        return {
            "trading_allowed": status.trading_allowed,
            "state": status.state,
            "pause_until": status.pause_until,
            "size_multiplier": status.size_multiplier,
            "consecutive_losses": status.consecutive_losses,
            "rolling_win_rate_10": status.rolling_win_rate_10,
            "daily_drawdown_pct": status.daily_drawdown_pct,
            "low_regime_confidence_bars": status.low_regime_confidence_bars,
            "reasons": list(status.reasons),
            "trigger_log": list(self.trigger_log),
        }


__all__ = ["CircuitBreakerConfig", "CircuitBreakerStatus", "DailyCircuitBreaker"]
