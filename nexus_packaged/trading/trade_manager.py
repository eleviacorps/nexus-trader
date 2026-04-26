"""Trade state management with risk-limit enforcement."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


@dataclass
class TradeRecord:
    """Unified trade record for auto/manual/backtest."""

    trade_id: str
    entry_time: datetime
    exit_time: datetime | None
    direction: str
    lot_size: float
    entry_price: float
    exit_price: float | None
    sl: float
    tp: float
    pnl_usd: float
    pnl_pips: float
    exit_reason: str
    confidence: float
    margin_used: float
    commission: float
    source: str
    comment: str = ""

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "direction": self.direction,
            "lot_size": self.lot_size,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "sl": self.sl,
            "tp": self.tp,
            "pnl_usd": self.pnl_usd,
            "pnl_pips": self.pnl_pips,
            "exit_reason": self.exit_reason,
            "confidence": self.confidence,
            "margin_used": self.margin_used,
            "commission": self.commission,
            "source": self.source,
            "comment": self.comment,
        }


@dataclass
class SessionSummary:
    """Session-level metrics for TUI/API panels."""

    open_trades: int
    total_trades: int
    daily_trades: int
    daily_pnl_usd: float
    daily_pnl_pct: float
    session_drawdown_pct: float
    equity: float
    peak_equity: float


class TradeManager:
    """In-memory trade manager with live risk checks."""

    def __init__(
        self,
        *,
        initial_equity: float = 10000.0,
        pip_size: float = 0.01,
        pip_value_per_lot: float = 1.0,
        leverage: int = 200,
        contract_size: float = 100.0,
    ) -> None:
        self.logger = logging.getLogger("nexus.trades")
        self.error_logger = logging.getLogger("nexus.errors")
        self._open: dict[str, TradeRecord] = {}
        self._closed: list[TradeRecord] = []
        self._initial_equity = float(initial_equity)
        self._equity = float(initial_equity)
        self._peak_equity = float(initial_equity)
        self._pip_size = float(pip_size)
        self._pip_value_per_lot = float(pip_value_per_lot)
        self._leverage = int(max(1, leverage))
        self._contract_size = float(contract_size)
        self._price_provider: Callable[[], float] | None = None
        self._session_start = datetime.now(timezone.utc)
        self._json_log_path = Path("nexus_packaged/logs/trades.log")
        self._json_log_path.parent.mkdir(parents=True, exist_ok=True)

    def set_price_provider(self, provider: Callable[[], float]) -> None:
        """Inject latest-price callback for PnL calculation."""
        self._price_provider = provider

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _latest_price(self) -> float:
        if self._price_provider is None:
            return 0.0
        try:
            return float(self._price_provider())
        except Exception:  # noqa: BLE001
            return 0.0

    def _log(self, payload: dict) -> None:
        payload["timestamp"] = self._now().isoformat()
        self.logger.info(json.dumps(payload))
        try:
            with self._json_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception:  # noqa: BLE001
            self.error_logger.exception("Failed to append trade JSON log line.")

    def open_trade(self, direction: str, lot: float, sl: float, tp: float, comment: str) -> TradeRecord:
        """Open a new trade record."""
        direction = direction.upper().strip()
        if direction not in {"BUY", "SELL"}:
            raise ValueError("direction must be BUY or SELL")
        entry = self._latest_price()
        margin_used = (float(lot) * self._contract_size * max(entry, 1e-9)) / self._leverage
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            entry_time=self._now(),
            exit_time=None,
            direction=direction,
            lot_size=float(lot),
            entry_price=float(entry),
            exit_price=None,
            sl=float(sl),
            tp=float(tp),
            pnl_usd=0.0,
            pnl_pips=0.0,
            exit_reason="",
            confidence=0.0,
            margin_used=float(margin_used),
            commission=0.0,
            source="auto",
            comment=comment,
        )
        self._open[trade.trade_id] = trade
        self._log(
            {
                "event": "OPEN",
                "trade_id": trade.trade_id,
                "source": trade.source,
                "direction": trade.direction,
                "lot": trade.lot_size,
                "entry": trade.entry_price,
                "sl": trade.sl,
                "tp": trade.tp,
                "confidence": trade.confidence,
                "margin_used": trade.margin_used,
                "reason": "OPEN",
            }
        )
        return trade

    def close_trade(self, trade_id: str, reason: str) -> None:
        """Close an existing open trade."""
        if trade_id not in self._open:
            return
        trade = self._open.pop(trade_id)
        exit_price = self._latest_price()
        sign = 1.0 if trade.direction == "BUY" else -1.0
        pnl_pips = sign * (exit_price - trade.entry_price) / self._pip_size
        pnl_usd = pnl_pips * self._pip_value_per_lot * trade.lot_size
        trade.exit_time = self._now()
        trade.exit_price = float(exit_price)
        trade.pnl_pips = float(pnl_pips)
        trade.pnl_usd = float(pnl_usd)
        trade.exit_reason = reason
        self._equity += trade.pnl_usd
        self._peak_equity = max(self._peak_equity, self._equity)
        self._closed.append(trade)
        self._log(
            {
                "event": "CLOSE",
                "trade_id": trade.trade_id,
                "source": trade.source,
                "direction": trade.direction,
                "lot": trade.lot_size,
                "entry": trade.entry_price,
                "sl": trade.sl,
                "tp": trade.tp,
                "confidence": trade.confidence,
                "margin_used": trade.margin_used,
                "reason": reason,
            }
        )

    def get_open_trades(self) -> list[TradeRecord]:
        """Return currently open trades."""
        return list(self._open.values())

    def get_trade_history(self, *, limit: int = 100, source: str | None = None) -> list[TradeRecord]:
        """Return recent closed trades."""
        history = self._closed
        if source:
            history = [trade for trade in history if trade.source == source]
        return history[-int(limit) :]

    def get_daily_trade_count(self) -> int:
        """Number of closed trades for current UTC day."""
        today = self._now().date()
        return len([trade for trade in self._closed if trade.exit_time and trade.exit_time.date() == today])

    def get_daily_pnl_pct(self) -> float:
        """Daily realized PnL percentage versus session start equity."""
        today = self._now().date()
        pnl = sum(
            trade.pnl_usd
            for trade in self._closed
            if trade.exit_time is not None and trade.exit_time.date() == today
        )
        return (pnl / max(1e-9, self._initial_equity)) * 100.0

    def get_session_summary(self) -> SessionSummary:
        """Collect summary stats for UI/API."""
        today = self._now().date()
        daily_pnl = sum(
            trade.pnl_usd
            for trade in self._closed
            if trade.exit_time is not None and trade.exit_time.date() == today
        )
        drawdown_pct = ((self._peak_equity - self._equity) / max(1e-9, self._peak_equity)) * 100.0
        return SessionSummary(
            open_trades=len(self._open),
            total_trades=len(self._closed),
            daily_trades=self.get_daily_trade_count(),
            daily_pnl_usd=float(daily_pnl),
            daily_pnl_pct=float(self.get_daily_pnl_pct()),
            session_drawdown_pct=float(drawdown_pct),
            equity=float(self._equity),
            peak_equity=float(self._peak_equity),
        )

    def check_risk_limits(self, config) -> tuple[bool, str]:
        """Check risk limits before opening a new trade."""
        if len(self._open) >= int(getattr(config, "max_open_trades", 3)):
            return False, "max_open_trades"
        max_daily = int(getattr(config, "max_daily_trades", 0))
        if max_daily > 0 and self.get_daily_trade_count() >= max_daily:
            return False, "max_daily_trades"
        max_daily_loss_pct = float(getattr(config, "max_daily_loss_pct", 0.0))
        if max_daily_loss_pct > 0 and self.get_daily_pnl_pct() <= -abs(max_daily_loss_pct):
            return False, "max_daily_loss_pct"
        drawdown = ((self._peak_equity - self._equity) / max(1e-9, self._peak_equity)) * 100.0
        max_dd = float(getattr(config, "max_drawdown_pct", 0.0))
        if max_dd > 0 and drawdown >= abs(max_dd):
            return False, "max_drawdown_pct"
        return True, ""

    def close_all(self, reason: str = "MANUAL") -> list[str]:
        """Close all open trades and return closed IDs."""
        ids = list(self._open.keys())
        for trade_id in ids:
            self.close_trade(trade_id, reason)
        return ids
