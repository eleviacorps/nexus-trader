from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class PaperTradeConstraints:
    risk_per_trade: float = 0.0025
    max_simultaneous_trades: int = 3
    allow_pyramiding: bool = False


class PaperTradeEngine:
    def __init__(self, starting_balance: float = 10000.0, constraints: PaperTradeConstraints | None = None):
        self.constraints = constraints or PaperTradeConstraints()
        self.starting_balance = float(starting_balance)
        self.balance = float(starting_balance)
        self.open_positions: list[dict[str, Any]] = []
        self.closed_positions: list[dict[str, Any]] = []

    def can_open(self, signal: dict[str, Any]) -> tuple[bool, str]:
        direction = str(signal.get("direction", "HOLD")).upper()
        if direction not in {"BUY", "SELL"}:
            return False, "direction_not_tradeable"
        if len(self.open_positions) >= self.constraints.max_simultaneous_trades:
            return False, "max_simultaneous_trades_reached"
        if not self.constraints.allow_pyramiding:
            symbol = str(signal.get("symbol", "XAUUSD"))
            if any(str(pos.get("symbol")) == symbol and str(pos.get("direction")) == direction for pos in self.open_positions):
                return False, "pyramiding_blocked"
        return True, "ok"

    def open_trade(self, signal: dict[str, Any]) -> dict[str, Any]:
        allowed, reason = self.can_open(signal)
        if not allowed:
            return {"opened": False, "reason": reason}
        risk_amount = self.balance * float(self.constraints.risk_per_trade)
        stop_distance = max(float(signal.get("stop_distance", 1.0) or 1.0), 1e-6)
        size = risk_amount / stop_distance
        trade = {
            "trade_id": f"paper-{len(self.open_positions) + len(self.closed_positions) + 1}",
            "opened_at": datetime.now(tz=UTC).isoformat(),
            "symbol": str(signal.get("symbol", "XAUUSD")),
            "direction": str(signal.get("direction", "HOLD")).upper(),
            "entry_price": float(signal.get("entry_price", 0.0) or 0.0),
            "stop_loss": float(signal.get("stop_loss", 0.0) or 0.0),
            "take_profit": float(signal.get("take_profit", 0.0) or 0.0),
            "size": float(size),
            "risk_amount": float(risk_amount),
            "meta": dict(signal),
        }
        self.open_positions.append(trade)
        return {"opened": True, "trade": trade}

    def close_trade(self, trade_id: str, exit_price: float, reason: str = "manual_close") -> dict[str, Any]:
        for idx, position in enumerate(self.open_positions):
            if str(position.get("trade_id")) != str(trade_id):
                continue
            trade = self.open_positions.pop(idx)
            direction = str(trade.get("direction", "BUY")).upper()
            signed = 1.0 if direction == "BUY" else -1.0
            pnl = signed * (float(exit_price) - float(trade.get("entry_price", 0.0))) * float(trade.get("size", 0.0))
            self.balance += pnl
            closed = {
                **trade,
                "closed_at": datetime.now(tz=UTC).isoformat(),
                "exit_price": float(exit_price),
                "close_reason": reason,
                "pnl": float(pnl),
                "balance_after": float(self.balance),
            }
            self.closed_positions.append(closed)
            return {"closed": True, "trade": closed}
        return {"closed": False, "reason": "trade_id_not_found"}

    def summary(self) -> dict[str, Any]:
        wins = [t for t in self.closed_positions if float(t.get("pnl", 0.0)) > 0.0]
        losses = [t for t in self.closed_positions if float(t.get("pnl", 0.0)) <= 0.0]
        return {
            "starting_balance": float(self.starting_balance),
            "balance": float(self.balance),
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
            "win_rate": float(len(wins) / max(len(self.closed_positions), 1)),
            "gross_pnl": float(sum(float(t.get("pnl", 0.0)) for t in self.closed_positions)),
            "gross_loss": float(sum(float(t.get("pnl", 0.0)) for t in losses)),
        }

