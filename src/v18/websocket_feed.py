from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable


def seconds_to_next_15m(now: datetime | None = None) -> int:
    current = now or datetime.now(timezone.utc)
    seconds_in_hour = (current.minute * 60) + current.second
    seconds_into_bar = seconds_in_hour % 900
    remaining = 900 - seconds_into_bar
    return int(remaining if remaining > 0 else 900)


def get_position_pnl(position: dict[str, Any], current_price: float) -> float:
    entry = float(position.get("entry_price", current_price) or current_price)
    lot = float(position.get("lot", 0.01) or 0.01)
    direction = str(position.get("direction", "BUY")).upper()
    pip_size = float(position.get("pip_size", 0.1) or 0.1)
    pip_value = float(position.get("pip_value_per_lot", 10.0) or 10.0)
    direction_sign = 1.0 if direction == "BUY" else -1.0
    pnl_pips = ((float(current_price) - entry) * direction_sign) / max(pip_size, 1e-9)
    return round(float(pnl_pips * pip_value * lot), 2)


class LiveFeedManager:
    def __init__(self) -> None:
        self._connections: dict[Any, str] = {}
        self._paper_engine: Any = None
        self._last_price_by_symbol: dict[str, float] = {}

    def set_paper_engine(self, engine: Any) -> None:
        self._paper_engine = engine

    async def connect(self, ws: Any, symbol: str = "XAUUSD") -> None:
        await ws.accept()
        self._connections[ws] = str(symbol).upper()

    def set_symbol(self, ws: Any, symbol: str) -> None:
        if ws in self._connections:
            self._connections[ws] = str(symbol).upper()

    def disconnect(self, ws: Any) -> None:
        self._connections.pop(ws, None)

    async def broadcast(self, data: dict[str, Any]) -> None:
        disconnected = []
        for ws in list(self._connections):
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    def _position_payload(self, symbol: str, price: float) -> list[dict[str, Any]]:
        if self._paper_engine is None:
            return []
        try:
            state = self._paper_engine.state(current_prices={symbol: price})
        except Exception:
            return []
        rows: list[dict[str, Any]] = []
        for position in state.get("open_positions", []):
            if str(position.get("symbol", "")).upper() != str(symbol).upper():
                continue
            stop_loss = position.get("stop_loss", position.get("stop_price"))
            take_profit = position.get("take_profit", position.get("take_profit_price"))
            direction = str(position.get("direction", "BUY")).upper()
            sl_hit = stop_loss is not None and (
                (direction == "BUY" and float(price) <= float(stop_loss))
                or (direction == "SELL" and float(price) >= float(stop_loss))
            )
            tp_hit = take_profit is not None and (
                (direction == "BUY" and float(price) >= float(take_profit))
                or (direction == "SELL" and float(price) <= float(take_profit))
            )
            rows.append(
                {
                    "trade_id": position.get("trade_id"),
                    "symbol": str(position.get("symbol", symbol)).upper(),
                    "direction": direction,
                    "lot": float(position.get("lot", 0.0) or 0.0),
                    "entry_price": float(position.get("entry_price", price) or price),
                    "current_price": float(price),
                    "unrealized_pnl_usd": float(position.get("unrealized_pnl_usd", get_position_pnl(position, price)) or 0.0),
                    "stop_loss": None if stop_loss is None else float(stop_loss),
                    "take_profit": None if take_profit is None else float(take_profit),
                    "sl_hit": bool(sl_hit),
                    "tp_hit": bool(tp_hit),
                }
            )
            if (sl_hit or tp_hit) and self._paper_engine is not None:
                try:
                    self._paper_engine.close_position(str(position.get("trade_id")), exit_price=float(price))
                except Exception:
                    pass
        return rows

    def _paper_summary(self, symbol: str, price: float) -> dict[str, Any]:
        if self._paper_engine is None:
            return {}
        try:
            state = self._paper_engine.state(current_prices={symbol: price})
        except Exception:
            return {}
        summary = state.get("summary", {})
        return dict(summary) if isinstance(summary, dict) else {}

    async def heartbeat_loop(
        self,
        price_fn: Callable[[str], float],
        sqt_fn: Callable[[], dict[str, Any]],
    ) -> None:
        while True:
            disconnected = []
            for ws, symbol in list(self._connections.items()):
                try:
                    price = float(price_fn(symbol) or 0.0)
                    if price > 0.0:
                        self._last_price_by_symbol[symbol] = price
                    else:
                        price = float(self._last_price_by_symbol.get(symbol, 0.0))
                    payload = {
                        "type": "tick",
                        "symbol": symbol,
                        "price": price,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "positions": self._position_payload(symbol, price),
                        "paper_summary": self._paper_summary(symbol, price),
                        "sqt": sqt_fn(),
                        "bar_countdown": seconds_to_next_15m(),
                    }
                    payload["bar_progress"] = round((900 - int(payload["bar_countdown"])) / 900.0, 6)
                    await ws.send_json(payload)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                self.disconnect(ws)
            await asyncio.sleep(1)
