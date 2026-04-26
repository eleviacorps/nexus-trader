from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.v25.mt5_bridge import MT5Bridge
from src.v25.paper_trade_engine import PaperTradeEngine


@dataclass(frozen=True)
class AutoExecutionResult:
    executed: bool
    channel: str
    reason: str
    payload: dict[str, Any]


class AutoExecutionEngine:
    def __init__(self, paper_engine: PaperTradeEngine, mt5_bridge: MT5Bridge | None = None):
        self.paper_engine = paper_engine
        self.mt5_bridge = mt5_bridge

    def execute(self, payload: dict[str, Any], execution_channel: str = "paper") -> AutoExecutionResult:
        signal = {
            "symbol": payload.get("symbol", "XAUUSD"),
            "direction": payload.get("direction", "HOLD"),
            "entry_price": payload.get("entry_price", payload.get("market_price", 0.0)),
            "stop_loss": payload.get("stop_loss", 0.0),
            "take_profit": payload.get("take_profit", 0.0),
            "stop_distance": abs(float(payload.get("entry_price", 0.0)) - float(payload.get("stop_loss", 0.0))),
            "reason": payload.get("reason", ""),
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "meta": dict(payload),
        }
        if execution_channel == "mt5":
            if self.mt5_bridge is None:
                return AutoExecutionResult(False, "mt5", "mt5_bridge_missing", payload)
            path = self.mt5_bridge.export_signal(signal)
            return AutoExecutionResult(True, "mt5", f"exported_to:{path}", signal)

        opened = self.paper_engine.open_trade(signal)
        if not opened.get("opened", False):
            return AutoExecutionResult(False, "paper", str(opened.get("reason", "paper_open_failed")), signal)
        return AutoExecutionResult(True, "paper", "paper_trade_opened", opened.get("trade", signal))

