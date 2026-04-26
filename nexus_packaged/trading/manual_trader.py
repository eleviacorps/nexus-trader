"""Manual trading interface for TUI/API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from nexus_packaged.trading.trade_manager import TradeManager, TradeRecord


@dataclass
class ManualOrderRequest:
    """Manual order request payload."""

    direction: str
    lot_size: float
    entry_type: str
    entry_price: float | None
    sl: float
    tp: float
    comment: str = ""
    expiry: datetime | None = None


class ManualTrader:
    """Handles validated manual trade placement/modification/closure."""

    def __init__(self, mt5, trade_manager: TradeManager, config: dict):
        self.mt5 = mt5
        self.trade_manager = trade_manager
        self.config = config
        self.broker_cfg = dict(config.get("broker", {}))
        self.mt5_cfg = dict(config.get("mt5", {}))

    def _validate_lot(self, lot: float) -> None:
        lot_min = float(self.broker_cfg.get("lot_min", 0.01))
        lot_max = float(self.broker_cfg.get("lot_max", 500.0))
        lot_step = float(self.broker_cfg.get("lot_step", 0.01))
        if lot < lot_min or lot > lot_max:
            raise ValueError(f"lot_size must be within [{lot_min}, {lot_max}]")
        step_units = round(lot / lot_step)
        if abs(step_units * lot_step - lot) > 1e-9:
            raise ValueError(f"lot_size must be a multiple of lot_step={lot_step}")

    def _validate_sides(self, direction: str, entry: float, sl: float, tp: float) -> None:
        direction = direction.upper()
        if direction not in {"BUY", "SELL"}:
            raise ValueError("direction must be BUY or SELL")
        if direction == "BUY":
            if not (sl < entry):
                raise ValueError("Invalid SL: for BUY, stop loss must be below entry.")
            if not (tp > entry):
                raise ValueError("Invalid TP: for BUY, take profit must be above entry.")
        else:
            if not (sl > entry):
                raise ValueError("Invalid SL: for SELL, stop loss must be above entry.")
            if not (tp < entry):
                raise ValueError("Invalid TP: for SELL, take profit must be below entry.")

    def _validate_stops_level(self, entry: float, sl: float, tp: float) -> None:
        points = float(self.broker_cfg.get("stops_level_points", 10))
        point_size = 0.01
        min_distance = points * point_size
        if abs(entry - sl) < min_distance:
            raise ValueError(f"SL distance must be >= broker stops level ({points} points).")
        if abs(entry - tp) < min_distance:
            raise ValueError(f"TP distance must be >= broker stops level ({points} points).")

    def _current_price(self, direction: str) -> float:
        price = 0.0
        if hasattr(self.mt5, "get_last_tick_price"):
            try:
                price = float(self.mt5.get_last_tick_price(direction))
            except Exception:  # noqa: BLE001
                price = 0.0
        if price <= 0:
            price = float(self.trade_manager._latest_price())  # noqa: SLF001
        return price

    def place_trade(self, order: ManualOrderRequest) -> TradeRecord:
        """Validate and place manual order."""
        self._validate_lot(order.lot_size)
        direction = order.direction.upper().strip()
        entry = float(order.entry_price) if order.entry_price is not None else self._current_price(direction)
        self._validate_sides(direction, entry, float(order.sl), float(order.tp))
        self._validate_stops_level(entry, float(order.sl), float(order.tp))

        execution_enabled = bool(self.mt5_cfg.get("execution_enabled", False))
        if execution_enabled and not bool(getattr(self.mt5, "is_connected", False)):
            raise ValueError("MT5 execution is enabled but terminal is disconnected/network unavailable.")

        # Unified TradeManager path.
        trade = self.trade_manager.open_trade(
            direction=direction,
            lot=float(order.lot_size),
            sl=float(order.sl),
            tp=float(order.tp),
            comment=order.comment,
        )
        trade.source = "manual"

        # Optional live execution (disabled by default).
        if execution_enabled and hasattr(self.mt5, "place_order"):
            request_payload = {
                "direction": direction,
                "lot_size": float(order.lot_size),
                "entry_type": order.entry_type,
                "entry_price": entry if order.entry_type != "market" else None,
                "sl": float(order.sl),
                "tp": float(order.tp),
                "comment": order.comment,
                "expiry": order.expiry.isoformat() if order.expiry else None,
            }
            # ManualTrader API is synchronous by design; best-effort async bridge.
            coro = self.mt5.place_order(request_payload)
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(coro)

                def _done_callback(done_task: asyncio.Task) -> None:
                    try:
                        _ = done_task.result()
                    except Exception as exc:  # noqa: BLE001
                        __import__("logging").getLogger("nexus.errors").warning(
                            "Manual MT5 execution failed for trade %s: %s",
                            trade.trade_id,
                            exc,
                        )

                task.add_done_callback(_done_callback)
            except RuntimeError:
                asyncio.run(coro)
        return trade

    def modify_trade(self, trade_id: str, new_sl: float, new_tp: float) -> bool:
        """Modify SL/TP for open trade."""
        open_trades = {trade.trade_id: trade for trade in self.trade_manager.get_open_trades()}
        if trade_id not in open_trades:
            return False
        trade = open_trades[trade_id]
        self._validate_sides(trade.direction, trade.entry_price, float(new_sl), float(new_tp))
        self._validate_stops_level(trade.entry_price, float(new_sl), float(new_tp))
        trade.sl = float(new_sl)
        trade.tp = float(new_tp)
        if bool(self.mt5_cfg.get("execution_enabled", False)) and hasattr(self.mt5, "modify_order"):
            coro = self.mt5.modify_order(ticket=trade_id, sl=float(new_sl), tp=float(new_tp))
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                asyncio.run(coro)
        return True

    def close_trade(self, trade_id: str) -> bool:
        """Close one trade by id."""
        open_ids = {trade.trade_id for trade in self.trade_manager.get_open_trades()}
        if trade_id not in open_ids:
            return False
        self.trade_manager.close_trade(trade_id, reason="MANUAL")
        if bool(self.mt5_cfg.get("execution_enabled", False)) and hasattr(self.mt5, "close_order"):
            coro = self.mt5.close_order(ticket=trade_id)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                asyncio.run(coro)
        return True

    def close_all_manual(self) -> list[str]:
        """Close all manual trades currently open."""
        to_close = [trade.trade_id for trade in self.trade_manager.get_open_trades() if trade.source == "manual"]
        for trade_id in to_close:
            self.close_trade(trade_id)
        return to_close
