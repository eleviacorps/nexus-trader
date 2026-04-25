"""Async MetaTrader5 connector."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd


class MT5Connector:
    """Async wrapper for MetaTrader5 API."""

    def __init__(self, config: dict):
        self.config = config
        self.mt5_cfg = dict(config.get("mt5", {}))
        self._connected = False
        self._mt5 = None
        self._tick_task: asyncio.Task | None = None
        self.logger = logging.getLogger("nexus.system")
        self.error_logger = logging.getLogger("nexus.errors")

    @staticmethod
    def _mask_password(password: str) -> str:
        """Return a masked password for safe UI/API responses."""
        pwd = str(password or "")
        if not pwd:
            return ""
        if len(pwd) <= 2:
            return "*" * len(pwd)
        return f"{pwd[:1]}{'*' * (len(pwd) - 2)}{pwd[-1:]}"

    def get_runtime_config(self) -> dict[str, Any]:
        """Return current MT5 runtime configuration with masked password."""
        return {
            "login": int(self.mt5_cfg.get("login", 0)),
            "password_masked": self._mask_password(str(self.mt5_cfg.get("password", ""))),
            "server": str(self.mt5_cfg.get("server", "")),
            "execution_enabled": bool(self.mt5_cfg.get("execution_enabled", False)),
            "reconnect_attempts": int(self.mt5_cfg.get("reconnect_attempts", 3)),
            "reconnect_delay_seconds": int(self.mt5_cfg.get("reconnect_delay_seconds", 5)),
            "connected": bool(self._connected),
        }

    def update_runtime_config(self, update: dict[str, Any]) -> None:
        """Update MT5 runtime config in-memory without writing to disk."""
        normalized: dict[str, Any] = {}
        if "login" in update and update["login"] is not None:
            normalized["login"] = int(update["login"])
        if "password" in update and update["password"] is not None:
            normalized["password"] = str(update["password"])
        if "server" in update and update["server"] is not None:
            normalized["server"] = str(update["server"])
        if "execution_enabled" in update and update["execution_enabled"] is not None:
            normalized["execution_enabled"] = bool(update["execution_enabled"])
        if "reconnect_attempts" in update and update["reconnect_attempts"] is not None:
            normalized["reconnect_attempts"] = int(update["reconnect_attempts"])
        if "reconnect_delay_seconds" in update and update["reconnect_delay_seconds"] is not None:
            normalized["reconnect_delay_seconds"] = int(update["reconnect_delay_seconds"])
        if not normalized:
            return
        self.mt5_cfg.update(normalized)
        self.config.setdefault("mt5", {}).update(normalized)
        self.logger.info(
            "MT5 runtime configuration updated (login=%s, server=%s, execution_enabled=%s)",
            self.mt5_cfg.get("login", 0),
            self.mt5_cfg.get("server", ""),
            self.mt5_cfg.get("execution_enabled", False),
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Connect to MT5 with retry/backoff."""
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.error_logger.warning("MetaTrader5 module unavailable: %s", exc)
            return False
        self._mt5 = mt5
        attempts = int(self.mt5_cfg.get("reconnect_attempts", 3))
        delay = int(self.mt5_cfg.get("reconnect_delay_seconds", 5))
        login = int(self.mt5_cfg.get("login", 0))
        password = str(self.mt5_cfg.get("password", ""))
        server = str(self.mt5_cfg.get("server", ""))
        for attempt in range(1, attempts + 1):
            ok = await asyncio.to_thread(mt5.initialize)
            if ok and login and password and server:
                ok = await asyncio.to_thread(mt5.login, login, password=password, server=server)
            if ok:
                self._connected = True
                self.logger.info("MT5 connected")
                return True
            await asyncio.sleep(delay)
        self._connected = False
        return False

    async def disconnect(self) -> None:
        """Shutdown MT5 connection and tick stream."""
        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None
        if self._mt5 is not None:
            try:
                await asyncio.to_thread(self._mt5.shutdown)
            except Exception:  # noqa: BLE001
                pass
        self._connected = False

    async def reconnect(self) -> bool:
        """Reconnect MT5 using the current runtime credentials."""
        await self.disconnect()
        return await self.connect()

    async def get_latest_bars(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        """Fetch latest bars from MT5."""
        if not self._connected or self._mt5 is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        rates = await asyncio.to_thread(self._mt5.copy_rates_from_pos, symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(rates)
        frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame = frame.set_index("timestamp")
        frame = frame.rename(columns={"tick_volume": "volume"})
        return frame[["open", "high", "low", "close", "volume"]]

    async def stream_ticks(self, symbol: str, callback: Callable) -> None:
        """Stream ticks and invoke callback on each new tick."""
        if not self._connected or self._mt5 is None:
            return

        async def _run() -> None:
            last_ts = None
            while self._connected:
                tick = await asyncio.to_thread(self._mt5.symbol_info_tick, symbol)
                if tick is None:
                    await asyncio.sleep(0.25)
                    continue
                ts = getattr(tick, "time_msc", getattr(tick, "time", None))
                if ts == last_ts:
                    await asyncio.sleep(0.1)
                    continue
                last_ts = ts
                payload = {
                    "timestamp": datetime.fromtimestamp(float(getattr(tick, "time", 0)), tz=timezone.utc).isoformat(),
                    "bid": float(getattr(tick, "bid", 0.0)),
                    "ask": float(getattr(tick, "ask", 0.0)),
                    "last": float(getattr(tick, "last", 0.0)),
                    "volume": float(getattr(tick, "volume", 0.0)),
                }
                result = callback(payload)
                if asyncio.iscoroutine(result):
                    await result
                await asyncio.sleep(0.05)

        self._tick_task = asyncio.create_task(_run(), name="nexus_mt5_ticks")
        await self._tick_task

    async def place_order(self, request: dict[str, Any]) -> dict:
        """Place an order if execution is enabled."""
        if not bool(self.mt5_cfg.get("execution_enabled", False)):
            raise RuntimeError("execution_disabled")
        if not self._connected or self._mt5 is None:
            raise RuntimeError("mt5_disconnected")
        # Broker-side request format should be adapted per account/symbol.
        # Keep this implementation explicit and transparent.
        symbol = request.get("symbol", self.config.get("data", {}).get("symbol", "XAUUSD"))
        direction = str(request.get("direction", "BUY")).upper()
        tick = await asyncio.to_thread(self._mt5.symbol_info_tick, symbol)
        if tick is None:
            raise RuntimeError("symbol_tick_unavailable")
        price = float(tick.ask if direction == "BUY" else tick.bid)
        order_type = self._mt5.ORDER_TYPE_BUY if direction == "BUY" else self._mt5.ORDER_TYPE_SELL
        payload = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(request.get("lot_size", 0.01)),
            "type": order_type,
            "price": price,
            "sl": float(request.get("sl", 0.0)),
            "tp": float(request.get("tp", 0.0)),
            "deviation": 20,
            "comment": str(request.get("comment", "nexus_manual")),
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = await asyncio.to_thread(self._mt5.order_send, payload)
        if result is None:
            raise RuntimeError("order_send_failed")
        return {"retcode": int(getattr(result, "retcode", -1)), "order": getattr(result, "order", None)}

    async def modify_order(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify order/position stops."""
        if not bool(self.mt5_cfg.get("execution_enabled", False)):
            raise RuntimeError("execution_disabled")
        if not self._connected or self._mt5 is None:
            return False
        payload = {
            "action": self._mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "sl": float(sl),
            "tp": float(tp),
        }
        result = await asyncio.to_thread(self._mt5.order_send, payload)
        return bool(result is not None and int(getattr(result, "retcode", -1)) in {10009, 10008})

    async def close_order(self, ticket: int) -> bool:
        """Close position by ticket."""
        if not bool(self.mt5_cfg.get("execution_enabled", False)):
            raise RuntimeError("execution_disabled")
        if not self._connected or self._mt5 is None:
            return False
        positions = await asyncio.to_thread(self._mt5.positions_get, ticket=int(ticket))
        if not positions:
            return False
        position = positions[0]
        symbol = position.symbol
        volume = float(position.volume)
        tick = await asyncio.to_thread(self._mt5.symbol_info_tick, symbol)
        if tick is None:
            return False
        direction_close = self._mt5.ORDER_TYPE_SELL if int(position.type) == self._mt5.ORDER_TYPE_BUY else self._mt5.ORDER_TYPE_BUY
        price = float(tick.bid if direction_close == self._mt5.ORDER_TYPE_SELL else tick.ask)
        payload = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "symbol": symbol,
            "volume": volume,
            "type": direction_close,
            "price": price,
            "deviation": 20,
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = await asyncio.to_thread(self._mt5.order_send, payload)
        return bool(result is not None and int(getattr(result, "retcode", -1)) in {10009, 10008})

    async def get_account_info(self) -> dict:
        """Return account metrics."""
        if not self._connected or self._mt5 is None:
            return {
                "equity": 0.0,
                "balance": 0.0,
                "margin": 0.0,
                "free_margin": 0.0,
                "leverage": 0,
                "currency": "USD",
            }
        info = await asyncio.to_thread(self._mt5.account_info)
        if info is None:
            return {
                "equity": 0.0,
                "balance": 0.0,
                "margin": 0.0,
                "free_margin": 0.0,
                "leverage": 0,
                "currency": "USD",
            }
        return {
            "equity": float(getattr(info, "equity", 0.0)),
            "balance": float(getattr(info, "balance", 0.0)),
            "margin": float(getattr(info, "margin", 0.0)),
            "free_margin": float(getattr(info, "margin_free", 0.0)),
            "leverage": int(getattr(info, "leverage", 0)),
            "currency": str(getattr(info, "currency", "USD")),
        }

    async def get_open_positions(self) -> list[dict]:
        """Return open positions from MT5."""
        if not self._connected or self._mt5 is None:
            return []
        positions = await asyncio.to_thread(self._mt5.positions_get)
        if not positions:
            return []
        out: list[dict] = []
        for pos in positions:
            out.append(
                {
                    "ticket": int(getattr(pos, "ticket", 0)),
                    "symbol": str(getattr(pos, "symbol", "")),
                    "type": int(getattr(pos, "type", 0)),
                    "volume": float(getattr(pos, "volume", 0.0)),
                    "price_open": float(getattr(pos, "price_open", 0.0)),
                    "sl": float(getattr(pos, "sl", 0.0)),
                    "tp": float(getattr(pos, "tp", 0.0)),
                    "profit": float(getattr(pos, "profit", 0.0)),
                }
            )
        return out

    def get_last_tick_price(self, direction: str = "BUY") -> float:
        """Sync helper for manual trader dialogs."""
        if not self._connected or self._mt5 is None:
            return 0.0
        symbol = self.config.get("data", {}).get("symbol", "XAUUSD")
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return 0.0
        return float(tick.ask if str(direction).upper() == "BUY" else tick.bid)

    def get_live_price(self) -> float:
        """Return live MT5 tick last price (single source of truth)."""
        if not self._connected or self._mt5 is None:
            return 0.0
        symbol = self.config.get("data", {}).get("symbol", "XAUUSD")
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return 0.0
        last = float(getattr(tick, "last", 0.0))
        if last > 0:
            return last
        # Some brokers may not populate `last` for FX/CFD symbols.
        bid = float(getattr(tick, "bid", 0.0))
        ask = float(getattr(tick, "ask", 0.0))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return 0.0
