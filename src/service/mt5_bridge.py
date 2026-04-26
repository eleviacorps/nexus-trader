from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.project_config import OUTPUTS_V21_DIR


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class MT5Bridge:
    def __init__(self, state_path: Path | None = None) -> None:
        self.state_path = Path(state_path or (OUTPUTS_V21_DIR / "mt5_bridge_state.json"))
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.state_path.exists():
            self._write_state(
                {
                    "connected": False,
                    "login": None,
                    "server": "",
                    "path": "",
                    "symbol_prefix": "",
                    "symbol_suffix": "",
                    "symbol_overrides": {},
                    "autotrade_enabled": False,
                    "autotrade_lot_mode": "range",
                    "autotrade_fixed_lot": None,
                    "autotrade_min_lot": 0.01,
                    "autotrade_max_lot": 0.05,
                    "last_error": "",
                    "last_action": "idle",
                    "last_order": None,
                    "last_bucket_by_symbol": {},
                    "updated_at": time.time(),
                }
            )

    def _import_mt5(self):
        try:
            import MetaTrader5 as mt5  # type: ignore

            return mt5
        except Exception:
            return None

    def _read_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "connected": False,
                "login": None,
                "server": "",
                "path": "",
                "symbol_prefix": "",
                "symbol_suffix": "",
                "symbol_overrides": {},
                "autotrade_enabled": False,
                "autotrade_lot_mode": "range",
                "autotrade_fixed_lot": None,
                "autotrade_min_lot": 0.01,
                "autotrade_max_lot": 0.05,
                "last_error": "",
                "last_action": "idle",
                "last_order": None,
                "last_bucket_by_symbol": {},
                "updated_at": time.time(),
            }

    def _write_state(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = time.time()
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _patch_state(self, **updates: Any) -> dict[str, Any]:
        with self._lock:
            state = self._read_state()
            state.update(updates)
            self._write_state(state)
            return state

    def _autotrade_config_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "lot_mode": str(state.get("autotrade_lot_mode", "range") or "range"),
            "fixed_lot": _safe_float(state.get("autotrade_fixed_lot"), 0.0) or None,
            "min_lot": max(_safe_float(state.get("autotrade_min_lot", 0.01), 0.01), 0.01),
            "max_lot": max(_safe_float(state.get("autotrade_max_lot", 0.05), 0.05), 0.01),
        }

    def _ensure_initialized(self, path_override: str | None = None) -> tuple[Any | None, str]:
        mt5 = self._import_mt5()
        if mt5 is None:
            return None, "MetaTrader5 Python package is not installed."
        state = self._read_state()
        path = str(path_override if path_override is not None else state.get("path", "") or "").strip()
        try:
            initialized = mt5.initialize(path) if path else mt5.initialize()
        except Exception as exc:
            return mt5, str(exc)
        if not initialized:
            return mt5, f"MT5 initialize failed: {mt5.last_error()}"
        return mt5, ""

    def _connected_snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._read_state()
        mt5, init_error = self._ensure_initialized()
        installed = mt5 is not None
        connected = False
        account_payload: dict[str, Any] = {}
        terminal_payload: dict[str, Any] = {}
        if installed and not init_error:
            try:
                terminal = mt5.terminal_info()
                account = mt5.account_info()
                terminal_connected = bool(getattr(terminal, "connected", False)) if terminal else False
                connected = terminal_connected and bool(account)
                if terminal:
                    terminal_payload = {
                        "company": getattr(terminal, "company", ""),
                        "name": getattr(terminal, "name", ""),
                        "connected": bool(getattr(terminal, "connected", False)),
                        "trade_allowed": bool(getattr(terminal, "trade_allowed", False)),
                    }
                if account:
                    account_payload = {
                        "login": int(getattr(account, "login", 0) or 0),
                        "server": str(getattr(account, "server", "")),
                        "name": str(getattr(account, "name", "")),
                        "balance": _safe_float(getattr(account, "balance", 0.0)),
                        "equity": _safe_float(getattr(account, "equity", 0.0)),
                    }
            except Exception:
                connected = False
        with self._lock:
            state = self._read_state()
            state["connected"] = connected
            if account_payload.get("login"):
                state["login"] = account_payload["login"]
            if account_payload.get("server"):
                state["server"] = account_payload["server"]
            if init_error:
                state["last_error"] = init_error
            self._write_state(state)
        return {
            "installed": installed,
            "connected": connected,
            "login": account_payload.get("login") or state.get("login"),
            "server": account_payload.get("server") or state.get("server", ""),
            "path": state.get("path", ""),
            "symbol_prefix": state.get("symbol_prefix", ""),
            "symbol_suffix": state.get("symbol_suffix", ""),
            "symbol_overrides": dict(state.get("symbol_overrides", {}) or {}),
            "terminal": terminal_payload,
            "account": account_payload,
            "autotrade_enabled": bool(state.get("autotrade_enabled", False)),
            "autotrade_config": self._autotrade_config_from_state(state),
            "last_error": str(state.get("last_error", "")) if not init_error else init_error,
            "last_action": str(state.get("last_action", "idle")),
            "last_order": state.get("last_order"),
        }

    def status(self) -> dict[str, Any]:
        snapshot = self._connected_snapshot()
        if not snapshot["installed"]:
            snapshot["availability_reason"] = "MetaTrader5 Python package is not installed."
        elif not snapshot["connected"]:
            snapshot["availability_reason"] = "MT5 terminal is not connected yet."
        else:
            snapshot["availability_reason"] = ""
        return snapshot

    def connect(
        self,
        *,
        login: int,
        password: str,
        server: str,
        path: str = "",
        symbol_prefix: str = "",
        symbol_suffix: str = "",
        symbol_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        requested_path = str(path or "").strip()
        mt5, init_error = self._ensure_initialized(requested_path or None)
        if mt5 is None:
            state = self._patch_state(
                connected=False,
                login=int(login),
                server=str(server),
                path=str(path),
                symbol_prefix=str(symbol_prefix),
                symbol_suffix=str(symbol_suffix),
                symbol_overrides=dict(symbol_overrides or {}),
                last_error="MetaTrader5 Python package is not installed.",
                last_action="connect_failed",
            )
            return self.status() | {"ok": False, "detail": state["last_error"]}
        if init_error:
            self._patch_state(
                connected=False,
                login=int(login),
                server=str(server),
                path=str(path),
                symbol_prefix=str(symbol_prefix),
                symbol_suffix=str(symbol_suffix),
                symbol_overrides=dict(symbol_overrides or {}),
                last_error=init_error,
                last_action="connect_failed",
            )
            return self.status() | {"ok": False, "detail": init_error}

        try:
            terminal = mt5.terminal_info()
            account = mt5.account_info()
            current_login = int(getattr(account, "login", 0) or 0) if account else 0
            terminal_connected = bool(getattr(terminal, "connected", False)) if terminal else False
            if terminal_connected and current_login == int(login):
                self._patch_state(
                    connected=True,
                    login=int(login),
                    server=str(server),
                    path=str(path),
                    symbol_prefix=str(symbol_prefix),
                    symbol_suffix=str(symbol_suffix),
                    symbol_overrides=dict(symbol_overrides or {}),
                    last_error="",
                    last_action="connected",
                )
                return self.status() | {"ok": True, "detail": "MT5 connected."}
            authorized = mt5.login(int(login), password=str(password), server=str(server))
            if not authorized:
                error = f"MT5 login failed: {mt5.last_error()}"
                self._patch_state(
                    connected=False,
                    login=int(login),
                    server=str(server),
                    path=str(path),
                    symbol_prefix=str(symbol_prefix),
                    symbol_suffix=str(symbol_suffix),
                    symbol_overrides=dict(symbol_overrides or {}),
                    last_error=error,
                    last_action="connect_failed",
                )
                return self.status() | {"ok": False, "detail": error}
            self._patch_state(
                connected=True,
                login=int(login),
                server=str(server),
                path=str(path),
                symbol_prefix=str(symbol_prefix),
                symbol_suffix=str(symbol_suffix),
                symbol_overrides=dict(symbol_overrides or {}),
                last_error="",
                last_action="connected",
            )
            return self.status() | {"ok": True, "detail": "MT5 connected."}
        except Exception as exc:
            self._patch_state(
                connected=False,
                login=int(login),
                server=str(server),
                path=str(path),
                symbol_prefix=str(symbol_prefix),
                symbol_suffix=str(symbol_suffix),
                symbol_overrides=dict(symbol_overrides or {}),
                last_error=str(exc),
                last_action="connect_failed",
            )
            return self.status() | {"ok": False, "detail": str(exc)}

    def disconnect(self) -> dict[str, Any]:
        mt5 = self._import_mt5()
        if mt5 is not None:
            try:
                mt5.shutdown()
            except Exception:
                pass
        self._patch_state(connected=False, last_action="disconnected")
        return self.status() | {"ok": True, "detail": "MT5 disconnected."}

    def set_autotrade(
        self,
        enabled: bool,
        *,
        lot_mode: str | None = None,
        fixed_lot: float | None = None,
        min_lot: float | None = None,
        max_lot: float | None = None,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "autotrade_enabled": bool(enabled),
            "last_action": "autotrade_on" if enabled else "autotrade_off",
        }
        if lot_mode is not None:
            updates["autotrade_lot_mode"] = str(lot_mode or "range").lower()
        if fixed_lot is not None:
            updates["autotrade_fixed_lot"] = max(round(float(fixed_lot), 2), 0.01)
        if min_lot is not None:
            updates["autotrade_min_lot"] = max(round(float(min_lot), 2), 0.01)
        if max_lot is not None:
            updates["autotrade_max_lot"] = max(round(float(max_lot), 2), 0.01)
        state = self._patch_state(**updates)
        config = self._autotrade_config_from_state(state)
        if config["max_lot"] < config["min_lot"]:
            state = self._patch_state(autotrade_max_lot=config["min_lot"])
        return self.status() | {"ok": True, "autotrade_enabled": bool(enabled)}

    def resolve_autotrade_volume(self, suggested_volume: float) -> tuple[float, str]:
        state = self._read_state()
        config = self._autotrade_config_from_state(state)
        suggested = max(round(float(suggested_volume or 0.01), 2), 0.01)
        lot_mode = str(config.get("lot_mode", "range") or "range").lower()
        if lot_mode == "fixed":
            fixed = max(round(float(config.get("fixed_lot") or suggested), 2), 0.01)
            return fixed, "fixed_lot"
        min_lot = max(round(float(config.get("min_lot") or 0.01), 2), 0.01)
        max_lot = max(round(float(config.get("max_lot") or min_lot), 2), min_lot)
        return min(max(suggested, min_lot), max_lot), "range_clamp"

    def resolve_symbol(self, symbol: str) -> str:
        mt5 = self._import_mt5()
        state = self._read_state()
        overrides = dict(state.get("symbol_overrides", {}) or {})
        requested = str(symbol).upper()
        if requested in overrides and str(overrides[requested]).strip():
            return str(overrides[requested]).strip()
        candidates = [
            requested,
            f"{state.get('symbol_prefix', '')}{requested}{state.get('symbol_suffix', '')}",
            f"{requested}{state.get('symbol_suffix', '')}",
            f"{state.get('symbol_prefix', '')}{requested}",
        ]
        deduped = [item for index, item in enumerate(candidates) if item and item not in candidates[:index]]
        if mt5 is None:
            return deduped[0]
        try:
            available = mt5.symbols_get()
            names = {str(item.name): item for item in (available or [])}
            for candidate in deduped:
                if candidate in names:
                    return candidate
            for candidate in deduped:
                match = next((name for name in names if candidate in name), None)
                if match:
                    return match
        except Exception:
            pass
        return deduped[0]

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        mt5 = self._import_mt5()
        if mt5 is None:
            return []
        try:
            resolved = self.resolve_symbol(symbol) if symbol else None
            positions = mt5.positions_get(symbol=resolved) if resolved else mt5.positions_get()
            results: list[dict[str, Any]] = []
            for position in positions or []:
                results.append(
                    {
                        "ticket": int(getattr(position, "ticket", 0)),
                        "symbol": str(getattr(position, "symbol", "")),
                        "type": int(getattr(position, "type", 0)),
                        "volume": _safe_float(getattr(position, "volume", 0.0)),
                        "price_open": _safe_float(getattr(position, "price_open", 0.0)),
                        "sl": _safe_float(getattr(position, "sl", 0.0)),
                        "tp": _safe_float(getattr(position, "tp", 0.0)),
                        "profit": _safe_float(getattr(position, "profit", 0.0)),
                    }
                )
            return results
        except Exception:
            return []

    def quote(self, symbol: str) -> float | None:
        mt5, init_error = self._ensure_initialized()
        if mt5 is None or init_error:
            return None
        try:
            resolved_symbol = self.resolve_symbol(symbol)
            if not mt5.symbol_select(resolved_symbol, True):
                return None
            tick = mt5.symbol_info_tick(resolved_symbol)
            if tick is None:
                return None
            bid = _safe_float(getattr(tick, "bid", 0.0), 0.0)
            ask = _safe_float(getattr(tick, "ask", 0.0), 0.0)
            if bid > 0 and ask > 0:
                return round((bid + ask) / 2.0, 5)
            return bid or ask or None
        except Exception:
            return None

    def recent_candles(self, symbol: str, *, count: int = 240) -> list[dict[str, Any]]:
        mt5, init_error = self._ensure_initialized()
        if mt5 is None or init_error:
            return []
        try:
            resolved_symbol = self.resolve_symbol(symbol)
            if not mt5.symbol_select(resolved_symbol, True):
                return []
            rates = mt5.copy_rates_from_pos(resolved_symbol, getattr(mt5, "TIMEFRAME_M5", 5), 0, max(int(count), 60))
            results: list[dict[str, Any]] = []
            for item in rates or []:
                timestamp = datetime.fromtimestamp(int(item["time"]), tz=timezone.utc).isoformat()
                results.append(
                    {
                        "timestamp": timestamp,
                        "open": _safe_float(item["open"]),
                        "high": _safe_float(item["high"]),
                        "low": _safe_float(item["low"]),
                        "close": _safe_float(item["close"]),
                        "volume": _safe_float(item["tick_volume"]),
                    }
                )
            return results
        except Exception:
            return []

    def has_open_position(self, symbol: str) -> bool:
        resolved = self.resolve_symbol(symbol)
        return any(str(position.get("symbol", "")) == resolved for position in self.positions(symbol))

    def place_market_order(
        self,
        *,
        symbol: str,
        direction: str,
        volume: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        comment: str = "NexusTrader",
    ) -> dict[str, Any]:
        mt5, init_error = self._ensure_initialized()
        if mt5 is None:
            raise RuntimeError("MetaTrader5 Python package is not installed.")
        if init_error:
            raise RuntimeError(init_error)
        terminal = mt5.terminal_info()
        if not terminal or not bool(getattr(terminal, "connected", False)):
            raise RuntimeError("MT5 terminal is not connected to the trade server.")
        if not bool(getattr(terminal, "trade_allowed", False)):
            raise RuntimeError("MT5 terminal has AutoTrading disabled by client.")
        resolved_symbol = self.resolve_symbol(symbol)
        if not mt5.symbol_select(resolved_symbol, True):
            raise RuntimeError(f"MT5 could not select symbol {resolved_symbol}.")
        tick = mt5.symbol_info_tick(resolved_symbol)
        info = mt5.symbol_info(resolved_symbol)
        if tick is None or info is None:
            raise RuntimeError(f"MT5 quote unavailable for {resolved_symbol}.")
        side = str(direction).upper()
        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        price = _safe_float(tick.ask if side == "BUY" else tick.bid)
        filling = self._resolve_filling_mode(mt5, info)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": resolved_symbol,
            "volume": round(max(float(volume), 0.01), 2),
            "type": order_type,
            "price": price,
            "sl": _safe_float(stop_loss, 0.0) if stop_loss is not None else 0.0,
            "tp": _safe_float(take_profit, 0.0) if take_profit is not None else 0.0,
            "deviation": 20,
            "magic": 21042026,
            "comment": str(comment)[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError("MT5 order_send returned no result.")
        retcode = int(getattr(result, "retcode", 0))
        success_codes = {
            getattr(mt5, "TRADE_RETCODE_DONE", -1),
            getattr(mt5, "TRADE_RETCODE_PLACED", -1),
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", -1),
        }
        payload = {
            "retcode": retcode,
            "order": int(getattr(result, "order", 0)),
            "deal": int(getattr(result, "deal", 0)),
            "volume": round(max(float(volume), 0.01), 2),
            "symbol": resolved_symbol,
            "direction": side,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "comment": comment,
        }
        if retcode not in success_codes:
            error = f"MT5 order failed: retcode={retcode}"
            self._patch_state(last_error=error, last_action="order_failed")
            raise RuntimeError(error)
        self._patch_state(last_error="", last_action="order_sent", last_order=payload)
        return payload

    def can_trade(self) -> bool:
        status = self.status()
        return bool(status.get("installed")) and bool(status.get("connected"))

    def _resolve_filling_mode(self, mt5: Any, info: Any) -> int:
        supported = int(getattr(info, "filling_mode", 0) or 0)
        # MT5 symbol_info().filling_mode is a bitmask of allowed modes, but order_send
        # expects one concrete ORDER_FILLING_* value.
        candidates = [
            getattr(mt5, "ORDER_FILLING_FOK", 0),
            getattr(mt5, "ORDER_FILLING_IOC", 1),
            getattr(mt5, "ORDER_FILLING_RETURN", 2),
        ]
        for candidate in candidates:
            if supported & (1 << int(candidate)):
                return int(candidate)
        for candidate in candidates:
            if supported == int(candidate):
                return int(candidate)
        return int(getattr(mt5, "ORDER_FILLING_IOC", 1))


__all__ = ["MT5Bridge"]
