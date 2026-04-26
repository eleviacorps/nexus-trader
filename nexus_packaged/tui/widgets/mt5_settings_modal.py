"""Settings modal for MT5 account and auto-trade config."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


class MT5SettingsModal(ModalScreen):
    """Modal for updating MT5 credentials and core auto-trade settings."""

    CSS = """
    MT5SettingsModal {
        align: center middle;
    }
    #settings-root {
        width: 90%;
        height: 90%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    .row {
        height: auto;
        margin: 0 0 1 0;
    }
    .row Label {
        width: 22;
    }
    .row Input {
        width: 1fr;
    }
    """

    def __init__(
        self,
        mt5_connector: Any,
        settings_path: str,
        auto_trader: Any | None = None,
        settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mt5_connector = mt5_connector
        self.settings_path = settings_path
        self.auto_trader = auto_trader
        self.settings = settings or {}
        self._status = Static("", id="settings-status")

    def compose(self):
        mt5_cfg = self.mt5_connector.get_runtime_config()
        auto_cfg = self.auto_trader.config if self.auto_trader is not None else None
        with VerticalScroll(id="settings-root"):
            yield Label("SETTINGS")
            yield Static(
                f"Connected: {'YES' if mt5_cfg['connected'] else 'NO'} | "
                f"Server: {mt5_cfg['server'] or '(unset)'} | Login: {mt5_cfg['login']}",
                id="mt5-connection-line",
            )

            yield Label("AUTO TRADE")
            with Horizontal(classes="row"):
                yield Label("Mode")
                yield Input(str(getattr(auto_cfg, "mode", "selective")), id="auto-mode")
            with Horizontal(classes="row"):
                yield Label("Lot Mode")
                yield Input(str(getattr(auto_cfg, "lot_mode", "fixed")), id="auto-lot-mode")
            with Horizontal(classes="row"):
                yield Label("Fixed Lot Size")
                yield Input(f"{float(getattr(auto_cfg, 'fixed_lot_size', 0.01)):.2f}", id="auto-fixed-lot")
            with Horizontal(classes="row"):
                yield Label("Lot Min")
                yield Input(f"{float(getattr(auto_cfg, 'lot_min', 0.01)):.2f}", id="auto-lot-min")
            with Horizontal(classes="row"):
                yield Label("Lot Max")
                yield Input(f"{float(getattr(auto_cfg, 'lot_max', 0.10)):.2f}", id="auto-lot-max")
            with Horizontal(classes="row"):
                yield Label("Confidence Threshold")
                yield Input(f"{float(getattr(auto_cfg, 'confidence_threshold', 0.65)):.2f}", id="auto-confidence")
            with Horizontal(classes="row"):
                yield Label("Interval Minutes")
                yield Input(str(int(getattr(auto_cfg, "interval_minutes", 15))), id="auto-interval")
            with Horizontal(classes="row"):
                yield Label("Max Daily Trades")
                yield Input(str(int(getattr(auto_cfg, "max_daily_trades", 20))), id="auto-max-daily")
            with Horizontal(classes="row"):
                yield Label("Max Drawdown %")
                yield Input(f"{float(getattr(auto_cfg, 'max_drawdown_pct', 10.0)):.2f}", id="auto-max-dd")
            with Horizontal(classes="row"):
                yield Label("Paper Mode")
                yield Input("true" if bool(getattr(auto_cfg, "paper_mode", True)) else "false", id="auto-paper")
            with Horizontal(classes="row"):
                dirs = list(getattr(auto_cfg, "allowed_directions", ["BUY", "SELL"]))
                yield Label("Allowed Directions")
                yield Input(",".join(str(x).upper() for x in dirs), id="auto-directions")

            yield Label("MT5 ACCOUNT")
            with Horizontal(classes="row"):
                yield Label("Login")
                yield Input(str(mt5_cfg["login"]) if int(mt5_cfg["login"]) > 0 else "", id="mt5-login")
            with Horizontal(classes="row"):
                yield Label("Password")
                yield Input("", password=True, placeholder=mt5_cfg["password_masked"] or "(unchanged)", id="mt5-password")
            with Horizontal(classes="row"):
                yield Label("Server")
                yield Input(str(mt5_cfg["server"]), id="mt5-server")
            with Horizontal(classes="row"):
                yield Label("Execution Enabled")
                yield Input("true" if bool(mt5_cfg["execution_enabled"]) else "false", id="mt5-execution")
            with Horizontal(classes="row"):
                yield Label("Reconnect Attempts")
                yield Input(str(mt5_cfg["reconnect_attempts"]), id="mt5-attempts")
            with Horizontal(classes="row"):
                yield Label("Reconnect Delay Seconds")
                yield Input(str(mt5_cfg["reconnect_delay_seconds"]), id="mt5-delay")

            yield self._status
            with Horizontal(classes="row"):
                yield Button("Apply Runtime + Reconnect", id="apply-runtime", variant="primary")
                yield Button("Apply + Persist + Reconnect", id="apply-persist", variant="success")
            with Horizontal(classes="row"):
                yield Button("Disable Auto", id="disable-auto", variant="warning")
                yield Button("Disconnect MT5", id="disconnect", variant="warning")
                yield Button("Cancel", id="cancel", variant="default")

    @staticmethod
    def _to_bool(text: str) -> bool:
        return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _build_mt5_payload(self) -> dict[str, Any]:
        login_text = self.query_one("#mt5-login", Input).value.strip()
        password = self.query_one("#mt5-password", Input).value
        server = self.query_one("#mt5-server", Input).value.strip()
        attempts_text = self.query_one("#mt5-attempts", Input).value.strip() or "3"
        delay_text = self.query_one("#mt5-delay", Input).value.strip() or "5"
        execution_text = self.query_one("#mt5-execution", Input).value.strip() or "false"

        payload: dict[str, Any] = {
            "execution_enabled": self._to_bool(execution_text),
            "reconnect_attempts": int(attempts_text),
            "reconnect_delay_seconds": int(delay_text),
        }
        if login_text:
            payload["login"] = int(login_text)
        if password:
            payload["password"] = password
        if server:
            payload["server"] = server
        return payload

    def _build_auto_payload(self) -> dict[str, Any]:
        mode = self.query_one("#auto-mode", Input).value.strip() or "selective"
        lot_mode = self.query_one("#auto-lot-mode", Input).value.strip() or "fixed"
        fixed_lot = float(self.query_one("#auto-fixed-lot", Input).value.strip() or "0.01")
        lot_min = float(self.query_one("#auto-lot-min", Input).value.strip() or "0.01")
        lot_max = float(self.query_one("#auto-lot-max", Input).value.strip() or "0.10")
        confidence = float(self.query_one("#auto-confidence", Input).value.strip() or "0.65")
        interval = int(self.query_one("#auto-interval", Input).value.strip() or "15")
        max_daily = int(self.query_one("#auto-max-daily", Input).value.strip() or "20")
        max_dd = float(self.query_one("#auto-max-dd", Input).value.strip() or "10.0")
        paper_mode = self._to_bool(self.query_one("#auto-paper", Input).value.strip() or "true")
        raw_dirs = self.query_one("#auto-directions", Input).value.strip()
        allowed_directions = [d.strip().upper() for d in raw_dirs.split(",") if d.strip()]
        if not allowed_directions:
            allowed_directions = ["BUY", "SELL"]
        return {
            "mode": mode,
            "lot_mode": lot_mode,
            "fixed_lot_size": fixed_lot,
            "lot_min": lot_min,
            "lot_max": lot_max,
            "confidence_threshold": confidence,
            "interval_minutes": interval,
            "max_daily_trades": max_daily,
            "max_drawdown_pct": max_dd,
            "paper_mode": paper_mode,
            "allowed_directions": allowed_directions,
        }

    def _persist_payloads(self, mt5_payload: dict[str, Any], auto_payload: dict[str, Any]) -> None:
        path = Path(self.settings_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("mt5", {}).update(mt5_payload)
        data.setdefault("auto_trade", {}).update(auto_payload)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _apply_auto_payload(self, auto_payload: dict[str, Any]) -> None:
        if self.auto_trader is None:
            return
        current_cfg = self.auto_trader.config
        cfg_dict = asdict(current_cfg)
        cfg_dict.update(auto_payload)
        new_cfg = type(current_cfg)(**cfg_dict)
        # Do not implicitly enable auto trading from settings apply.
        new_cfg.enabled = bool(current_cfg.enabled)
        self.auto_trader.update_config(new_cfg)

    async def _apply(self, *, persist: bool) -> None:
        try:
            mt5_payload = self._build_mt5_payload()
            auto_payload = self._build_auto_payload()
            self._apply_auto_payload(auto_payload)
            self.mt5_connector.update_runtime_config(mt5_payload)
            if persist:
                await asyncio.to_thread(self._persist_payloads, mt5_payload, auto_payload)
            connected = await self.mt5_connector.reconnect()
            self._status.update(
                f"Saved. MT5 connected={connected}. "
                f"Auto mode={auto_payload['mode']} lot={auto_payload['lot_mode']}"
            )
            self.app.notify("Settings updated.")
        except Exception as exc:  # noqa: BLE001
            self._status.update(f"Error: {exc}")

    async def _disconnect(self) -> None:
        try:
            await self.mt5_connector.disconnect()
            self._status.update("MT5 disconnected.")
            self.app.notify("MT5 disconnected.")
        except Exception as exc:  # noqa: BLE001
            self._status.update(f"Disconnect error: {exc}")

    def _disable_auto(self) -> None:
        if self.auto_trader is None:
            return
        cfg = self.auto_trader.config
        if cfg.enabled:
            self.auto_trader.toggle()
        self._status.update("Auto trading disabled.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id == "apply-runtime":
            asyncio.create_task(self._apply(persist=False))
            return
        if event.button.id == "apply-persist":
            asyncio.create_task(self._apply(persist=True))
            return
        if event.button.id == "disable-auto":
            self._disable_auto()
            return
        if event.button.id == "disconnect":
            asyncio.create_task(self._disconnect())
