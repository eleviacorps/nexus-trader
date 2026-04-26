"""Auto-trading coordinator."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nexus_packaged.trading.lot_calculator import LotCalculator
from nexus_packaged.trading.trade_manager import TradeManager


@dataclass
class AutoTradeConfig:
    """Auto-trade configuration."""

    enabled: bool = False
    mode: str = "selective"
    confidence_threshold: float = 0.65
    frequency_every_n_cycles: int = 1
    interval_minutes: int = 15
    max_trade_count: int = 10
    lot_mode: str = "fixed"
    fixed_lot_size: float = 0.01
    lot_min: float = 0.01
    lot_max: float = 0.10
    lot_range_mode: str = "confidence"
    risk_pct_per_trade: float = 1.0
    kelly_fraction: float = 0.25
    max_open_trades: int = 3
    max_daily_trades: int = 20
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    risk_reward: float = 2.0
    sl_atr_multiplier: float = 1.5
    trade_expiry_bars: int = 10
    allowed_directions: list = field(default_factory=lambda: ["BUY", "SELL"])
    paper_mode: bool = True


class AutoTrader:
    """Event-driven auto-trading loop."""

    def __init__(self, config: AutoTradeConfig, mt5, inference, trade_manager: TradeManager | None = None, settings: dict | None = None):
        self._config = config
        self._config.enabled = False  # Safety override at startup.
        self.mt5 = mt5
        self.inference = inference
        self.settings = settings or {}
        self.trade_manager = trade_manager or TradeManager(
            initial_equity=float(self.settings.get("backtest", {}).get("default_initial_equity", 10000.0)),
            pip_value_per_lot=float(self.settings.get("backtest", {}).get("pip_value_per_lot", 1.0)),
            leverage=int(self.settings.get("backtest", {}).get("default_leverage", 200)),
            contract_size=float(self.settings.get("backtest", {}).get("contract_size", 100.0)),
        )
        self.trade_manager.set_price_provider(self.inference.current_price)
        self.logger = __import__("logging").getLogger("nexus.trades")
        self._task: asyncio.Task | None = None
        self._running = False
        self._paused = False
        self._cycle_count = 0
        self._trade_count = 0
        self._last_trade_time: datetime | None = None
        self._last_snapshot_id: str | None = None
        self._lot_calculator = LotCalculator()
        self._live_exec_failures = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def config(self) -> AutoTradeConfig:
        return self._config

    def update_config(self, new_config: AutoTradeConfig) -> None:
        self._config = new_config
        # Safety: never auto-enable on config update.
        if not self._running:
            self._config.enabled = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="nexus_auto_trader")

    async def stop(self) -> None:
        self._running = False
        self._config.enabled = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def toggle(self) -> bool:
        self._config.enabled = not bool(self._config.enabled)
        self.logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "AUTO_TOGGLE",
                    "enabled": self._config.enabled,
                }
            )
        )
        return self._config.enabled

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def reset_state(self) -> None:
        """Reset transient runtime counters/locks."""
        self._last_snapshot_id = None
        self._cycle_count = 0
        self._trade_count = 0
        self._last_trade_time = None

    def _mode_allows(self, event) -> bool:
        cfg = self._config
        mode = cfg.mode.lower()
        if mode == "selective":
            return float(event.confidence) >= float(cfg.confidence_threshold)
        if mode == "frequency":
            return self._cycle_count % max(1, int(cfg.frequency_every_n_cycles)) == 0
        if mode == "interval":
            if self._last_trade_time is None:
                return True
            delta = datetime.now(timezone.utc) - self._last_trade_time
            return delta.total_seconds() >= float(cfg.interval_minutes) * 60.0
        if mode == "count":
            return self._trade_count < int(cfg.max_trade_count)
        return True

    async def _maybe_place_trade(self, event) -> None:
        live_signal = event.meta.get("live_signal") if isinstance(event.meta.get("live_signal"), dict) else None
        snapshot_signal = event.meta.get("snapshot_signal") if isinstance(event.meta.get("snapshot_signal"), dict) else None
        current_signal = snapshot_signal if snapshot_signal else (live_signal or {})
        decision = str(current_signal.get("decision", event.signal)).upper()
        confidence = float(current_signal.get("confidence", event.confidence))
        ev = float(current_signal.get("ev", event.meta.get("ev", 0.0)))
        ev_threshold = float(current_signal.get("ev_threshold", event.meta.get("ev_threshold", 0.0)))
        conf_min = float(self.settings.get("execution", {}).get("conf_min", 0.10))

        allowed_by_signal = True
        if snapshot_signal and bool(snapshot_signal.get("snapshot_active", False)):
            allowed_by_signal = False
        if abs(ev) < ev_threshold:
            allowed_by_signal = False
        if confidence < conf_min:
            allowed_by_signal = False

        self.logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "AUTO_CHECK",
                    "decision": decision,
                    "confidence": confidence,
                    "ev": ev,
                    "ev_threshold": ev_threshold,
                    "allowed": bool(allowed_by_signal),
                    "signal_source": "snapshot" if snapshot_signal else "live",
                }
            )
        )

        if not allowed_by_signal:
            return
        if decision not in {"BUY", "SELL"}:
            return
        if decision not in set(str(x).upper() for x in self._config.allowed_directions):
            return
        if not self._mode_allows(event):
            return
        allowed, reason = self.trade_manager.check_risk_limits(self._config)
        if not allowed:
            self.logger.info(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event": "SKIP",
                        "source": "auto",
                        "direction": decision,
                        "lot": 0.0,
                        "entry": self.inference.current_price(),
                        "sl": 0.0,
                        "tp": 0.0,
                        "confidence": confidence,
                        "margin_used": 0.0,
                        "reason": "RISK_LIMIT:" + reason,
                    }
                )
            )
            return

        price = float(self.inference.current_price())
        rr = float(current_signal.get("rr", event.meta.get("rr", self._config.risk_reward)))
        if rr <= 0:
            rr = float(self._config.risk_reward)
        # Prefer snapshot execution distances when present.
        sl_distance = float(current_signal.get("sl_distance", event.meta.get("sl_distance", 0.0)) or 0.0)
        if sl_distance <= 0:
            path_dispersion = float(abs(float(event.band_90[0]) - float(event.band_10[0])))
            sl_distance = max(price * 0.001, path_dispersion * max(1.0, float(self._config.sl_atr_multiplier)))
        sl_pips = sl_distance / 0.01
        summary = self.trade_manager.get_session_summary()
        win_rate_hist = 0.5
        history = self.trade_manager.get_trade_history(limit=200)
        if history:
            win_rate_hist = len([x for x in history if x.pnl_usd > 0]) / len(history)
        lot = self._lot_calculator.calculate(
            mode=self._config.lot_mode,
            config=self._config,
            account_equity=float(summary.equity),
            sl_pips=float(sl_pips),
            pip_value=float(self.settings.get("broker", {}).get("pip_value_per_lot", 1.0)),
            confidence=float(confidence),
            win_rate_history=float(win_rate_hist),
            broker_lot_min=float(self.settings.get("broker", {}).get("lot_min", 0.01)),
            broker_lot_max=float(self.settings.get("broker", {}).get("lot_max", 500.0)),
            broker_lot_step=float(self.settings.get("broker", {}).get("lot_step", 0.01)),
            risk_reward=float(self._config.risk_reward),
        )
        if decision == "BUY":
            sl = price - sl_distance
            tp = price + sl_distance * rr
        else:
            sl = price + sl_distance
            tp = price - sl_distance * rr

        execution_enabled = bool(self.settings.get("mt5", {}).get("execution_enabled", False))
        paper_mode = bool(self._config.paper_mode or (not execution_enabled))
        if not paper_mode and hasattr(self.mt5, "place_order"):
            request_payload = {
                "direction": decision,
                "lot_size": lot,
                "entry_type": "market",
                "entry_price": None,
                "sl": sl,
                "tp": tp,
                "comment": f"auto:{self._config.mode}",
            }
            if not bool(getattr(self.mt5, "is_connected", False)):
                self._config.enabled = False
                self._config.paper_mode = True
                self.logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "event": "SKIP",
                            "source": "auto",
                            "direction": decision,
                            "lot": lot,
                            "entry": price,
                            "sl": sl,
                            "tp": tp,
                            "confidence": confidence,
                            "margin_used": 0.0,
                            "reason": "LIVE_EXEC_DISABLED:MT5_DISCONNECTED",
                        }
                    )
                )
                return
            try:
                await self.mt5.place_order(request_payload)
                self._live_exec_failures = 0
            except Exception as exc:  # noqa: BLE001
                self._live_exec_failures += 1
                self._config.enabled = False
                self._config.paper_mode = True
                self.logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "event": "SKIP",
                            "source": "auto",
                            "direction": decision,
                            "lot": lot,
                            "entry": price,
                            "sl": sl,
                            "tp": tp,
                            "confidence": confidence,
                            "margin_used": 0.0,
                            "reason": f"LIVE_EXEC_DISABLED:{type(exc).__name__}",
                        }
                    )
                )
                __import__("logging").getLogger("nexus.errors").warning(
                    "Auto live execution failed; switched to paper mode: %s",
                    exc,
                )
                return

        snapshot_payload = None
        if hasattr(self.inference, "create_snapshot_from_event"):
            snapshot_payload = self.inference.create_snapshot_from_event(event)
        snapshot_id = str((snapshot_payload or {}).get("snapshot_id", "")).strip()
        if snapshot_id and snapshot_id == self._last_snapshot_id:
            return

        trade = self.trade_manager.open_trade(
            direction=decision,
            lot=lot,
            sl=sl,
            tp=tp,
            comment=f"auto:{self._config.mode}",
        )
        trade.source = "auto"
        trade.confidence = float(confidence)
        self._trade_count += 1
        self._last_trade_time = datetime.now(timezone.utc)
        if snapshot_id:
            self._last_snapshot_id = snapshot_id

    async def _run(self) -> None:
        while self._running:
            try:
                event = await self.inference.event_queue.get()
                self._cycle_count += 1
                if not self._config.enabled or self._paused:
                    continue
                await self._maybe_place_trade(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                __import__("logging").getLogger("nexus.errors").exception("Auto trader error: %s", exc)
