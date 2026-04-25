"""Textual application for Nexus Trader packaged runtime."""

from __future__ import annotations

import asyncio
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, ProgressBar, Static

from nexus_packaged.tui.widgets.chart_view import ChartViewWidget
from nexus_packaged.tui.widgets.diffusion_panel import DiffusionPanel
from nexus_packaged.tui.widgets.manual_trade_modal import ManualTradeModal
from nexus_packaged.tui.widgets.mt5_settings_modal import MT5SettingsModal
from nexus_packaged.tui.widgets.news_panel import NewsPanel
from nexus_packaged.tui.widgets.status_panel import StatusPanel
from nexus_packaged.tui.widgets.trading_panel import TradingPanel


@dataclass
class RuntimeContext:
    """Runtime object wiring used by the TUI."""

    settings: dict[str, Any]
    inference_runner: Any
    news_aggregator: Any
    auto_trader: Any
    manual_trader: Any
    trade_manager: Any
    mt5_connector: Any
    api_running_flag: callable
    integrity_ok_flag: callable
    ohlcv: pd.DataFrame
    settings_path: str


class MessageModal(ModalScreen):
    """Simple message dialog."""

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.title = title
        self.body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(self.title)
            yield Static(self.body)
            yield Static("Press any key to close.")

    def on_key(self, _event) -> None:
        self.dismiss(True)


class NexusTraderApp(App):
    """Main TUI application."""

    CSS_PATH = "styles/theme.tcss"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_all", "Refresh"),
        Binding("b", "open_backtest_modal", "Backtest"),
        Binding("s", "open_settings_modal", "Settings"),
        Binding("w", "open_web_chart", "Web Chart"),
        Binding("t", "toggle_auto_trade", "Toggle Auto"),
        Binding("m", "open_manual_trade_modal", "Manual Trade"),
        Binding("h", "toggle_help", "Help"),
        Binding("l", "open_trade_log_viewer", "Trade Log"),
    ]

    def __init__(self, runtime: RuntimeContext, *, no_webview: bool = False):
        super().__init__()
        self.runtime = runtime
        self.no_webview = no_webview
        self._help_visible = False
        self._last_update_ts = None
        self._poll_task: asyncio.Task | None = None

        self.splash = Static(f"NEXUS TRADER {runtime.settings.get('version', 'v27.1')}", id="splash")
        self.progress = ProgressBar(total=100, show_eta=False, id="init-progress")
        self.help_overlay = Static(
            "Keys: q quit | r refresh | b backtest | s settings | w web chart | t toggle auto | m manual | h help | l log",
            id="help-overlay",
        )
        self.chart = ChartViewWidget(no_webview=no_webview, id="chart-view")
        self.diffusion = DiffusionPanel(id="diffusion-panel")
        self.news = NewsPanel(
            news_aggregator=runtime.news_aggregator,
            refresh_seconds=int(runtime.settings.get("news", {}).get("cache_ttl_seconds", 300)),
            id="news-panel",
        )
        self.status = StatusPanel(id="status-panel")
        self.trading = TradingPanel(id="trading-panel")
        self.manual_stub = Static("Manual Trade: press `m` to open modal", id="manual-panel")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main-layout"):
            yield self.splash
            yield self.progress
            yield self.help_overlay
            yield self.chart
            yield self.diffusion
            yield self.news
            yield self.status
            yield self.trading
            yield self.manual_stub
        yield Footer()

    async def on_mount(self) -> None:
        self.help_overlay.display = False
        self.chart.set_initial_bars(self.runtime.ohlcv.tail(int(self.runtime.settings.get("ui", {}).get("chart_initial_bars", 500))))
        splash_seconds = float(self.runtime.settings.get("ui", {}).get("splash_duration_seconds", 1.5))
        await self._initialize_components(splash_seconds)
        self._poll_task = asyncio.create_task(self._poll_loop(), name="nexus_tui_poll")

    async def on_unmount(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def _initialize_components(self, splash_seconds: float) -> None:
        # Startup sequence: splash -> async init -> degrade on failure.
        steps = [
            ("Model", 20),
            ("MT5", 40),
            ("API", 60),
            ("Inference", 80),
            ("UI Ready", 100),
        ]
        for label, pct in steps:
            self.splash.update(f"NEXUS TRADER {self.runtime.settings.get('version', 'v27.1')} | {label}")
            self.progress.update(progress=pct)
            await asyncio.sleep(max(0.15, splash_seconds / len(steps)))
        self.splash.display = False
        self.progress.display = False

    async def _poll_loop(self) -> None:
        fps = int(self.runtime.settings.get("ui", {}).get("max_fps", 10))
        sleep_seconds = 1.0 / max(1, fps)
        bar_step_seconds = int(self.runtime.settings.get("data", {}).get("base_timeframe_minutes", 1)) * 60
        bar_step_seconds = max(60, bar_step_seconds)
        while True:
            event = self.runtime.inference_runner.latest_event
            if event is not None and event.timestamp != self._last_update_ts:
                self._last_update_ts = event.timestamp
                self.diffusion.update_paths(event.paths)
                bar_ts = pd.Timestamp(event.bar_timestamp)
                if bar_ts in self.runtime.ohlcv.index:
                    row = self.runtime.ohlcv.loc[bar_ts]
                    self.chart.append_bar(
                        {
                            "time": int(bar_ts.timestamp()),
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                        }
                    )
                chart_paths = []
                start_ts = int(pd.Timestamp(event.bar_timestamp).timestamp())
                for path in event.paths[:64]:
                    series = [
                        {"time": int(start_ts + i * bar_step_seconds), "value": float(v)}
                        for i, v in enumerate(path.tolist())
                    ]
                    chart_paths.append(series)
                self.chart.set_diffusion_paths(chart_paths)
                self.status.update_status(
                    signal=event.signal,
                    confidence=event.confidence,
                    ev_threshold=float(event.meta.get("ev_threshold", 0.0)),
                    ev=float(event.meta.get("ev", 0.0)),
                    std=float(event.meta.get("std", 0.0)),
                    positive_ratio=float(event.meta.get("positive_ratio", 0.0)),
                    negative_ratio=float(event.meta.get("negative_ratio", 0.0)),
                    regime=event.regime,
                    latency_ms=event.latency_ms,
                    paths=int(event.paths.shape[0]),
                    auto_trade=bool(self.runtime.auto_trader.config.enabled),
                    mt5_connected=bool(self.runtime.mt5_connector.is_connected),
                    api_running=bool(self.runtime.api_running_flag()),
                    integrity_ok=bool(self.runtime.integrity_ok_flag()),
                )
            summary = self.runtime.trade_manager.get_session_summary()
            self.trading.update_panel(auto_trader=self.runtime.auto_trader, session_summary=summary)
            await asyncio.sleep(sleep_seconds)

    def action_refresh_all(self) -> None:
        event = self.runtime.inference_runner.latest_event
        if event is not None:
            self.diffusion.update_paths(event.paths)
        summary = self.runtime.trade_manager.get_session_summary()
        self.trading.update_panel(auto_trader=self.runtime.auto_trader, session_summary=summary)

    def action_toggle_help(self) -> None:
        self._help_visible = not self._help_visible
        self.help_overlay.display = self._help_visible

    def action_toggle_auto_trade(self) -> None:
        enabled = self.runtime.auto_trader.toggle()
        self.notify(f"Auto trading {'enabled' if enabled else 'disabled'}")

    def action_open_manual_trade_modal(self) -> None:
        self.push_screen(
            ManualTradeModal(
                manual_trader=self.runtime.manual_trader,
                inference_runner=self.runtime.inference_runner,
                settings=self.runtime.settings,
            )
        )

    def action_open_backtest_modal(self) -> None:
        self.push_screen(
            MessageModal(
                "Backtest",
                "Use API endpoint POST /backtest/run for full async backtests.\n"
                "This build keeps backtests non-blocking via ProcessPool.",
            )
        )

    def action_open_settings_modal(self) -> None:
        self.push_screen(
            MT5SettingsModal(
                mt5_connector=self.runtime.mt5_connector,
                settings_path=self.runtime.settings_path,
                auto_trader=self.runtime.auto_trader,
                settings=self.runtime.settings,
            )
        )

    def action_open_trade_log_viewer(self) -> None:
        log_path = Path("nexus_packaged/logs/trades.log")
        if not log_path.exists():
            self.push_screen(MessageModal("Trade Log", "No trade log yet."))
            return
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]
        body = "\n".join(lines) if lines else "No trade log entries."
        self.push_screen(MessageModal("Trade Log", body))

    def action_open_web_chart(self) -> None:
        api_cfg = self.runtime.settings.get("api", {})
        host = str(api_cfg.get("host", "127.0.0.1"))
        port = int(api_cfg.get("port", 8765))
        url = f"http://{host}:{port}/execution"
        webbrowser.open(url)
        self.notify(f"Opened execution chart: {url}")
