"""Widget exports for Nexus TUI."""

from __future__ import annotations

from nexus_packaged.tui.widgets.chart_view import ChartViewWidget
from nexus_packaged.tui.widgets.diffusion_panel import DiffusionPanel
from nexus_packaged.tui.widgets.manual_trade_modal import ManualTradeModal
from nexus_packaged.tui.widgets.mt5_settings_modal import MT5SettingsModal
from nexus_packaged.tui.widgets.news_panel import NewsPanel
from nexus_packaged.tui.widgets.status_panel import StatusPanel
from nexus_packaged.tui.widgets.trading_panel import TradingPanel

__all__ = [
    "ChartViewWidget",
    "DiffusionPanel",
    "NewsPanel",
    "StatusPanel",
    "TradingPanel",
    "ManualTradeModal",
    "MT5SettingsModal",
]
