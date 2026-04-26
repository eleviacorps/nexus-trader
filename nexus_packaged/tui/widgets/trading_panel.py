"""Auto-trading control panel widget."""

from __future__ import annotations

from typing import Any

from textual.widget import Widget
from textual.widgets import Static


class TradingPanel(Widget):
    """Displays auto-trading mode and risk stats."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._static = Static("AUTO TRADE: [OFF] MODE: selective LOT: fixed(0.01)")

    def compose(self):
        yield self._static

    @staticmethod
    def _lot_display(cfg: Any) -> str:
        if cfg.lot_mode == "fixed":
            return f"fixed({cfg.fixed_lot_size:.2f})"
        if cfg.lot_mode == "range":
            return f"range({cfg.lot_min:.2f}-{cfg.lot_max:.2f},{cfg.lot_range_mode})"
        if cfg.lot_mode == "risk_pct":
            return f"risk_pct({cfg.risk_pct_per_trade:.2f}%)"
        if cfg.lot_mode == "kelly":
            return f"kelly({cfg.kelly_fraction:.2f})"
        return str(cfg.lot_mode)

    def update_panel(self, *, auto_trader, session_summary) -> None:
        cfg = auto_trader.config
        on = bool(cfg.enabled)
        text = (
            f"AUTO TRADE: [{'ON' if on else 'OFF'}]  "
            f"MODE: [{cfg.mode}]  "
            f"LOT: [{self._lot_display(cfg)}]  \n"
            f"Confidence: >={cfg.confidence_threshold:.2f}   "
            f"Interval: {cfg.interval_minutes}m   "
            f"Max Trades: {cfg.max_daily_trades}/day   "
            f"Max DD: {cfg.max_drawdown_pct:.1f}%\n"
            f"Open: {session_summary.open_trades}/{cfg.max_open_trades}   "
            f"Today: {session_summary.daily_trades} trades   "
            f"Daily PnL: ${session_summary.daily_pnl_usd:.2f}   "
            f"Session DD: {session_summary.session_drawdown_pct:.2f}%\n"
            "[CONFIGURE IN SETTINGS (s)] [CLOSE ALL: TODO] [PAUSE: TODO]"
        )
        self._static.update(text)
