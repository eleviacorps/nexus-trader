"""Model status panel widget."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.text import Text
from textual.widget import Widget
from textual.widgets import Static


class StatusPanel(Widget):
    """Single-row operational status bar."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._static = Static("Waiting for first inference cycle...")

    def compose(self):
        yield self._static

    def update_status(
        self,
        *,
        signal: str,
        confidence: float,
        ev_threshold: float = 0.0,
        ev: float = 0.0,
        std: float = 0.0,
        positive_ratio: float = 0.0,
        negative_ratio: float = 0.0,
        regime: str,
        latency_ms: float,
        paths: int,
        auto_trade: bool,
        mt5_connected: bool,
        api_running: bool,
        integrity_ok: bool,
    ) -> None:
        conf_style = "grey62" if confidence < 0.05 else ("yellow" if confidence < 0.15 else "green")
        signal_style = "green" if signal == "BUY" else ("red" if signal == "SELL" else "white")
        text = Text()
        text.append("SIGNAL ", style="white")
        text.append(f"{signal}", style=signal_style)
        text.append(" | CONF ", style="white")
        text.append(f"{confidence:.2f}", style=conf_style)
        text.append(" | ", style="white")
        text.append(f"EV {ev:+.5f} | STD {std:.5f}", style="white")
        text.append(" | ", style="white")
        text.append(f"EV THR {ev_threshold:.5f}", style="cyan")
        text.append(" | ", style="white")
        text.append(f"REGIME {regime}", style="magenta")
        text.append(" | ", style="white")
        text.append(f"LATENCY {latency_ms:.1f}ms", style="white")
        text.append(" | ", style="white")
        text.append(f"PATHS {paths}", style="white")
        text.append(" | ", style="white")
        text.append(f"AUTO {'ON' if auto_trade else 'OFF'}", style="green" if auto_trade else "grey62")
        text.append(" | ", style="white")
        text.append(f"MT5 {'CONNECTED' if mt5_connected else 'DISCONNECTED'}", style="green" if mt5_connected else "red")
        text.append(" | ", style="white")
        text.append(f"API {'RUNNING' if api_running else 'STOPPED'}", style="green" if api_running else "red")
        text.append(" | ", style="white")
        text.append(f"DIR +{positive_ratio:.2f}/-{negative_ratio:.2f}", style="white")
        text.append(" | ", style="white")
        text.append(f"LAST UPDATE {datetime.now().strftime('%H:%M:%S')}", style="white")
        text.append(" | ", style="white")
        text.append(f"INTEGRITY {'OK' if integrity_ok else 'TAMPERED'}", style="green" if integrity_ok else "red")
        self._static.update(text)
