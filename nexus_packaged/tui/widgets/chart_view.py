"""Chart widget with pywebview primary path and ASCII fallback."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import pandas as pd
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Nexus Chart</title>
  <style>html, body, #chart { width: 100%; height: 100%; margin: 0; background: #0a0a0f; }</style>
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body>
  <div id="chart"></div>
  <script>
    const chart = LightweightCharts.createChart(document.getElementById('chart'), {
      layout: { background: { color: '#0a0a0f' }, textColor: '#00e5ff' },
      grid: { vertLines: { color: '#20232f' }, horzLines: { color: '#20232f' } },
      rightPriceScale: { borderColor: '#00e5ff' },
      timeScale: { borderColor: '#00e5ff' }
    });
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00c853',
      downColor: '#f44336',
      borderVisible: false,
      wickUpColor: '#00c853',
      wickDownColor: '#f44336'
    });
    const pathSeries = [];
    const markerSeries = candleSeries;
    window.setBars = function(bars) {
      candleSeries.setData(bars);
      chart.timeScale().fitContent();
    }
    window.appendBar = function(bar) {
      candleSeries.update(bar);
    }
    window.setPaths = function(paths) {
      while (pathSeries.length) {
        const series = pathSeries.pop();
        chart.removeSeries(series);
      }
      for (const path of paths) {
        const s = chart.addLineSeries({ color: 'rgba(0,229,255,0.25)', lineWidth: 1 });
        s.setData(path);
        pathSeries.push(s);
      }
    }
    window.setTradeMarkers = function(markers) {
      markerSeries.setMarkers(markers);
    }
  </script>
</body>
</html>
"""


class ChartViewWidget(Widget):
    """Chart widget; uses external webview window when available."""

    bars_loaded = reactive(0)

    def __init__(self, *, no_webview: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.logger = logging.getLogger("nexus.system")
        self.no_webview = no_webview
        self._webview_ready = False
        self._webview_window = None
        self._fallback = Static("Chart initializing...", id="chart-fallback")
        self._latest_bars: list[dict[str, Any]] = []
        self._latest_paths: list[list[dict[str, Any]]] = []
        self._latest_markers: list[dict[str, Any]] = []

    def compose(self):
        yield self._fallback

    def on_mount(self) -> None:
        if self.no_webview:
            self.logger.info("WebView disabled by flag; using ASCII fallback.")
            self._render_fallback_plot()
            return
        try:
            import webview  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("pywebview unavailable; fallback enabled: %s", exc)
            self._render_fallback_plot()
            return

        def _start() -> None:
            self._webview_window = webview.create_window("Nexus Chart", html=HTML_TEMPLATE, width=1280, height=640)
            self._webview_ready = True
            webview.start(gui="qt" if webview.gui is None else None, debug=False)

        thread = threading.Thread(target=_start, daemon=True, name="nexus_chart_webview")
        thread.start()
        self._fallback.update("WebView chart running in separate window.")

    def _render_fallback_plot(self) -> None:
        """Render a compact ASCII chart: recent closes + predicted paths."""
        try:
            import numpy as np
            import plotext as plt  # type: ignore

            if not self._latest_bars:
                self._fallback.update("Chart fallback: waiting for bars...")
                return

            closes = [float(bar.get("close", 0.0)) for bar in self._latest_bars]
            history = closes[-200:]
            x_hist = list(range(len(history)))

            plt.clear_figure()
            plt.theme("clear")
            plt.plot(x_hist, history, color="cyan")

            overlay = []
            for series in self._latest_paths[:64]:
                values = [float(p.get("value", 0.0)) for p in series]
                if values:
                    overlay.append(values)

            if overlay:
                horizon = len(overlay[0])
                x_future = list(range(max(0, len(history) - 1), max(0, len(history) - 1) + horizon))
                step = max(1, len(overlay) // 12)
                for values in overlay[::step][:12]:
                    plt.plot(x_future, values, color="gray")
                arr = np.asarray(overlay, dtype=np.float32)
                median = np.median(arr, axis=0).tolist()
                p10 = np.percentile(arr, 10, axis=0).tolist()
                p90 = np.percentile(arr, 90, axis=0).tolist()
                plt.plot(x_future, median, color="white")
                plt.plot(x_future, p10, color="yellow")
                plt.plot(x_future, p90, color="yellow")

            plt.title(f"Bars: {len(self._latest_bars)} | Paths: {len(overlay)}")
            self._fallback.update(plt.build())
        except Exception as exc:  # noqa: BLE001
            self._fallback.update(f"Bars loaded: {self.bars_loaded} | Chart fallback error: {exc}")

    def _to_candles(self, ohlcv: pd.DataFrame) -> list[dict[str, Any]]:
        bars = []
        for idx, row in ohlcv.tail(500).iterrows():
            bars.append(
                {
                    "time": int(pd.Timestamp(idx).timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
        return bars

    def set_initial_bars(self, ohlcv: pd.DataFrame) -> None:
        bars = self._to_candles(ohlcv)
        self._latest_bars = bars
        self.bars_loaded = len(bars)
        self._fallback.update(f"Bars loaded: {len(bars)}")
        self._render_fallback_plot()
        if self._webview_ready and self._webview_window is not None:
            try:
                self._webview_window.evaluate_js(f"window.setBars({json.dumps(bars)});")
            except Exception:  # noqa: BLE001
                pass

    def append_bar(self, bar: dict[str, Any]) -> None:
        self._latest_bars.append(bar)
        if len(self._latest_bars) > 500:
            self._latest_bars = self._latest_bars[-500:]
        if not self._webview_ready:
            self._render_fallback_plot()
        if self._webview_ready and self._webview_window is not None:
            try:
                self._webview_window.evaluate_js(f"window.appendBar({json.dumps(bar)});")
            except Exception:  # noqa: BLE001
                pass

    def set_diffusion_paths(self, paths: list[list[dict[str, Any]]]) -> None:
        self._latest_paths = paths
        if self._webview_ready and self._webview_window is not None:
            try:
                self._webview_window.evaluate_js(f"window.setPaths({json.dumps(paths)});")
            except Exception:  # noqa: BLE001
                pass
        else:
            self._render_fallback_plot()

    def set_trade_markers(self, markers: list[dict[str, Any]]) -> None:
        self._latest_markers = markers
        if self._webview_ready and self._webview_window is not None:
            try:
                self._webview_window.evaluate_js(f"window.setTradeMarkers({json.dumps(markers)});")
            except Exception:  # noqa: BLE001
                pass
