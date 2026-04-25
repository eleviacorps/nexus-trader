"""Hybrid chart/API bridge endpoints for V27."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import HTTPException, Query
from fastapi.responses import HTMLResponse

from nexus_packaged.v27_hybrid.path_mapper import paths_to_time_value, summarize_path_distribution


def _hybrid_dashboard_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Nexus Hybrid Chart</title>
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    body { margin:0; background:#0a0a0f; color:#d6f7ff; font-family: monospace; }
    #top { padding:8px 12px; border-bottom:1px solid #1f2d3a; }
    #chart { width:100vw; height:72vh; }
    #meta { padding:8px 12px; border-top:1px solid #1f2d3a; white-space:pre; }
  </style>
</head>
<body>
  <div id="top">Nexus V27 Hybrid Chart</div>
  <div id="chart"></div>
  <div id="meta">Loading...</div>
  <script>
    const chart = LightweightCharts.createChart(document.getElementById('chart'), {
      layout: { background: { color: '#0a0a0f' }, textColor: '#d6f7ff' },
      grid: { vertLines: { color: '#1a2230' }, horzLines: { color: '#1a2230' } },
      timeScale: { borderColor: '#00e5ff' },
      rightPriceScale: { borderColor: '#00e5ff' }
    });
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00c853', downColor: '#f44336', borderVisible: false,
      wickUpColor: '#00c853', wickDownColor: '#f44336'
    });
    const meanSeries = chart.addLineSeries({ color: '#ffffff', lineWidth: 2 });
    const p10Series = chart.addLineSeries({ color: 'rgba(255,235,59,0.7)', lineWidth: 1 });
    const p90Series = chart.addLineSeries({ color: 'rgba(255,235,59,0.7)', lineWidth: 1 });
    let pathSeries = [];
    let lastBarTs = 0;

    function clearPathSeries() {
      for (const s of pathSeries) chart.removeSeries(s);
      pathSeries = [];
    }

    async function refreshOHLC() {
      const r = await fetch('/ohlc?limit=500', { cache: 'no-store' });
      if (!r.ok) return;
      const bars = await r.json();
      candleSeries.setData(bars);
      if (bars.length) lastBarTs = bars[bars.length - 1].time;
      chart.timeScale().fitContent();
    }

    async function refreshPaths() {
      const r = await fetch('/paths', { cache: 'no-store' });
      if (!r.ok) return;
      const data = await r.json();
      if (!data.paths || !data.paths.length) return;

      clearPathSeries();
      for (const p of data.paths) {
        const up = p[p.length - 1].value >= p[0].value;
        const color = up ? 'rgba(0,200,83,0.20)' : 'rgba(244,67,54,0.20)';
        const s = chart.addLineSeries({ color, lineWidth: 1 });
        s.setData(p);
        pathSeries.push(s);
      }
      meanSeries.setData(data.mean_path);
      p10Series.setData(data.confidence_band_10);
      p90Series.setData(data.confidence_band_90);

      document.getElementById('meta').textContent =
        `SIGNAL ${data.decision} | CONF ${data.confidence.toFixed(3)} | EV ${data.ev.toFixed(6)} | STD ${data.std.toFixed(6)} | SKEW ${data.skew.toFixed(4)} | REGIME ${data.regime}`;
    }

    async function loop() {
      await refreshOHLC();
      await refreshPaths();
      setTimeout(loop, 800);
    }
    loop();
  </script>
</body>
</html>
"""


def register_hybrid_routes(app, app_state: Any) -> None:
    """Register hybrid API and chart routes on the existing FastAPI app."""

    @app.get("/hybrid", response_class=HTMLResponse)
    async def hybrid_dashboard() -> str:
        return _hybrid_dashboard_html()

    @app.get("/paths")
    async def hybrid_paths() -> dict[str, Any]:
        event = app_state.inference_runner.latest_event
        if event is None:
            raise HTTPException(status_code=404, detail={"error": "no_prediction_available"})
        step_seconds = max(60, int(app_state.settings.get("data", {}).get("base_timeframe_minutes", 1)) * 60)
        start_ts = int(pd.Timestamp(event.bar_timestamp).timestamp())
        paths = paths_to_time_value(event.paths, start_ts=start_ts, step_seconds=step_seconds)
        summary = summarize_path_distribution(event.paths)
        mean_path = paths_to_time_value(summary["mean"][None, :], start_ts=start_ts, step_seconds=step_seconds)[0]
        p10_path = paths_to_time_value(summary["p10"][None, :], start_ts=start_ts, step_seconds=step_seconds)[0]
        p90_path = paths_to_time_value(summary["p90"][None, :], start_ts=start_ts, step_seconds=step_seconds)[0]
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": float(app_state.inference_runner.current_price()),
            "paths": paths,
            "decision": str(event.signal),
            "confidence": float(event.confidence),
            "ev": float(event.meta.get("ev", 0.0)),
            "std": float(event.meta.get("std", 0.0)),
            "skew": float(event.meta.get("skew", 0.0)),
            "regime": str(event.regime),
            "mean_path": mean_path,
            "confidence_band_10": p10_path,
            "confidence_band_90": p90_path,
        }

    @app.get("/signal")
    async def hybrid_signal() -> dict[str, Any]:
        event = app_state.inference_runner.latest_event
        if event is None:
            raise HTTPException(status_code=404, detail={"error": "no_prediction_available"})
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": str(event.signal),
            "confidence": float(event.confidence),
            "ev": float(event.meta.get("ev", 0.0)),
            "std": float(event.meta.get("std", 0.0)),
            "skew": float(event.meta.get("skew", 0.0)),
            "regime": str(event.regime),
        }

    @app.get("/ohlc")
    async def hybrid_ohlc(limit: int = Query(default=500, ge=50, le=2000)) -> list[dict[str, Any]]:
        # Use in-memory runner frame for near-real-time state.
        frame = getattr(app_state.inference_runner, "_ohlcv", None)
        if frame is None:
            frame = pd.read_parquet(app_state.ohlcv_path)
        bars = frame.tail(int(limit))
        payload: list[dict[str, Any]] = []
        for idx, row in bars.iterrows():
            payload.append(
                {
                    "time": int(pd.Timestamp(idx).timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
        return payload
