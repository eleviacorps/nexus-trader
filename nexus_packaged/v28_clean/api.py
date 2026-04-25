"""V28 clean FastAPI + clean web UI."""

from __future__ import annotations

import copy
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from nexus_packaged.v28_clean import engine as engine_module
from nexus_packaged.v28_clean.engine import V28CleanEngine, create_snapshot


class ToggleAutoPayload(BaseModel):
    enabled: bool


def _html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>NexusTrader V28 Clean</title>
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    body { margin:0; background:#0a0a0f; color:#d6f7ff; font-family: monospace; }
    #grid { display:grid; grid-template-rows: 72vh auto; min-height:100vh; }
    #chart { width:100%; height:72vh; border-bottom:1px solid #1f2d3a; }
    #panel { padding:10px; display:grid; gap:8px; }
    #debug { white-space:pre-wrap; border:1px solid #1f2d3a; padding:8px; background:#0f1420; }
    .row { display:flex; gap:8px; }
    button { background:#0f1420; border:1px solid #00e5ff; color:#d6f7ff; padding:6px 10px; cursor:pointer; }
  </style>
</head>
<body>
  <div id="grid">
    <div id="chart"></div>
    <div id="panel">
      <div class="row">
        <button id="toggleAuto">Toggle Auto</button>
        <button id="createSnap">Create Snapshot</button>
      </div>
      <div id="debug">Loading...</div>
    </div>
  </div>
  <script>
    const debugEl = document.getElementById("debug");
    const chart = LightweightCharts.createChart(document.getElementById("chart"), {
      layout: { background: { color: "#0a0a0f" }, textColor: "#d6f7ff" },
      grid: { vertLines: { color: "#1f2d3a" }, horzLines: { color: "#1f2d3a" } },
      timeScale: { borderColor: "#00e5ff" },
      rightPriceScale: { borderColor: "#00e5ff" }
    });
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#00c853",
      downColor: "#f44336",
      wickUpColor: "#00c853",
      wickDownColor: "#f44336",
      borderVisible: false
    });
    const meanSeries = chart.addLineSeries({ color: "rgba(255,255,255,0.95)", lineWidth: 3 });
    let sampleSeries = [];
    let initialized = false;

    function buildPath(path, baseTime, tf) {
      return path.map((p, i) => ({
        time: baseTime + (i + 1) * tf,
        value: p
      }));
    }

    function clearSamples() {
      for (const s of sampleSeries) chart.removeSeries(s);
      sampleSeries = [];
    }

    async function refreshCandles() {
      const res = await fetch("/ohlc?limit=500", { cache: "no-store" });
      if (!res.ok) return;
      const bars = await res.json();
      if (!Array.isArray(bars) || !bars.length) return;
      const visibleRange = chart.timeScale().getVisibleRange();
      candleSeries.setData(bars);
      if (initialized && visibleRange) {
        chart.timeScale().setVisibleRange(visibleRange);
      }
    }

    function renderState(data) {
      if (!data || !Array.isArray(data.paths) || !data.paths.length) {
        debugEl.textContent = "NO DATA";
        clearSamples();
        meanSeries.setData([]);
        return;
      }

      const baseTime = Number(data.base_time || Math.floor(Date.now() / 1000));
      const tf = Number(data.timeframe_sec || 60);
      const paths = data.paths.map(p => p.map(Number));
      const mean = paths[0].map((_, i) =>
        paths.reduce((sum, p) => sum + Number(p[i] || 0), 0) / paths.length
      );

      const visibleRange = chart.timeScale().getVisibleRange();
      clearSamples();
      const sampleCount = Math.min(8, paths.length);
      for (let i = 0; i < sampleCount; i++) {
        const s = chart.addLineSeries({ color: "rgba(0,229,255,0.18)", lineWidth: 1 });
        s.setData(buildPath(paths[i], baseTime, tf));
        sampleSeries.push(s);
      }
      meanSeries.setData(buildPath(mean, baseTime, tf));
      if (!initialized) {
        chart.timeScale().fitContent();
        initialized = true;
      } else if (visibleRange) {
        chart.timeScale().setVisibleRange(visibleRange);
      }

      const m = data.metrics || {};
      debugEl.textContent =
        `price: ${Number(data.price || 0).toFixed(3)}\\n` +
        `last update: ${new Date(Number(data.timestamp || 0) * 1000).toISOString()}\\n` +
        `ev: ${Number(m.ev || 0).toFixed(6)}\\n` +
        `std: ${Number(m.std || 0).toFixed(6)}\\n` +
        `prob_up: ${Number(m.prob_up || 0).toFixed(4)}\\n` +
        `prob_down: ${Number(m.prob_down || 0).toFixed(4)}\\n` +
        `confidence: ${Number(m.confidence || 0).toFixed(4)}\\n` +
        `decision: ${String(data.decision || "HOLD")}\\n` +
        `pipeline: ${String(data.pipeline_status || "UNKNOWN")}`;
    }

    async function pollState() {
      try {
        const res = await fetch("/state", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderState(data);
      } catch (err) {
        debugEl.textContent = `PIPELINE BROKEN\\n${String(err)}`;
        console.error(err);
      }
    }

    async function toggleAuto() {
      const state = await fetch("/state", { cache: "no-store" }).then(r => r.json());
      const current = Boolean(state.auto_trade_enabled);
      await fetch("/auto_trade/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !current })
      });
    }

    async function makeSnapshot() {
      await fetch("/snapshot/create", { method: "POST" });
    }

    document.getElementById("toggleAuto").addEventListener("click", toggleAuto);
    document.getElementById("createSnap").addEventListener("click", makeSnapshot);
    refreshCandles();
    setInterval(refreshCandles, 1000);
    setInterval(async () => {
      const res = await fetch("/state", { cache: "no-store" });
      const data = await res.json();
      renderState(data);
    }, 500);
    pollState();
  </script>
</body>
</html>
"""


def create_v28_app(engine: V28CleanEngine) -> FastAPI:
    """Create clean app with isolated state routes."""
    app = FastAPI(title="Nexus V28 Clean", version="v28-clean")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def home() -> str:
        return _html()

    @app.get("/state")
    async def get_state() -> dict[str, Any]:
        state = engine.get_state()
        if not state:
            raise HTTPException(status_code=503, detail={"error": "state_unavailable"})
        state["auto_trade_enabled"] = bool(engine.cfg.auto_trade_enabled)
        with engine_module._SNAPSHOT_LOCK:
            state["snapshot"] = copy.deepcopy(engine_module.SNAPSHOT)
        return state

    @app.get("/ohlc")
    async def get_ohlc(limit: int = 500) -> list[dict[str, float | int]]:
        return engine.get_ohlc_payload(limit=limit)

    @app.get("/snapshot")
    async def get_snapshot() -> dict[str, Any]:
        with engine_module._SNAPSHOT_LOCK:
            return {"snapshot": copy.deepcopy(engine_module.SNAPSHOT)}

    @app.post("/snapshot/create")
    async def snapshot_create() -> dict[str, Any]:
        state = engine.get_state()
        snap = create_snapshot(state)
        return {"snapshot": snap}

    @app.post("/auto_trade/toggle")
    async def toggle_auto(payload: ToggleAutoPayload) -> dict[str, Any]:
        engine.cfg.auto_trade_enabled = bool(payload.enabled)
        return {"auto_trade_enabled": bool(engine.cfg.auto_trade_enabled)}

    return app
