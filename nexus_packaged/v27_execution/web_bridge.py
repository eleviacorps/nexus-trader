"""Execution web bridge routes and lightweight chart dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException, Query
from fastapi.responses import HTMLResponse


def _paths_to_series(paths: np.ndarray, start_ts: int, step_seconds: int) -> list[list[dict[str, float | int]]]:
    series: list[list[dict[str, float | int]]] = []
    step = max(1, int(step_seconds))
    if paths.ndim != 2 or paths.size == 0:
        return series
    for row in paths:
        points: list[dict[str, float | int]] = []
        for i, v in enumerate(row.tolist()):
            fv = float(v)
            if not np.isfinite(fv):
                continue
            points.append({"time": int(start_ts + i * step), "value": fv})
        if len(points) >= 2:
            series.append(points)
    return series


def _execution_dashboard_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Nexus V27 Execution Chart</title>
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root { --bg:#0a0a0f; --panel:#0f1420; --line:#1f2d3a; --accent:#00e5ff; --text:#d6f7ff; --ok:#00c853; --bad:#f44336; --warn:#ff9800; }
    body { margin:0; background:var(--bg); color:var(--text); font-family: monospace; }
    #header { padding:10px 14px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:10px; align-items:center; }
    #layout { display:grid; grid-template-columns: 1fr 340px; min-height: calc(100vh - 49px); }
    #left { display:flex; flex-direction:column; min-width:0; }
    #chart { width:100%; height:74vh; border-bottom:1px solid var(--line); }
    #meta { padding:10px 14px; white-space:pre-wrap; min-height: 90px; }
    #right { border-left:1px solid var(--line); background:var(--panel); padding:10px; overflow:auto; }
    .sec { border:1px solid var(--line); padding:10px; margin-bottom:10px; }
    .sec h3 { margin:0 0 8px 0; color:var(--accent); font-size:13px; }
    .row { display:flex; gap:8px; margin-bottom:8px; }
    .row > * { flex:1; }
    label { display:block; font-size:12px; margin-bottom:4px; color:#9cc8d0; }
    input, select, textarea, button {
      width:100%; box-sizing:border-box; background:#0c111b; color:var(--text);
      border:1px solid var(--line); padding:6px; font-family: monospace;
    }
    textarea { min-height:140px; resize:vertical; }
    button { border-color:var(--accent); cursor:pointer; }
    .muted { color:#90a8b0; font-size:12px; }
    .ok { color:var(--ok); }
    .bad { color:var(--bad); }
    .warn { color:var(--warn); }
  </style>
</head>
<body>
  <div id="header">
    <div>Nexus V27 Execution Chart</div>
    <div style="display:flex; align-items:center; gap:14px;">
      <label style="display:flex; align-items:center; gap:6px; margin:0; color:#9cc8d0;">
        View
        <select id="viewMode" style="width:auto;">
          <option value="LIVE">LIVE</option>
          <option value="SNAPSHOT">SNAPSHOT</option>
        </select>
      </label>
      <label style="display:flex; align-items:center; gap:6px; margin:0; color:#9cc8d0;">
        <input id="autoFollowToggle" type="checkbox" checked style="width:auto;" />
        Auto-follow
      </label>
      <div id="quickState" class="muted">connecting...</div>
    </div>
  </div>
  <div id="layout">
    <div id="left">
      <div id="chart"></div>
      <div id="meta">Loading...</div>
    </div>
    <div id="right">
      <div class="sec">
        <h3>Execution Controls</h3>
        <div class="row">
          <button id="btnAutoToggle">Toggle Auto</button>
          <button id="btnReloadRuntime">Reload Runtime</button>
        </div>
        <button id="btnRefreshCfg" style="margin-bottom:8px;">Reload Config</button>
        <div class="row">
          <div>
            <label>Mode</label>
            <select id="cfgMode">
              <option value="selective">selective</option>
              <option value="frequency">frequency</option>
              <option value="interval">interval</option>
              <option value="count">count</option>
            </select>
          </div>
          <div>
            <label>Lot Mode</label>
            <select id="cfgLotMode">
              <option value="fixed">fixed</option>
              <option value="range">range</option>
              <option value="risk_pct">risk_pct</option>
              <option value="kelly">kelly</option>
            </select>
          </div>
        </div>
        <div class="row">
          <div>
            <label>Confidence Threshold</label>
            <input id="cfgConfidence" type="number" min="0" max="1" step="0.01" />
          </div>
          <div>
            <label>Fixed Lot Size</label>
            <input id="cfgFixedLot" type="number" min="0.01" step="0.01" />
          </div>
        </div>
        <div class="row">
          <div>
            <label>Risk Reward</label>
            <input id="cfgRR" type="number" min="1" max="8" step="0.1" />
          </div>
          <div>
            <label>Paper Mode</label>
            <select id="cfgPaper"><option value="true">true</option><option value="false">false</option></select>
          </div>
        </div>
        <button id="btnApplyCfg">Apply Config</button>
        <div id="cfgResult" class="muted" style="margin-top:8px;"></div>
      </div>
      <div class="sec">
        <h3>Advanced Config JSON</h3>
        <textarea id="cfgJson"></textarea>
        <button id="btnApplyJson">Apply Full Config</button>
      </div>
      <div class="sec">
        <h3>MT5 Runtime</h3>
        <div class="row">
          <div><label>Login</label><input id="mt5Login" type="number" /></div>
          <div><label>Server</label><input id="mt5Server" /></div>
        </div>
        <div class="row">
          <div><label>Password</label><input id="mt5Password" type="password" /></div>
          <div><label>Execution Enabled</label><select id="mt5Exec"><option value="false">false</option><option value="true">true</option></select></div>
        </div>
        <button id="btnApplyMt5">Apply MT5 Config</button>
        <div id="mt5Result" class="muted" style="margin-top:8px;"></div>
      </div>
    </div>
  </div>
  <script>
    const metaEl = document.getElementById("meta");
    const quickStateEl = document.getElementById("quickState");
    const viewModeEl = document.getElementById("viewMode");
    let chart = null;
    let candle = null;
    let meanSeries = null;
    let p10Series = null;
    let p90Series = null;
    let pathSeries = [];
    let latestState = null;
    let initialized = false;
    let autoFollow = true;
    let internalRangeUpdate = false;
    let lastBaseTime = 0;
    let lastTimeframeSec = 60;

    function createCandleSeries(chart, options) {
      if (typeof chart.addCandlestickSeries === "function") {
        return chart.addCandlestickSeries(options);
      }
      if (typeof chart.addSeries === "function" && LightweightCharts.CandlestickSeries) {
        return chart.addSeries(LightweightCharts.CandlestickSeries, options);
      }
      throw new Error("candlestick series API unavailable");
    }

    function createLineSeries(chart, options) {
      if (typeof chart.addLineSeries === "function") {
        return chart.addLineSeries(options);
      }
      if (typeof chart.addSeries === "function" && LightweightCharts.LineSeries) {
        return chart.addSeries(LightweightCharts.LineSeries, options);
      }
      throw new Error("line series API unavailable");
    }

    function initChart() {
      chart = LightweightCharts.createChart(document.getElementById("chart"), {
        layout: { background: { color: "#0a0a0f" }, textColor: "#d6f7ff" },
        grid: { vertLines: { color: "#1a2230" }, horzLines: { color: "#1a2230" } },
        rightPriceScale: { borderColor: "#00e5ff" },
        timeScale: { borderColor: "#00e5ff" }
      });
      candle = createCandleSeries(chart, {
        upColor: "#00c853",
        downColor: "#f44336",
        borderVisible: false,
        wickUpColor: "#00c853",
        wickDownColor: "#f44336"
      });
      meanSeries = createLineSeries(chart, { color: "rgba(255,255,255,0.95)", lineWidth: 3 });
      p10Series = createLineSeries(chart, { color: "rgba(255,235,59,0.85)", lineWidth: 1 });
      p90Series = createLineSeries(chart, { color: "rgba(255,235,59,0.85)", lineWidth: 1 });
      chart.timeScale().subscribeVisibleTimeRangeChange(() => {
        if (internalRangeUpdate || !initialized) return;
        autoFollow = false;
        const toggle = document.getElementById("autoFollowToggle");
        if (toggle) toggle.checked = false;
      });
      chart.subscribeCrosshairMove((param) => {
        if (!param || !param.time || !latestState) return;
        const time = Number(param.time);
        const offsetMin = Math.round((time - lastBaseTime) / 60);
        const meanPoint = param.seriesData?.get(meanSeries);
        const candlePoint = param.seriesData?.get(candle);
        const price = Number(meanPoint?.value ?? candlePoint?.close ?? latestState.price ?? 0);
        if (Number.isFinite(price) && offsetMin >= 0) {
          quickStateEl.textContent = `t+${offsetMin}m @ ${price.toFixed(2)} | mode ${viewModeEl.value}`;
        }
      });
    }

    function clearPaths() {
      if (!chart) return;
      for (const series of pathSeries) chart.removeSeries(series);
      pathSeries = [];
    }

    async function refreshOhlc() {
      try {
        const response = await fetch("/ohlc?limit=500", { cache: "no-store" });
        if (!response.ok) {
          metaEl.textContent = `NO DATA (OHLC HTTP ${response.status})`;
          return;
        }
        const bars = await response.json();
        if (Array.isArray(bars) && bars.length) {
          const prevRange = chart.timeScale().getVisibleRange();
          candle.setData(bars);
          if (!initialized) return;
          if (autoFollow) {
            internalRangeUpdate = true;
            chart.timeScale().scrollToRealTime();
            setTimeout(() => { internalRangeUpdate = false; }, 0);
          } else if (prevRange) {
            internalRangeUpdate = true;
            chart.timeScale().setVisibleRange(prevRange);
            setTimeout(() => { internalRangeUpdate = false; }, 0);
          }
        }
      } catch (err) {
        console.error("OHLC fetch failed", err);
        metaEl.textContent = `NO DATA (OHLC ERROR: ${String(err)})`;
      }
    }

    function buildSeriesFromRaw(rawPaths, baseTime, timeframeSec, horizonSteps) {
      if (!Array.isArray(rawPaths)) return [];
      const out = [];
      for (const path of rawPaths.slice(0, 64)) {
        if (!Array.isArray(path) || path.length < 2) continue;
        const useOffset = path.length >= (horizonSteps + 1) ? 1 : 0;
        const pts = [];
        for (let i = 0; i < horizonSteps; i++) {
          const idx = i + useOffset;
          if (idx >= path.length) break;
          const v = Number(path[idx]);
          if (!Number.isFinite(v)) continue;
          pts.push({ time: Number(baseTime) + (i + 1) * Number(timeframeSec), value: v });
        }
        if (pts.length >= 2) out.push(pts);
      }
      return out;
    }

    function summarizeBands(seriesPaths) {
      if (!Array.isArray(seriesPaths) || !seriesPaths.length) return { mean: [], p10: [], p90: [] };
      const steps = Math.min(...seriesPaths.map(p => p.length));
      const mean = [];
      const p10 = [];
      const p90 = [];
      for (let i = 0; i < steps; i++) {
        const time = seriesPaths[0][i].time;
        const vals = seriesPaths.map(p => Number(p[i].value)).filter(Number.isFinite).sort((a,b)=>a-b);
        if (!vals.length) continue;
        const m = vals.reduce((a,b)=>a+b,0) / vals.length;
        const i10 = Math.max(0, Math.floor((vals.length - 1) * 0.10));
        const i90 = Math.max(0, Math.floor((vals.length - 1) * 0.90));
        mean.push({ time, value: m });
        p10.push({ time, value: vals[i10] });
        p90.push({ time, value: vals[i90] });
      }
      return { mean, p10, p90 };
    }

    function drawState(data) {
      console.log("STATE:", data);
      if (!data || !data.live) {
        clearPaths();
        meanSeries.setData([]);
        p10Series.setData([]);
        p90Series.setData([]);
        metaEl.textContent = "NO DATA";
        return;
      }
      const mode = viewModeEl.value || "LIVE";
      const selected = (mode === "SNAPSHOT" && data.snapshot) ? data.snapshot : data.live;
      const rawPaths = Array.isArray(selected.paths) ? selected.paths : [];
      const baseTime = Number(data.base_time ?? 0);
      const timeframeSec = Number(data.timeframe_sec ?? 60);
      const horizonSteps = Number(data.horizon_steps ?? 20);
      lastBaseTime = baseTime;
      lastTimeframeSec = timeframeSec;

      if (!rawPaths.length) {
        clearPaths();
        meanSeries.setData([]);
        p10Series.setData([]);
        p90Series.setData([]);
        metaEl.textContent = `NO DATA (${mode})`;
        return;
      }

      // Alignment guard: path origin must track current price.
      const firstValue = Number(rawPaths[0]?.[0]);
      const refPrice = Number(
        (mode === "SNAPSHOT" ? selected.entry : undefined) ??
        data.live?.entry ??
        data.price ??
        0
      );
      const mismatch = Math.abs(firstValue - refPrice);
      const mismatchLimit = Math.max(2.0, Math.abs(refPrice) * 0.01);
      if (!Number.isFinite(firstValue) || !Number.isFinite(refPrice) || mismatch > mismatchLimit) {
        console.error("PATH/CANDLE SCALE MISMATCH", { firstValue, refPrice, mismatch, mismatchLimit });
        metaEl.textContent = `ALIGNMENT ERROR: path/candle scale mismatch (${mismatch.toFixed(4)})`;
        return;
      }

      const prevRange = chart.timeScale().getVisibleRange();
      clearPaths();
      const conf = Number(selected.confidence ?? 0);
      const alpha = Math.min(0.5, Math.max(0.18, conf * 0.7));
      const seriesPaths = buildSeriesFromRaw(rawPaths, baseTime, timeframeSec, horizonSteps);
      for (const path of seriesPaths) {
        const up = path[path.length - 1].value >= path[0].value;
        const color = up ? `rgba(0,200,83,${alpha})` : `rgba(244,67,54,${alpha})`;
        const series = createLineSeries(chart, { color, lineWidth: 2 });
        series.setData(path);
        pathSeries.push(series);
      }
      const bands = summarizeBands(seriesPaths);
      meanSeries.setData(bands.mean);
      p10Series.setData(bands.p10);
      p90Series.setData(bands.p90);
      if (!initialized) {
        internalRangeUpdate = true;
        chart.timeScale().fitContent();
        initialized = true;
        setTimeout(() => { internalRangeUpdate = false; }, 0);
      } else if (autoFollow) {
        internalRangeUpdate = true;
        chart.timeScale().scrollToRealTime();
        setTimeout(() => { internalRangeUpdate = false; }, 0);
      } else if (prevRange) {
        internalRangeUpdate = true;
        chart.timeScale().setVisibleRange(prevRange);
        setTimeout(() => { internalRangeUpdate = false; }, 0);
      }
      latestState = data;
      const liveLine = `LIVE: ${data.live.signal ?? "HOLD"} | CONF ${Number(data.live.confidence ?? 0).toFixed(3)} | EV ${Number(data.live.ev ?? 0).toFixed(5)}`;
      const snapLine = data.snapshot
        ? `SNAPSHOT: ${data.snapshot.signal ?? "HOLD"} | CONF ${Number(data.snapshot.confidence ?? 0).toFixed(3)} | ACTIVE ${Boolean(data.snapshot.active)}`
        : "SNAPSHOT: none";
      quickStateEl.textContent = `${liveLine} || ${snapLine}`;
      metaEl.textContent =
        `STATE: ${String(data.timestamp ?? "")} | LAST UPDATE: ${String(data.last_update_time ?? "")} | PRICE ${Number(data.price ?? 0).toFixed(3)}\n` +
        `${liveLine}\n${snapLine}\n` +
        `VIEW: ${mode} | BASE ${new Date(baseTime * 1000).toISOString()} | STEP ${timeframeSec}s | H ${horizonSteps}\n` +
        `ACTIVE: ${selected.signal ?? "HOLD"} | CONF ${Number(selected.confidence ?? 0).toFixed(3)} | ` +
        `EV ${Number(selected.ev ?? 0).toFixed(6)} | STD ${Number(selected.std ?? 0).toFixed(6)} | ` +
        `EV THR ${Number(selected.ev_threshold ?? 0).toFixed(6)} | REGIME ${selected.regime ?? "UNKNOWN"}`;
    }

    async function pollState() {
      try {
        const response = await fetch("/state", { cache: "no-store" });
        if (!response.ok) {
          metaEl.textContent = `NO DATA (STATE HTTP ${response.status})`;
          return;
        }
        const data = await response.json();
        drawState(data);
      } catch (err) {
        console.error("STATE fetch failed", err);
        metaEl.textContent = `NO DATA (STATE ERROR: ${String(err)})`;
      }
    }

    async function loadConfig() {
      try {
        const r = await fetch("/execution/config", { cache: "no-store" });
        if (!r.ok) return;
        const payload = await r.json();
        const cfg = payload.auto_trade || {};
        document.getElementById("cfgMode").value = cfg.mode ?? "selective";
        document.getElementById("cfgLotMode").value = cfg.lot_mode ?? "fixed";
        document.getElementById("cfgConfidence").value = cfg.confidence_threshold ?? 0.65;
        document.getElementById("cfgFixedLot").value = cfg.fixed_lot_size ?? 0.01;
        document.getElementById("cfgRR").value = cfg.risk_reward ?? 2.0;
        document.getElementById("cfgPaper").value = String(Boolean(cfg.paper_mode));
        document.getElementById("cfgJson").value = JSON.stringify(cfg, null, 2);

        const mt5 = payload.mt5 || {};
        document.getElementById("mt5Login").value = mt5.login ?? 0;
        document.getElementById("mt5Server").value = mt5.server ?? "";
        document.getElementById("mt5Exec").value = String(Boolean(mt5.execution_enabled));
      } catch (err) {
        console.error("loadConfig failed", err);
      }
    }

    async function applyConfigFromInputs() {
      const cfg = JSON.parse(document.getElementById("cfgJson").value || "{}");
      cfg.mode = document.getElementById("cfgMode").value;
      cfg.lot_mode = document.getElementById("cfgLotMode").value;
      cfg.confidence_threshold = Number(document.getElementById("cfgConfidence").value || cfg.confidence_threshold || 0.65);
      cfg.fixed_lot_size = Number(document.getElementById("cfgFixedLot").value || cfg.fixed_lot_size || 0.01);
      cfg.risk_reward = Number(document.getElementById("cfgRR").value || cfg.risk_reward || 2.0);
      cfg.paper_mode = document.getElementById("cfgPaper").value === "true";
      const r = await fetch("/auto_trade/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      const txt = await r.text();
      document.getElementById("cfgResult").textContent = r.ok ? `updated: ${txt}` : `error: ${txt}`;
      if (r.ok) document.getElementById("cfgJson").value = JSON.stringify(cfg, null, 2);
    }

    async function applyConfigFromJson() {
      const cfg = JSON.parse(document.getElementById("cfgJson").value || "{}");
      const r = await fetch("/auto_trade/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      const txt = await r.text();
      document.getElementById("cfgResult").textContent = r.ok ? `updated: ${txt}` : `error: ${txt}`;
    }

    async function toggleAuto() {
      const nextEnabled = !(latestState?.auto_trade_enabled ?? false);
      const r = await fetch("/auto_trade/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: nextEnabled }),
      });
      const txt = await r.text();
      document.getElementById("cfgResult").textContent = r.ok ? `toggle ok: ${txt}` : `toggle error: ${txt}`;
    }

    async function reloadRuntime() {
      const r = await fetch("/reload", { method: "POST", cache: "no-store" });
      const txt = await r.text();
      document.getElementById("cfgResult").textContent = r.ok ? `reloaded: ${txt}` : `reload error: ${txt}`;
    }

    async function applyMt5Config() {
      const payload = {
        login: Number(document.getElementById("mt5Login").value || 0),
        server: document.getElementById("mt5Server").value || "",
        password: document.getElementById("mt5Password").value || "",
        execution_enabled: document.getElementById("mt5Exec").value === "true",
        reconnect_now: true,
      };
      const r = await fetch("/mt5/account/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const txt = await r.text();
      document.getElementById("mt5Result").textContent = r.ok ? `updated: ${txt}` : `error: ${txt}`;
    }

    try {
      initChart();
      const autoFollowToggle = document.getElementById("autoFollowToggle");
      if (autoFollowToggle) {
        autoFollowToggle.addEventListener("change", (ev) => {
          autoFollow = Boolean(ev.target.checked);
          if (autoFollow && chart) {
            internalRangeUpdate = true;
            chart.timeScale().scrollToRealTime();
            setTimeout(() => { internalRangeUpdate = false; }, 0);
          }
        });
      }
      if (viewModeEl) {
        viewModeEl.addEventListener("change", () => {
          if (latestState) drawState(latestState);
        });
      }
      document.getElementById("btnAutoToggle").addEventListener("click", toggleAuto);
      document.getElementById("btnReloadRuntime").addEventListener("click", reloadRuntime);
      document.getElementById("btnRefreshCfg").addEventListener("click", loadConfig);
      document.getElementById("btnApplyCfg").addEventListener("click", applyConfigFromInputs);
      document.getElementById("btnApplyJson").addEventListener("click", applyConfigFromJson);
      document.getElementById("btnApplyMt5").addEventListener("click", applyMt5Config);
      refreshOhlc();
      setInterval(refreshOhlc, 3000);
      setInterval(pollState, 500);
      loadConfig();
      pollState();
    } catch (err) {
      console.error("CHART INIT FAILED", err);
      metaEl.textContent = `CHART INIT FAILED: ${String(err)}`;
    }
  </script>
</body>
</html>
"""


def _state_payload(app_state: Any) -> dict[str, Any]:
    if hasattr(app_state.inference_runner, "get_global_state"):
        payload = app_state.inference_runner.get_global_state()
        if payload:
            return payload
    raise HTTPException(status_code=404, detail={"error": "no_prediction_available"})


def register_execution_routes(app, app_state: Any) -> None:
    """Register execution chart and data routes."""

    @app.get("/execution", response_class=HTMLResponse)
    async def execution_dashboard() -> str:
        return _execution_dashboard_html()

    @app.get("/paths")
    async def execution_paths() -> dict[str, Any]:
        return _state_payload(app_state)

    @app.get("/execution/config")
    async def execution_config() -> dict[str, Any]:
        auto_cfg = app_state.auto_trader.config
        mt5_cfg = app_state.mt5_connector.get_runtime_config()
        return {
            "auto_trade": dict(vars(auto_cfg)),
            "mt5": {
                "login": int(mt5_cfg.get("login", 0)),
                "server": str(mt5_cfg.get("server", "")),
                "execution_enabled": bool(mt5_cfg.get("execution_enabled", False)),
                "connected": bool(mt5_cfg.get("connected", False)),
            },
        }

    @app.get("/signal")
    async def execution_signal() -> dict[str, Any]:
        payload = _state_payload(app_state)
        return {
            "timestamp": payload["timestamp"],
            "decision": payload["decision"],
            "confidence": payload["confidence"],
            "ev": payload["ev"],
            "std": payload["std"],
            "skew": payload["skew"],
            "ev_threshold": payload["ev_threshold"],
            "regime": payload["regime"],
        }

    @app.get("/ohlc")
    async def execution_ohlc(limit: int = Query(default=500, ge=50, le=2000)) -> list[dict[str, Any]]:
        frame = None
        connector = getattr(app_state, "mt5_connector", None)
        settings = getattr(app_state, "settings", {})
        symbol = str(settings.get("data", {}).get("symbol", "XAUUSD"))
        if connector is not None and bool(getattr(connector, "is_connected", False)):
            mt5_mod = getattr(connector, "_mt5", None)
            timeframe_map = {
                1: "TIMEFRAME_M1",
                5: "TIMEFRAME_M5",
                15: "TIMEFRAME_M15",
                30: "TIMEFRAME_M30",
                60: "TIMEFRAME_H1",
                240: "TIMEFRAME_H4",
            }
            timeframe_minutes = int(settings.get("data", {}).get("base_timeframe_minutes", 1))
            timeframe_name = timeframe_map.get(timeframe_minutes, "TIMEFRAME_M1")
            timeframe = getattr(mt5_mod, timeframe_name, None) if mt5_mod is not None else None
            if timeframe is not None:
                try:
                    mt5_frame = await connector.get_latest_bars(symbol, timeframe, int(limit))
                    if mt5_frame is not None and not mt5_frame.empty:
                        frame = mt5_frame
                except Exception:  # noqa: BLE001
                    frame = None
        if frame is None:
            frame = getattr(app_state.inference_runner, "_ohlcv", None)
            if frame is None:
                frame = pd.read_parquet(app_state.ohlcv_path)
        bars = frame.tail(int(limit))
        payload: list[dict[str, Any]] = []
        for idx, row in bars.iterrows():
            open_v = float(row["open"])
            high_v = float(row["high"])
            low_v = float(row["low"])
            close_v = float(row["close"])
            if not np.isfinite(open_v + high_v + low_v + close_v):
                continue
            payload.append(
                {
                    "time": int(pd.Timestamp(idx).timestamp()),
                    "open": open_v,
                    "high": high_v,
                    "low": low_v,
                    "close": close_v,
                }
            )
        return payload
