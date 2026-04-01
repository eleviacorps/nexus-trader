from __future__ import annotations


def render_web_app_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nexus Trader Simulator</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #071018;
      --panel: rgba(10, 22, 34, 0.92);
      --panel-alt: rgba(16, 31, 47, 0.95);
      --text: #eef5fb;
      --muted: #92a8b8;
      --border: rgba(255,255,255,0.08);
      --bull: #2ecc71;
      --bear: #ff5a5f;
      --accent: #4da3ff;
      --cone: rgba(46, 204, 113, 0.18);
      --shadow: 0 16px 40px rgba(0,0,0,0.28);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Segoe UI", system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(77,163,255,0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(46,204,113,0.12), transparent 28%),
        linear-gradient(160deg, #04090e, #071018 46%, #091722);
    }
    .shell {
      max-width: 1780px;
      margin: 0 auto;
      padding: 22px;
      display: grid;
      gap: 18px;
    }
    .hero {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      flex-wrap: wrap;
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: 0.02em;
    }
    .hero p {
      margin: 8px 0 0;
      max-width: 920px;
      color: var(--muted);
      line-height: 1.55;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      font-size: 12px;
      color: var(--text);
    }
    .panel, .card, .controls {
      border-radius: 18px;
      border: 1px solid var(--border);
      background: linear-gradient(180deg, var(--panel-alt), var(--panel));
      box-shadow: var(--shadow);
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      padding: 14px;
    }
    .control {
      display: grid;
      gap: 6px;
    }
    label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }
    select, button, input[type="checkbox"] {
      accent-color: var(--bull);
    }
    select, button {
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06);
      color: var(--text);
      padding: 10px 14px;
      font-size: 14px;
    }
    button {
      cursor: pointer;
      font-weight: 700;
      background: linear-gradient(135deg, #1d9b61, #156d45);
    }
    button:hover { filter: brightness(1.08); }
    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 22px;
      color: var(--muted);
      font-size: 13px;
    }
    .status {
      margin-left: auto;
      font-size: 13px;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 12px;
    }
    .card {
      padding: 16px;
      min-height: 110px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }
    .value {
      margin-top: 10px;
      font-size: clamp(22px, 2vw, 32px);
      font-weight: 700;
    }
    .sub {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(0, 1.4fr) minmax(340px, 0.95fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      padding: 16px;
    }
    .panel h2 {
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
    }
    .panel .caption {
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    #live-chart, #compare-chart {
      height: 680px;
      width: 100%;
    }
    .stack {
      display: grid;
      gap: 16px;
    }
    .list {
      display: grid;
      gap: 10px;
      max-height: 240px;
      overflow: auto;
      padding-right: 4px;
    }
    .list-item {
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
      padding: 12px;
    }
    .list-item a {
      color: var(--text);
      text-decoration: none;
      font-weight: 600;
    }
    .meta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .persona-list, .macro, .history-list {
      display: grid;
      gap: 10px;
    }
    .persona-row {
      display: grid;
      grid-template-columns: 90px 1fr 58px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }
    .bar {
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255,255,255,0.07);
    }
    .bar > span {
      display: block;
      height: 100%;
      border-radius: 999px;
    }
    .history-item {
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
      padding: 12px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.5;
    }
    .history-item strong {
      color: var(--text);
    }
    .js-plotly-plot .plotly .modebar {
      background: rgba(8, 16, 24, 0.78) !important;
      border-radius: 10px;
      padding: 2px;
    }
    .js-plotly-plot .plotly .modebar-btn svg {
      fill: #c8d8e5 !important;
    }
    .js-plotly-plot .plotly .hoverlayer .hovertext rect {
      fill: #0d1b26 !important;
      stroke: rgba(255,255,255,0.14) !important;
    }
    .js-plotly-plot .plotly .hoverlayer .hovertext text {
      fill: #eef5fb !important;
    }
    @media (max-width: 1440px) {
      .grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .layout { grid-template-columns: 1fr; }
      #live-chart, #compare-chart { height: 560px; }
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <h1>Nexus Trader Market Simulator</h1>
        <p>This console is for simulation quality, not execution. The left chart keeps updating with live market candles. The right chart freezes the predicted 5 minute future from the moment you press simulate and then overlays the real candles that arrive afterward so we can see whether the market tracks the scenario.</p>
      </div>
      <div class="pill">Dark mode - live market vs predicted future</div>
    </section>

    <section class="controls">
      <div class="control">
        <label for="symbol">Instrument</label>
        <select id="symbol">
          <option value="XAUUSD">XAUUSD</option>
          <option value="EURUSD">EURUSD</option>
          <option value="BTCUSD">BTCUSD</option>
        </select>
      </div>
      <div class="control">
        <label for="refresh-seconds">Refresh</label>
        <select id="refresh-seconds">
          <option value="15">15 sec</option>
          <option value="30" selected>30 sec</option>
          <option value="60">60 sec</option>
        </select>
      </div>
      <button id="simulate">Simulate Next 5 Minutes</button>
      <label class="toggle"><input type="checkbox" id="auto-refresh" checked /> Auto refresh live market</label>
      <div id="status" class="status">Loading simulator...</div>
    </section>

    <section class="grid">
      <article class="card"><div class="label">Current Price</div><div class="value" id="metric-price">-</div><div class="sub" id="metric-price-change">-</div></article>
      <article class="card"><div class="label">Simulation Confidence</div><div class="value" id="metric-confidence">-</div><div class="sub">Whole simulation conviction after collapse</div></article>
      <article class="card"><div class="label">Consensus Score</div><div class="value" id="metric-consensus">-</div><div class="sub">Agreement across surviving futures</div></article>
      <article class="card"><div class="label">Uncertainty Width</div><div class="value" id="metric-uncertainty">-</div><div class="sub">Cone spread from branch disagreement</div></article>
      <article class="card"><div class="label">Scenario Bias</div><div class="value" id="metric-bias">-</div><div class="sub" id="metric-dominant-driver">-</div></article>
      <article class="card"><div class="label">Neural Bias</div><div class="value" id="metric-neural">-</div><div class="sub" id="metric-neural-detail">Latest model output</div></article>
      <article class="card"><div class="label">Cone Hit Rate</div><div class="value" id="metric-hit-rate">-</div><div class="sub" id="metric-direction-match">Waiting for real candles after simulation</div></article>
      <article class="card"><div class="label">Timestamps</div><div class="value" id="metric-times">-</div><div class="sub" id="metric-live-time">-</div></article>
    </section>

    <section class="layout">
      <article class="panel">
        <h2>Live Market</h2>
        <div class="caption">Realtime candles keep moving here. The orange marker shows where the latest manual simulation started.</div>
        <div id="live-chart"></div>
      </article>

      <article class="panel">
        <h2>Predicted vs Actual</h2>
        <div class="caption">Predicted center path and cone are frozen from the simulation timestamp. Actual closes that arrive afterward are overlaid to test whether the scenario is tracking.</div>
        <div id="compare-chart"></div>
      </article>

      <div class="stack">
        <article class="panel">
          <h2>Persona Impact</h2>
          <div id="personas" class="persona-list"></div>
        </article>

        <article class="panel">
          <h2>Recent Simulation History</h2>
          <div id="history" class="history-list"></div>
        </article>

        <article class="panel">
          <h2>Macro Pulse</h2>
          <div id="macro" class="macro"></div>
        </article>

        <article class="panel">
          <h2>Headline Context</h2>
          <div id="news" class="list"></div>
        </article>

        <article class="panel">
          <h2>Public Discussion Pulse</h2>
          <div id="crowd" class="list"></div>
        </article>
      </div>
    </section>
  </div>

  <script>
    const state = {
      activeSymbol: 'XAUUSD',
      lastSimulation: null,
      lastMonitor: null,
      refreshTimer: null,
    };

    function fmtPct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
      return `${(Number(value) * 100).toFixed(1)}%`;
    }

    function fmtPrice(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
      return Number(value).toFixed(2);
    }

    function fmtTime(value) {
      if (!value) return '-';
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value);
      return date.toLocaleString();
    }

    function setStatus(message) {
      document.getElementById('status').textContent = message;
    }

    function colorForSignal(signal) {
      return signal === 'bullish' ? 'var(--bull)' : 'var(--bear)';
    }

    function renderMetrics(simPayload, monitorPayload) {
      const sim = (simPayload || {}).simulation || {};
      const market = (simPayload || {}).market || {};
      const model = (simPayload || {}).model_prediction || {};
      const activePrediction = (monitorPayload || {}).active_prediction || {};

      document.getElementById('metric-price').textContent = fmtPrice(market.current_price);
      const change = Number(market.price_change || 0);
      document.getElementById('metric-price-change').textContent = `${change >= 0 ? '+' : ''}${fmtPrice(change)} vs previous candle`;
      document.getElementById('metric-price-change').style.color = change >= 0 ? 'var(--bull)' : 'var(--bear)';

      document.getElementById('metric-confidence').textContent = fmtPct(sim.overall_confidence);
      document.getElementById('metric-consensus').textContent = fmtPct(sim.consensus_score);
      document.getElementById('metric-uncertainty').textContent = fmtPct(sim.uncertainty_width);
      document.getElementById('metric-bias').textContent = (sim.scenario_bias || '-').toUpperCase();
      document.getElementById('metric-bias').style.color = colorForSignal(sim.scenario_bias);
      document.getElementById('metric-dominant-driver').textContent = sim.dominant_driver || 'No dominant driver';

      if (model && typeof model.bullish_probability === 'number') {
        document.getElementById('metric-neural').textContent = fmtPct(model.bullish_probability);
        document.getElementById('metric-neural').style.color = colorForSignal(model.signal);
        document.getElementById('metric-neural-detail').textContent = `${(model.signal || 'neutral').toUpperCase()} | threshold ${Number(model.threshold || 0.5).toFixed(3)}`;
      } else {
        document.getElementById('metric-neural').textContent = 'N/A';
        document.getElementById('metric-neural').style.color = 'var(--text)';
        document.getElementById('metric-neural-detail').textContent = 'Live model inference unavailable';
      }

      document.getElementById('metric-hit-rate').textContent = activePrediction && activePrediction.hit_rate !== null && activePrediction.hit_rate !== undefined
        ? fmtPct(activePrediction.hit_rate)
        : 'Pending';
      const directionText = activePrediction && activePrediction.matched_points
        ? `${activePrediction.realized_direction.toUpperCase()} | direction match: ${activePrediction.direction_match ? 'YES' : 'NO'}`
        : 'Waiting for real candles after simulation';
      document.getElementById('metric-direction-match').textContent = directionText;
      document.getElementById('metric-direction-match').style.color =
        activePrediction && activePrediction.direction_match === true ? 'var(--bull)' :
        activePrediction && activePrediction.direction_match === false ? 'var(--bear)' : 'var(--muted)';

      document.getElementById('metric-times').textContent = fmtTime((simPayload || {}).generated_at);
      const liveCandle = ((monitorPayload || {}).live_market || []).slice(-1)[0];
      document.getElementById('metric-live-time').textContent = liveCandle ? `Latest live candle: ${fmtTime(liveCandle.timestamp)}` : '-';
    }

    function renderPersonas(personas) {
      const host = document.getElementById('personas');
      host.innerHTML = '';
      const entries = Object.entries(personas || {});
      if (!entries.length) {
        host.innerHTML = '<div class="list-item">No persona impacts available.</div>';
        return;
      }
      const maxAbs = Math.max(...entries.map(([, value]) => Math.abs(Number(value))), 0.001);
      for (const [name, valueRaw] of entries.sort((a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])))) {
        const value = Number(valueRaw);
        const width = Math.max(6, Math.round((Math.abs(value) / maxAbs) * 100));
        const gradient = value >= 0
          ? 'linear-gradient(90deg, rgba(46,204,113,0.45), rgba(46,204,113,1))'
          : 'linear-gradient(90deg, rgba(255,90,95,0.45), rgba(255,90,95,1))';
        const row = document.createElement('div');
        row.className = 'persona-row';
        row.innerHTML = `
          <div>${name}</div>
          <div class="bar"><span style="width:${width}%; background:${gradient}"></span></div>
          <div style="text-align:right;color:${value >= 0 ? 'var(--bull)' : 'var(--bear)'}">${value.toFixed(3)}</div>
        `;
        host.appendChild(row);
      }
    }

    function renderList(containerId, items) {
      const host = document.getElementById(containerId);
      host.innerHTML = '';
      if (!items || !items.length) {
        host.innerHTML = '<div class="list-item">No live items available right now.</div>';
        return;
      }
      for (const item of items) {
        const meta = [item.source, fmtTime(item.published_at), item.classification, item.sentiment !== undefined ? `sentiment ${Number(item.sentiment).toFixed(2)}` : '']
          .filter(Boolean)
          .join(' | ');
        const block = document.createElement('div');
        block.className = 'list-item';
        const title = item.title || item.source || 'Untitled';
        block.innerHTML = item.link
          ? `<a href="${item.link}" target="_blank" rel="noreferrer">${title}</a><div class="meta">${meta}</div>`
          : `<div>${title}</div><div class="meta">${meta}</div>`;
        host.appendChild(block);
      }
    }

    function renderMacro(payload) {
      const macro = (payload.feeds || {}).macro || {};
      const host = document.getElementById('macro');
      const rows = Object.entries(macro.components || {}).map(([key, value]) => `<div><strong>${key}</strong>: ${Number(value).toFixed(2)}</div>`).join('');
      host.innerHTML = `
        <div><strong>Driver</strong>: ${macro.driver || 'macro_neutral'}</div>
        <div><strong>Bias</strong>: ${fmtPct((Number(macro.macro_bias || 0) + 1) / 2)}</div>
        <div><strong>Shock</strong>: ${fmtPct(Number(macro.macro_shock || 0))}</div>
        ${rows || '<div>No live macro components available.</div>'}
      `;
    }

    function renderHistory(monitorPayload) {
      const host = document.getElementById('history');
      host.innerHTML = '';
      const items = (monitorPayload || {}).recent_simulations || [];
      if (!items.length) {
        host.innerHTML = '<div class="history-item">No recorded simulations yet. Press simulate to create a timestamped future path.</div>';
        return;
      }
      for (const item of items.slice(0, 6)) {
        const directionLine = item.realized_direction === 'pending'
          ? 'Waiting for real candles'
          : `${item.realized_direction.toUpperCase()} | match ${item.direction_match ? 'YES' : 'NO'}`;
        const node = document.createElement('div');
        node.className = 'history-item';
        node.innerHTML = `
          <div><strong>${fmtTime(item.generated_at)}</strong></div>
          <div>${(item.scenario_bias || 'neutral').toUpperCase()} | confidence ${fmtPct(item.overall_confidence)}</div>
          <div>${directionLine}</div>
          <div>Hit rate ${item.hit_rate !== null && item.hit_rate !== undefined ? fmtPct(item.hit_rate) : 'Pending'} | points ${item.matched_points || 0}</div>
          <div>Driver: ${item.dominant_driver || 'unknown'}</div>
        `;
        host.appendChild(node);
      }
    }

    function renderLiveChart(simPayload, monitorPayload) {
      const candles = (monitorPayload || {}).live_market || (simPayload.market || {}).candles || [];
      const activePrediction = (monitorPayload || {}).active_prediction || null;
      const anchorTs = activePrediction && activePrediction.simulation ? activePrediction.simulation.anchor_timestamp : null;

      const traces = [{
        type: 'candlestick',
        x: candles.map(row => row.timestamp),
        open: candles.map(row => row.open),
        high: candles.map(row => row.high),
        low: candles.map(row => row.low),
        close: candles.map(row => row.close),
        increasing: { line: { color: '#2ecc71' }, fillcolor: '#2ecc71' },
        decreasing: { line: { color: '#ff5a5f' }, fillcolor: '#ff5a5f' },
        name: 'Live market',
      }];

      const shapes = [];
      if (anchorTs) {
        shapes.push({
          type: 'line',
          x0: anchorTs,
          x1: anchorTs,
          yref: 'paper',
          y0: 0,
          y1: 1,
          line: { color: '#f39c12', width: 2, dash: 'dash' },
        });
      }

      Plotly.react('live-chart', traces, {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(8,16,24,0.22)',
        font: { color: '#eef5fb' },
        xaxis: { rangeslider: { visible: false }, gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
        margin: { l: 40, r: 12, t: 8, b: 32 },
        hovermode: 'x unified',
        hoverlabel: {
          bgcolor: '#0d1b26',
          bordercolor: 'rgba(255,255,255,0.18)',
          font: { color: '#eef5fb' },
        },
        shapes,
        uirevision: `${state.activeSymbol}-live-chart`,
      }, { responsive: true, displaylogo: false });
    }

    function renderCompareChart(monitorPayload) {
      const activePrediction = (monitorPayload || {}).active_prediction || null;
      if (!activePrediction || !activePrediction.simulation) {
        Plotly.react('compare-chart', [], {
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(8,16,24,0.22)',
          font: { color: '#eef5fb' },
          annotations: [{
            text: 'Press "Simulate Next 5 Minutes" to freeze a scenario and compare future candles against it.',
            showarrow: false,
            x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
            font: { size: 14, color: '#92a8b8' },
          }],
          xaxis: { visible: false },
          yaxis: { visible: false },
          margin: { l: 20, r: 20, t: 20, b: 20 },
        }, { responsive: true, displaylogo: false });
        return;
      }

      const simulation = activePrediction.simulation;
      const cone = simulation.cone || [];
      const center = simulation.center_path || [];
      const actual = activePrediction.actual_future || [];
      const anchorTs = simulation.anchor_timestamp;
      const anchorPrice = Number(simulation.anchor_price || 0);

      const traces = [{
        type: 'scatter',
        mode: 'markers',
        x: [anchorTs],
        y: [anchorPrice],
        marker: { size: 10, color: '#f39c12' },
        name: 'Simulation anchor',
      }];

      if (cone.length) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: cone.map(row => row.timestamp),
          y: cone.map(row => row.lower_price),
          line: { color: 'rgba(46,204,113,0.0)' },
          hoverinfo: 'skip',
          showlegend: false,
        });
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: cone.map(row => row.timestamp),
          y: cone.map(row => row.upper_price),
          line: { color: 'rgba(46,204,113,0.0)' },
          fill: 'tonexty',
          fillcolor: 'rgba(46,204,113,0.18)',
          name: 'Probability cone',
          hoverinfo: 'skip',
        });
      }

      traces.push({
        type: 'scatter',
        mode: 'lines+markers',
        x: [anchorTs].concat(center.map(row => row.timestamp)),
        y: [anchorPrice].concat(center.map(row => row.price)),
        line: { color: '#2ecc71', width: 3 },
        marker: { size: 6, color: '#2ecc71' },
        name: 'Predicted center',
      });

      if (actual.length) {
        traces.push({
          type: 'scatter',
          mode: 'lines+markers',
          x: [anchorTs].concat(actual.map(row => row.timestamp)),
          y: [anchorPrice].concat(actual.map(row => row.close)),
          line: { color: '#4da3ff', width: 3 },
          marker: {
            size: 7,
            color: actual.map(row => row.inside_cone ? '#4da3ff' : '#ff5a5f'),
          },
          name: 'Actual after simulation',
        });
      }

      Plotly.react('compare-chart', traces, {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(8,16,24,0.22)',
        font: { color: '#eef5fb' },
        xaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
        margin: { l: 40, r: 12, t: 8, b: 32 },
        hovermode: 'x unified',
        hoverlabel: {
          bgcolor: '#0d1b26',
          bordercolor: 'rgba(255,255,255,0.18)',
          font: { color: '#eef5fb' },
        },
        legend: { orientation: 'h', x: 0, y: 1.08 },
        uirevision: `${state.activeSymbol}-compare-chart`,
      }, { responsive: true, displaylogo: false });
    }

    async function simulate(symbol, announce = true) {
      state.activeSymbol = symbol;
      if (announce) setStatus(`Simulating ${symbol} with live candles, headlines, and public-discussion context...`);
      const response = await fetch(`/api/simulate-live?symbol=${encodeURIComponent(symbol)}`);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Request failed with ${response.status}`);
      }
      state.lastSimulation = await response.json();
      renderPersonas(state.lastSimulation.personas || {});
      renderMacro(state.lastSimulation);
      renderList('news', (state.lastSimulation.feeds || {}).news || []);
      renderList('crowd', (state.lastSimulation.feeds || {}).public_discussions || []);
      await refreshMonitor(false);
      setStatus(`Simulation frozen at ${fmtTime(state.lastSimulation.generated_at)} for ${symbol}. Live market will keep refreshing beside it.`);
    }

    async function refreshMonitor(announce = false) {
      if (!state.activeSymbol) return;
      if (announce) setStatus(`Refreshing live market for ${state.activeSymbol}...`);
      const response = await fetch(`/api/live-monitor?symbol=${encodeURIComponent(state.activeSymbol)}`);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Request failed with ${response.status}`);
      }
      state.lastMonitor = await response.json();
      if (state.lastSimulation) {
        renderMetrics(state.lastSimulation, state.lastMonitor);
        renderLiveChart(state.lastSimulation, state.lastMonitor);
        renderCompareChart(state.lastMonitor);
        renderHistory(state.lastMonitor);
      }
      if (announce) setStatus(`Live market refreshed at ${fmtTime((state.lastMonitor || {}).server_timestamp)}.`);
    }

    function startRefreshLoop() {
      if (state.refreshTimer) {
        clearInterval(state.refreshTimer);
        state.refreshTimer = null;
      }
      if (!document.getElementById('auto-refresh').checked) {
        return;
      }
      const seconds = Number(document.getElementById('refresh-seconds').value || 30);
      state.refreshTimer = setInterval(() => {
        refreshMonitor(false).catch((error) => {
          console.error(error);
          setStatus(`Live refresh failed: ${error.message}`);
        });
      }, seconds * 1000);
    }

    document.getElementById('simulate').addEventListener('click', async () => {
      try {
        await simulate(document.getElementById('symbol').value, true);
      } catch (error) {
        console.error(error);
        setStatus(`Simulation failed: ${error.message}`);
      }
    });

    document.getElementById('symbol').addEventListener('change', async (event) => {
      try {
        await simulate(event.target.value, true);
      } catch (error) {
        console.error(error);
        setStatus(`Simulation failed: ${error.message}`);
      }
    });

    document.getElementById('refresh-seconds').addEventListener('change', startRefreshLoop);
    document.getElementById('auto-refresh').addEventListener('change', startRefreshLoop);

    window.addEventListener('load', async () => {
      try {
        await simulate(document.getElementById('symbol').value, false);
        startRefreshLoop();
      } catch (error) {
        console.error(error);
        setStatus(`Initial load failed: ${error.message}`);
      }
    });
  </script>
</body>
</html>
"""
