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
      --panel: #0d1b26;
      --panel-2: #132433;
      --text: #eef5fb;
      --muted: #9db0bf;
      --accent: #26c281;
      --warn: #f39c12;
      --bear: #ff5a5f;
      --bull: #2ecc71;
      --border: rgba(255,255,255,0.08);
      --shadow: 0 18px 40px rgba(0,0,0,0.28);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(46, 204, 113, 0.14), transparent 30%),
        radial-gradient(circle at top right, rgba(255, 90, 95, 0.12), transparent 28%),
        linear-gradient(160deg, #04090e, #071018 45%, #091722);
      color: var(--text);
      min-height: 100vh;
    }
    .shell {
      max-width: 1680px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 20px;
    }
    .hero {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      flex-wrap: wrap;
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: 0.02em;
    }
    .hero p {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 780px;
      line-height: 1.5;
    }
    .controls, .card, .panel {
      background: linear-gradient(180deg, rgba(19, 36, 51, 0.96), rgba(11, 24, 35, 0.96));
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px;
      flex-wrap: wrap;
    }
    label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
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
      background: linear-gradient(135deg, #1f9d62, #16794c);
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { filter: brightness(1.08); }
    .status {
      margin-left: auto;
      color: var(--muted);
      font-size: 13px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 14px;
    }
    .card {
      padding: 16px;
      min-height: 110px;
    }
    .card .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }
    .card .value {
      margin-top: 10px;
      font-size: clamp(24px, 2vw, 34px);
      font-weight: 700;
    }
    .card .sub {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 2.15fr) minmax(340px, 0.95fr);
      gap: 20px;
    }
    .panel {
      padding: 16px;
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
    }
    #chart {
      height: 760px;
      width: 100%;
    }
    .stack {
      display: grid;
      gap: 16px;
    }
    .list {
      display: grid;
      gap: 10px;
      max-height: 230px;
      overflow: auto;
      padding-right: 4px;
    }
    .list-item {
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.03);
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
    }
    .persona-list {
      display: grid;
      gap: 10px;
    }
    .persona-row {
      display: grid;
      grid-template-columns: 92px 1fr 56px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }
    .bar {
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      overflow: hidden;
    }
    .bar > span {
      display: block;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(38,194,129,0.6), rgba(38,194,129,1));
    }
    .macro {
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .macro strong { color: var(--text); }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      color: var(--text);
      font-size: 12px;
    }
    @media (max-width: 1180px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .layout { grid-template-columns: 1fr; }
      #chart { height: 560px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <h1>Nexus Trader Market Simulator</h1>
        <p>This is a market simulator, not an execution bot. It fuses live candles, headline context, public-discussion proxies, persona reactions, branch disagreement, and the trained neural bias into a 5-minute simulation cone.</p>
      </div>
      <div class="pill">Live mode · 5 minute simulation horizon</div>
    </section>

    <section class="controls">
      <div>
        <label for="symbol">Instrument</label><br />
        <select id="symbol">
          <option value="XAUUSD">XAUUSD</option>
          <option value="EURUSD">EURUSD</option>
          <option value="BTCUSD">BTCUSD</option>
        </select>
      </div>
      <button id="simulate">Simulate Next 5 Minutes</button>
      <div id="status" class="status">Loading latest simulation…</div>
    </section>

    <section class="grid">
      <article class="card"><div class="label">Current Price</div><div class="value" id="metric-price">-</div><div class="sub" id="metric-price-change">-</div></article>
      <article class="card"><div class="label">Simulation Confidence</div><div class="value" id="metric-confidence">-</div><div class="sub">Whole-simulation conviction after collapse</div></article>
      <article class="card"><div class="label">Consensus Score</div><div class="value" id="metric-consensus">-</div><div class="sub">Agreement across surviving futures</div></article>
      <article class="card"><div class="label">Uncertainty Width</div><div class="value" id="metric-uncertainty">-</div><div class="sub">Cone spread from branch disagreement</div></article>
      <article class="card"><div class="label">Scenario Bias</div><div class="value" id="metric-bias">-</div><div class="sub" id="metric-dominant-driver">-</div></article>
      <article class="card"><div class="label">Neural Bias</div><div class="value" id="metric-neural">-</div><div class="sub" id="metric-neural-detail">Live model output on latest fused sequence</div></article>
    </section>

    <section class="layout">
      <article class="panel">
        <h2>Simulation Cone</h2>
        <div id="chart"></div>
      </article>

      <div class="stack">
        <article class="panel">
          <h2>Persona Impact</h2>
          <div id="personas" class="persona-list"></div>
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
    const state = { activeSymbol: 'XAUUSD', lastPayload: null };

    function fmtPct(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      return `${(value * 100).toFixed(1)}%`;
    }

    function fmtPrice(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return '-';
      return Number(value).toFixed(2);
    }

    function setStatus(message) {
      document.getElementById('status').textContent = message;
    }

    function renderMetrics(payload) {
      const sim = payload.simulation || {};
      const market = payload.market || {};
      const model = payload.model_prediction || {};

      document.getElementById('metric-price').textContent = fmtPrice(market.current_price);
      const priceChange = Number(market.price_change || 0);
      document.getElementById('metric-price-change').textContent = `${priceChange >= 0 ? '+' : ''}${fmtPrice(priceChange)} vs previous candle`;
      document.getElementById('metric-price-change').style.color = priceChange >= 0 ? 'var(--bull)' : 'var(--bear)';

      document.getElementById('metric-confidence').textContent = fmtPct(sim.overall_confidence);
      document.getElementById('metric-consensus').textContent = fmtPct(sim.consensus_score);
      document.getElementById('metric-uncertainty').textContent = fmtPct(sim.uncertainty_width);
      document.getElementById('metric-bias').textContent = (sim.scenario_bias || '-').toUpperCase();
      document.getElementById('metric-bias').style.color = sim.scenario_bias === 'bullish' ? 'var(--bull)' : 'var(--bear)';
      document.getElementById('metric-dominant-driver').textContent = sim.dominant_driver || 'No dominant driver';

      if (model && typeof model.bullish_probability === 'number') {
        document.getElementById('metric-neural').textContent = fmtPct(model.bullish_probability);
        document.getElementById('metric-neural-detail').textContent = `${(model.signal || 'neutral').toUpperCase()} · threshold ${Number(model.threshold || 0.5).toFixed(3)}`;
        document.getElementById('metric-neural').style.color = model.signal === 'bullish' ? 'var(--bull)' : 'var(--bear)';
      } else {
        document.getElementById('metric-neural').textContent = 'N/A';
        document.getElementById('metric-neural-detail').textContent = 'Live model inference unavailable';
        document.getElementById('metric-neural').style.color = 'var(--text)';
      }
    }

    function renderList(containerId, items, key) {
      const container = document.getElementById(containerId);
      container.innerHTML = '';
      if (!items || !items.length) {
        container.innerHTML = '<div class="list-item">No live items available right now.</div>';
        return;
      }
      for (const item of items) {
        const node = document.createElement('div');
        node.className = 'list-item';
        const title = item.title || item.source || 'Untitled';
        const meta = [item.source, item.published_at, item.classification, item.sentiment !== undefined ? `sentiment ${Number(item.sentiment).toFixed(2)}` : '']
          .filter(Boolean)
          .join(' · ');
        node.innerHTML = item.link
          ? `<a href="${item.link}" target="_blank" rel="noreferrer">${title}</a><div class="meta">${meta}</div>`
          : `<div>${title}</div><div class="meta">${meta}</div>`;
        container.appendChild(node);
      }
    }

    function renderMacro(payload) {
      const macro = (payload.feeds || {}).macro || {};
      const host = document.getElementById('macro');
      const components = macro.components || {};
      const rows = Object.entries(components).map(([name, value]) => {
        return `<div><strong>${name}</strong>: ${Number(value).toFixed(2)}</div>`;
      }).join('');
      host.innerHTML = `
        <div><strong>Driver</strong>: ${macro.driver || 'macro_neutral'}</div>
        <div><strong>Bias</strong>: ${fmtPct((Number(macro.macro_bias || 0) + 1) / 2)}</div>
        <div><strong>Shock</strong>: ${fmtPct(Number(macro.macro_shock || 0))}</div>
        ${rows || '<div>No live macro components available.</div>'}
      `;
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
      for (const [name, valueRaw] of entries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))) {
        const value = Number(valueRaw);
        const width = Math.max(6, Math.round((Math.abs(value) / maxAbs) * 100));
        const row = document.createElement('div');
        row.className = 'persona-row';
        row.innerHTML = `
          <div>${name}</div>
          <div class="bar"><span style="width:${width}%; background:${value >= 0 ? 'linear-gradient(90deg, rgba(46,204,113,0.55), rgba(46,204,113,1))' : 'linear-gradient(90deg, rgba(255,90,95,0.55), rgba(255,90,95,1))'}"></span></div>
          <div style="text-align:right;color:${value >= 0 ? 'var(--bull)' : 'var(--bear)'}">${value.toFixed(3)}</div>
        `;
        host.appendChild(row);
      }
    }

    function renderChart(payload) {
      const candles = (payload.market || {}).candles || [];
      const branches = payload.branches || [];
      const cone = payload.cone || [];

      const candleTrace = {
        type: 'candlestick',
        x: candles.map(row => row.timestamp),
        open: candles.map(row => row.open),
        high: candles.map(row => row.high),
        low: candles.map(row => row.low),
        close: candles.map(row => row.close),
        increasing: { line: { color: '#2ecc71' }, fillcolor: '#2ecc71' },
        decreasing: { line: { color: '#ff5a5f' }, fillcolor: '#ff5a5f' },
        name: 'Market',
      };

      const traces = [candleTrace];
      for (const branch of branches.slice(0, 10)) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: branch.timestamps,
          y: branch.predicted_prices,
          line: { color: 'rgba(84, 160, 255, 0.18)', width: 1.2 },
          hovertemplate: `Branch ${branch.path_id}<br>Price=%{y:.2f}<extra></extra>`,
          showlegend: false,
        });
      }

      if (cone.length) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: cone.map(row => row.timestamp),
          y: cone.map(row => row.lower_price),
          line: { color: 'rgba(46, 204, 113, 0.0)' },
          showlegend: false,
          hoverinfo: 'skip',
        });
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: cone.map(row => row.timestamp),
          y: cone.map(row => row.upper_price),
          line: { color: 'rgba(46, 204, 113, 0.0)' },
          fill: 'tonexty',
          fillcolor: 'rgba(46, 204, 113, 0.18)',
          name: 'Probability Cone',
          hoverinfo: 'skip',
        });
        traces.push({
          type: 'scatter',
          mode: 'lines+markers',
          x: cone.map(row => row.timestamp),
          y: cone.map(row => row.center_price),
          line: { color: '#26c281', width: 3 },
          marker: { size: 5, color: '#26c281' },
          name: 'Consensus Path',
        });
      }

      Plotly.react('chart', traces, {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(8,16,24,0.2)',
        font: { color: '#eef5fb' },
        xaxis: {
          rangeslider: { visible: false },
          gridcolor: 'rgba(255,255,255,0.06)',
          zerolinecolor: 'rgba(255,255,255,0.06)',
        },
        yaxis: {
          gridcolor: 'rgba(255,255,255,0.06)',
          zerolinecolor: 'rgba(255,255,255,0.06)',
          fixedrange: false,
        },
        margin: { l: 36, r: 16, t: 8, b: 34 },
        hovermode: 'x unified',
        showlegend: true,
        legend: { orientation: 'h', y: 1.08, x: 0 },
        uirevision: `${state.activeSymbol}-nexus-live`,
      }, { responsive: true, displaylogo: false });
    }

    async function simulate(symbol, announce = true) {
      state.activeSymbol = symbol;
      if (announce) setStatus(`Simulating ${symbol} with live market, headline, and public-discussion inputs…`);
      const response = await fetch(`/api/simulate-live?symbol=${encodeURIComponent(symbol)}`);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Request failed with ${response.status}`);
      }
      const payload = await response.json();
      state.lastPayload = payload;
      renderMetrics(payload);
      renderChart(payload);
      renderPersonas(payload.personas || {});
      renderMacro(payload);
      renderList('news', (payload.feeds || {}).news || [], 'news');
      renderList('crowd', (payload.feeds || {}).public_discussions || [], 'crowd');
      setStatus(`Simulation ready for ${symbol} · ${payload.generated_at}`);
    }

    document.getElementById('simulate').addEventListener('click', async () => {
      const symbol = document.getElementById('symbol').value;
      try {
        await simulate(symbol, true);
      } catch (error) {
        console.error(error);
        setStatus(`Simulation failed: ${error.message}`);
      }
    });

    window.addEventListener('load', async () => {
      try {
        await simulate(document.getElementById('symbol').value, false);
      } catch (error) {
        console.error(error);
        setStatus(`Initial load failed: ${error.message}`);
      }
    });
  </script>
</body>
</html>
"""
