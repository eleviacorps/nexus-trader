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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
    :root {
      --bg-black: #050505;
      --bg-deep: #080808;
      --bg-soft: #0b0b0b;
      --card-black: rgba(24, 24, 24, 0.55);
      --card-elevated: rgba(32, 32, 32, 0.46);
      --card-dense: rgba(18, 18, 18, 0.72);
      --soft-highlight: rgba(255, 255, 255, 0.08);
      --text: #ffffff;
      --text-secondary: rgba(255, 255, 255, 0.72);
      --text-muted: rgba(255, 255, 255, 0.45);
      --muted: rgba(255, 255, 255, 0.45);
      --border: rgba(255, 255, 255, 0.06);
      --accent-green: #00e38c;
      --accent-red: #ff4d57;
      --accent-blue: #5ba7ff;
      --accent-amber: #ffc857;
      --bull: #00e38c;
      --bear: #ff4d57;
      --accent: #5ba7ff;
      --deep-wine: #2a0008;
      --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.05);
      --shadow-soft: 18px 18px 36px rgba(0, 0, 0, 0.36), -8px -8px 18px rgba(255, 255, 255, 0.02);
    }
    * { box-sizing: border-box; }
    html { color-scheme: dark; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Inter", system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(120, 0, 0, 0.18), transparent 40%),
        radial-gradient(circle at bottom right, rgba(255, 255, 255, 0.03), transparent 35%),
        radial-gradient(circle at 12% 14%, rgba(86, 0, 15, 0.18), transparent 18%),
        radial-gradient(circle at 88% 18%, rgba(255, 77, 87, 0.06), transparent 20%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.02), transparent 45%),
        #050505;
      position: relative;
      overflow-x: hidden;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        radial-gradient(circle at center, transparent 60%, rgba(0, 0, 0, 0.38) 100%),
        repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0 1px, transparent 1px 3px);
      opacity: 0.55;
      mix-blend-mode: screen;
    }
    .shell {
      position: relative;
      z-index: 1;
      max-width: 1880px;
      margin: 0 auto;
      padding: 28px 26px 36px;
      display: grid;
      gap: 18px;
    }
    .status-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      flex-wrap: wrap;
    }
    .header-time {
      display: grid;
      gap: 6px;
    }
    .header-time-value {
      font-family: "JetBrains Mono", monospace;
      font-size: clamp(56px, 7vw, 72px);
      line-height: 0.9;
      letter-spacing: -0.04em;
      text-shadow: 0 8px 32px rgba(255, 77, 87, 0.12);
    }
    .header-time-subtitle {
      color: var(--text-secondary);
      font-size: 14px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .header-copy {
      color: var(--text-muted);
      max-width: 720px;
      line-height: 1.6;
      font-size: 14px;
    }
    .header-pills {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
      max-width: 860px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 38px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.06);
      background: rgba(24, 24, 24, 0.42);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 8px 24px rgba(0,0,0,0.22);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      color: var(--text-secondary);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-family: "JetBrains Mono", monospace;
    }
    .hero-grid,
    .grid,
    .main-grid,
    .analysis-grid,
    .intel-grid,
    .graph-grid,
    .footer-grid {
      display: grid;
      gap: 18px;
      align-items: stretch;
    }
    .hero-grid {
      grid-template-columns: 1.2fr 1.1fr;
    }
    .panel,
    .card,
    .controls,
    .hero-card {
      position: relative;
      min-width: 0;
      overflow: hidden;
      border-radius: 28px;
      border: 1px solid rgba(255, 255, 255, 0.06);
      background: rgba(28, 28, 28, 0.42);
      backdrop-filter: blur(28px);
      -webkit-backdrop-filter: blur(28px);
      box-shadow: var(--shadow-lg), var(--shadow-soft);
    }
    .panel::before,
    .card::before,
    .controls::before,
    .hero-card::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(135deg, rgba(255,255,255,0.06), transparent 32%),
        radial-gradient(circle at top right, rgba(255, 77, 87, 0.08), transparent 36%);
    }
    .hero-card {
      padding: 24px;
      min-height: 240px;
      display: grid;
      gap: 18px;
    }
    .hero-card-head {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    .hero-copy {
      margin-top: 10px;
      color: var(--text-muted);
      font-size: 13px;
      line-height: 1.55;
      max-width: 42ch;
    }
    .hero-card h1,
    .hero-card h2,
    .panel h2 {
      margin: 0;
      color: var(--text-secondary);
      font-size: 12px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-family: "JetBrains Mono", monospace;
    }
    .hero-signal {
      font-size: clamp(34px, 4vw, 54px);
      line-height: 0.95;
      font-weight: 800;
      letter-spacing: -0.04em;
    }
    .hero-metric {
      display: grid;
      gap: 6px;
    }
    .hero-metric .value {
      margin: 0;
      font-size: clamp(38px, 4vw, 58px);
      line-height: 0.94;
      letter-spacing: -0.05em;
      font-family: "JetBrains Mono", monospace;
    }
    .hero-strip {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .hero-chip {
      padding: 14px 16px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255,255,255,0.05);
      box-shadow: inset 6px 6px 14px rgba(0,0,0,0.22), inset -4px -4px 12px rgba(255,255,255,0.02);
    }
    .controls {
      display: flex;
      align-items: flex-end;
      gap: 14px;
      flex-wrap: wrap;
      padding: 18px;
      position: sticky;
      top: 10px;
      z-index: 10;
    }
    .control {
      display: grid;
      gap: 8px;
      min-width: 142px;
    }
    label {
      color: var(--text-muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-family: "JetBrains Mono", monospace;
    }
    select, button, input[type="checkbox"] {
      accent-color: var(--accent-green);
    }
    select, button {
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      background:
        linear-gradient(145deg, rgba(37,37,37,0.78), rgba(20,20,20,0.72));
      color: var(--text);
      padding: 13px 16px;
      font-size: 14px;
      box-shadow: inset 1px 1px 0 rgba(255,255,255,0.05), inset -8px -8px 16px rgba(0,0,0,0.26);
      font-family: "Inter", sans-serif;
    }
    select:focus,
    button:focus {
      outline: 0;
      border-color: rgba(91,167,255,0.34);
      box-shadow: 0 0 0 1px rgba(91,167,255,0.18), 0 12px 28px rgba(91,167,255,0.08);
    }
    button {
      cursor: pointer;
      font-weight: 700;
      min-width: 220px;
      background:
        linear-gradient(145deg, rgba(255, 77, 87, 0.16), rgba(18,18,18,0.78)),
        linear-gradient(135deg, rgba(0,227,140,0.18), rgba(91,167,255,0.08));
    }
    button:hover { filter: brightness(1.08); }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--text-secondary);
      font-size: 13px;
      margin-bottom: 2px;
    }
    .status {
      margin-left: auto;
      min-width: 280px;
      color: var(--text-secondary);
      font-size: 13px;
      line-height: 1.45;
      text-align: right;
    }
    .grid {
      grid-template-columns: repeat(6, minmax(0, 1fr));
    }
    .card {
      padding: 18px;
      min-height: 136px;
    }
    .label {
      color: var(--text-muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-family: "JetBrains Mono", monospace;
    }
    .value {
      margin-top: 12px;
      font-size: clamp(24px, 2vw, 36px);
      font-weight: 800;
      line-height: 1;
      letter-spacing: -0.04em;
      font-family: "JetBrains Mono", monospace;
    }
    .sub {
      margin-top: 10px;
      color: var(--text-muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .main-grid {
      grid-template-columns: minmax(0, 1.3fr) minmax(0, 1.2fr) minmax(360px, 0.95fr);
    }
    .analysis-grid {
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      grid-auto-rows: minmax(320px, auto);
    }
    .intel-grid {
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      grid-auto-rows: minmax(280px, auto);
    }
    .graph-grid {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(360px, 0.9fr);
    }
    .footer-grid {
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      grid-auto-rows: minmax(230px, auto);
    }
    .workspace-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      padding: 18px;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .panel .caption {
      margin: 8px 0 14px;
      color: var(--text-muted);
      font-size: 12px;
      line-height: 1.55;
      max-width: 90%;
    }
    #live-chart, #compare-chart {
      height: 640px;
      width: 100%;
    }
    #swarm-graph, #branch-graph-view, #tradingview-frame {
      height: 420px;
      width: 100%;
    }
    #tradingview-frame {
      height: 640px;
      flex: 1 1 auto;
    }
    .list,
    .persona-list,
    .macro,
    .history-list,
    .conversation-list,
    .tilt-list,
    .ta-grid,
    .bot-list,
    .reaction-list {
      display: grid;
      gap: 10px;
    }
    .list {
      max-height: 340px;
      overflow: auto;
      padding-right: 4px;
    }
    #swarm-judge,
    #conversation,
    #specialist-bots,
    #public-reaction,
    #technical-structure,
    #forecast-ladder,
    #order-blocks,
    #fair-value-gaps,
    #news,
    #crowd,
    #personas,
    #tilts,
    #macro,
    #history {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      padding-right: 4px;
    }
    #technical-structure,
    #forecast-ladder,
    #order-blocks,
    #fair-value-gaps,
    #personas,
    #tilts,
    #macro,
    #history {
      max-height: 100%;
    }
    * {
      scrollbar-width: thin;
      scrollbar-color: rgba(255,255,255,0.24) rgba(255,255,255,0.04);
    }
    *::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    *::-webkit-scrollbar-track {
      background: rgba(255,255,255,0.04);
      border-radius: 999px;
    }
    *::-webkit-scrollbar-thumb {
      background: linear-gradient(180deg, rgba(255,255,255,0.22), rgba(255,77,87,0.22));
      border-radius: 999px;
      border: 2px solid rgba(5,5,5,0.3);
    }
    *::-webkit-scrollbar-thumb:hover {
      background: linear-gradient(180deg, rgba(255,255,255,0.30), rgba(255,77,87,0.34));
    }
    .list-item,
    .history-item,
    .bot-item,
    .hero-chip {
      color: var(--text-secondary);
      line-height: 1.5;
    }
    .list-item,
    .history-item,
    .bot-item {
      padding: 13px;
      border-radius: 20px;
      border: 1px solid rgba(255,255,255,0.05);
      background: rgba(255,255,255,0.025);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), inset 0 -12px 20px rgba(0,0,0,0.14);
      font-size: 12px;
    }
    .list-item a {
      color: var(--text);
      text-decoration: none;
      font-weight: 600;
    }
    .meta {
      margin-top: 7px;
      color: var(--text-muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .judge-banner {
      display: grid;
      gap: 12px;
      padding: 16px;
      border-radius: 24px;
      border: 1px solid rgba(255,255,255,0.07);
      background:
        linear-gradient(145deg, rgba(42, 0, 8, 0.38), rgba(18, 18, 18, 0.84)),
        rgba(28,28,28,0.42);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 18px 36px rgba(0,0,0,0.22);
    }
    .judge-headline {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .judge-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      border: 1px solid rgba(255,255,255,0.08);
      font-family: "JetBrains Mono", monospace;
    }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .mono {
      font-family: "JetBrains Mono", monospace;
    }
    .tv-frame {
      border: 0;
      border-radius: 22px;
      background: rgba(9,9,9,0.92);
      width: 100%;
      height: 420px;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.05);
    }
    .workspace-shell {
      min-height: 620px;
    }
    .workspace-head {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .workspace-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .workspace-tab {
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255,255,255,0.03);
      color: var(--text-secondary);
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-family: "JetBrains Mono", monospace;
      cursor: pointer;
      user-select: none;
    }
    .workspace-tab.active {
      background: linear-gradient(135deg, rgba(255,77,87,0.18), rgba(91,167,255,0.14));
      color: var(--text);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 8px 18px rgba(0,0,0,0.22);
    }
    .workspace-body {
      position: relative;
      flex: 1 1 auto;
      min-height: 0;
    }
    .workspace-panel {
      display: none;
      height: 100%;
      min-height: 0;
      flex-direction: column;
      gap: 10px;
    }
    .workspace-panel.active {
      display: flex;
    }
    .workspace-panel > h3 {
      margin: 0;
      color: var(--text-secondary);
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      font-family: "JetBrains Mono", monospace;
    }
    .workspace-panel > .caption {
      margin: 0 0 8px;
      color: var(--text-muted);
      font-size: 12px;
      line-height: 1.55;
    }
    .workspace-panel .plot-host,
    .workspace-panel .content-host {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      padding-right: 4px;
    }
    .bot-item strong,
    .history-item strong {
      color: var(--text);
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
      box-shadow: inset 0 0 8px rgba(0,0,0,0.34);
    }
    .bar > span {
      display: block;
      height: 100%;
      border-radius: 999px;
      box-shadow: 0 0 20px rgba(0, 227, 140, 0.16);
    }
    .js-plotly-plot .plotly .modebar {
      background: rgba(18, 18, 18, 0.78) !important;
      backdrop-filter: blur(14px);
      border-radius: 14px;
      padding: 4px;
      border: 1px solid rgba(255,255,255,0.06);
    }
    .js-plotly-plot .plotly .modebar-btn svg {
      fill: #f5f5f5 !important;
    }
    .js-plotly-plot .plotly .hoverlayer .hovertext rect {
      fill: #101010 !important;
      stroke: rgba(255,255,255,0.12) !important;
    }
    .js-plotly-plot .plotly .hoverlayer .hovertext text {
      fill: #f3f3f3 !important;
    }
    @media (max-width: 1600px) {
      .grid { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
      .hero-grid,
      .main-grid,
      .analysis-grid,
      .intel-grid,
      .graph-grid,
      .footer-grid,
      .workspace-grid {
        grid-template-columns: 1fr;
      }
      #live-chart,
      #compare-chart {
        height: 560px;
      }
      .status {
        margin-left: 0;
        text-align: left;
      }
    }
    @media (max-width: 960px) {
      .shell {
        padding: 18px 14px 24px;
      }
      .grid { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
      .hero-strip,
      .mini-grid {
        grid-template-columns: 1fr;
      }
      .header-pills {
        justify-content: flex-start;
      }
      .controls {
        padding: 16px;
        position: static;
      }
      .workspace-shell {
        min-height: 540px;
      }
    }
    @media (max-width: 680px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .header-time-value {
        font-size: 48px;
      }
      .status-header {
        gap: 12px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="status-header">
      <div class="header-time">
        <div id="header-time" class="header-time-value">--:--</div>
        <div id="header-date" class="header-time-subtitle">Loading local trading desk</div>
        <div class="header-copy">Dark glass manual-trading terminal. V8 runs as the direct simulator path generator here, LM Studio remains available as the local narrative sidecar, and the operator decides execution manually from the projected path.</div>
      </div>
      <div class="header-pills">
        <div id="header-session" class="pill">Session: --</div>
        <div id="header-market-status" class="pill">Market: --</div>
        <div id="header-model" class="pill">Model: V8 Direct</div>
        <div id="header-latency" class="pill">Latency: --</div>
        <div id="header-live-mode" class="pill">Mode: Manual</div>
        <div id="header-llm" class="pill">LLM: LM Studio</div>
      </div>
    </section>

    <section class="hero-grid">
      <article class="hero-card">
        <div class="hero-card-head">
          <div>
            <h2>Portfolio Summary</h2>
            <div class="hero-copy">Manual operator view for projected futures, recent signal quality, and live regime context.</div>
          </div>
          <div class="pill">Local Terminal</div>
        </div>
        <div class="hero-metric">
          <div class="label">Current XAUUSD</div>
          <div id="metric-price" class="value">-</div>
          <div id="metric-price-change" class="sub">-</div>
        </div>
        <div class="hero-strip">
          <div class="hero-chip">
            <div class="label">Filtered Hit Rate</div>
            <div id="metric-hit-rate" class="value">-</div>
            <div id="metric-direction-match" class="sub">Waiting for realized candles</div>
          </div>
          <div class="hero-chip">
            <div class="label">Simulation Confidence</div>
            <div id="metric-confidence" class="value">-</div>
            <div id="metric-live-time" class="sub">-</div>
          </div>
          <div class="hero-chip">
            <div class="label">Timestamp</div>
            <div id="metric-times" class="value">-</div>
            <div class="sub">Last frozen future</div>
          </div>
        </div>
      </article>

      <article class="hero-card">
        <div class="hero-card-head">
          <div>
            <h2>AI Signal Status</h2>
            <div class="hero-copy">Direct V8 output without external gate enforcement. This panel is for manual action judgment, not auto-execution.</div>
          </div>
          <div id="stack-badge" class="pill">V8 Direct</div>
        </div>
        <div id="metric-bias" class="hero-signal">-</div>
        <div id="metric-dominant-driver" class="sub">Waiting for current regime read</div>
        <div class="hero-strip">
          <div class="hero-chip">
            <div class="label">V8 Neural Bias</div>
            <div id="metric-neural" class="value">-</div>
            <div id="metric-neural-detail" class="sub">Latest routed predictor</div>
          </div>
          <div class="hero-chip">
            <div class="label">Final Desk View</div>
            <div id="metric-ensemble" class="value">-</div>
            <div id="metric-ensemble-detail" class="sub">Manual-mode projected path</div>
          </div>
          <div class="hero-chip">
            <div class="label">Bot Swarm</div>
            <div id="metric-bots" class="value">-</div>
            <div id="metric-bots-detail" class="sub">Specialist alignment</div>
          </div>
        </div>
      </article>
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
      <div class="control">
        <label for="forecast-horizon">Forecast Horizon</label>
        <select id="forecast-horizon">
          <option value="5">5 min</option>
          <option value="10">10 min</option>
          <option value="15" selected>15 min</option>
          <option value="30">30 min</option>
        </select>
      </div>
      <div class="control">
        <label for="llm-provider">LLM Route</label>
        <select id="llm-provider">
          <option value="lm_studio" selected>LM Studio Local</option>
          <option value="ollama">Ollama Cloud</option>
        </select>
      </div>
      <div class="control">
        <label for="signal-mode">Signal Stack</label>
        <select id="signal-mode">
          <option value="v8_direct" selected>V8 Direct Manual</option>
          <option value="hybrid">Hybrid Ensemble</option>
        </select>
      </div>
      <button id="simulate">Freeze Next Projection</button>
      <label class="toggle"><input type="checkbox" id="auto-refresh" checked /> Auto refresh live market</label>
      <div id="status" class="status">Loading local V8 terminal...</div>
    </section>

    <section class="grid">
      <article class="card"><div class="label">Consensus Score</div><div class="value" id="metric-consensus">-</div><div class="sub">Agreement across surviving futures</div></article>
      <article class="card"><div class="label">Uncertainty Width</div><div class="value" id="metric-uncertainty">-</div><div class="sub">Cone spread from branch disagreement</div></article>
      <article class="card"><div class="label">Manual Trade Mode</div><div class="value" id="metric-stack-mode">V8</div><div class="sub" id="metric-stack-mode-detail">Direct path selection without external gate veto</div></article>
      <article class="card"><div class="label">Judge Stance</div><div class="value" id="metric-judge-stance">-</div><div class="sub" id="metric-judge-detail">GPT sidecar narrative</div></article>
      <article class="card"><div class="label">Forecast Horizon</div><div class="value" id="metric-horizon-focus">15m</div><div class="sub">Live compare chart focus</div></article>
      <article class="card"><div class="label">Local Route</div><div class="value" id="metric-llm-route">LM Studio</div><div class="sub">Operator-controlled sidecar route</div></article>
    </section>

    <section class="main-grid">
      <article class="panel">
        <h2>Live Market Tape</h2>
        <div class="caption">TradingView now anchors the live tape so you get a richer charting surface with better navigation, zooming, and manual tape reading.</div>
        <iframe id="tradingview-frame" class="tv-frame" title="TradingView Live Market Tape"></iframe>
      </article>

      <article class="panel">
        <h2>Predicted vs Actual Future</h2>
        <div class="caption">The frozen V8 path is compared against arriving candles after the trigger timestamp so you can inspect path realism instead of raw classifier output.</div>
        <div id="compare-chart"></div>
      </article>

      <article class="panel">
        <h2>Final Judge Desk</h2>
        <div class="caption">Narrative explanation, trade stance, and dissent framing for manual execution review.</div>
        <div id="swarm-judge" class="conversation-list"></div>
      </article>
    </section>

    <section class="workspace-grid">
      <article class="panel workspace-shell">
        <div class="workspace-head">
          <div>
            <h2>Signal Workspace</h2>
            <div class="caption">Switch between dense signal panels instead of keeping every intelligence tile visible at once.</div>
          </div>
          <div class="workspace-tabs" data-workspace="signal">
            <button class="workspace-tab active" data-target="ws-conversation">Conversation</button>
            <button class="workspace-tab" data-target="ws-bots">Bots</button>
            <button class="workspace-tab" data-target="ws-reaction">Reaction</button>
            <button class="workspace-tab" data-target="ws-structure">Structure</button>
            <button class="workspace-tab" data-target="ws-ladder">Forecast</button>
            <button class="workspace-tab" data-target="ws-order-blocks">Order Blocks</button>
            <button class="workspace-tab" data-target="ws-fvg">FVG</button>
            <button class="workspace-tab" data-target="ws-news">News</button>
            <button class="workspace-tab" data-target="ws-crowd">Crowd</button>
          </div>
        </div>
        <div class="workspace-body">
          <section id="ws-conversation" class="workspace-panel active">
            <h3>Branch Conversation</h3>
            <div class="content-host">
              <div id="conversation" class="conversation-list"></div>
            </div>
          </section>
          <section id="ws-bots" class="workspace-panel">
            <h3>Specialist Bots</h3>
            <div class="content-host">
              <div id="specialist-bots" class="bot-list"></div>
            </div>
          </section>
          <section id="ws-reaction" class="workspace-panel">
            <h3>Public Reaction Theater</h3>
            <div class="caption">How the crowd might be reacting, summarized through the judge and the raw feeds.</div>
            <div class="content-host">
              <div id="public-reaction" class="reaction-list"></div>
            </div>
          </section>
          <section id="ws-structure" class="workspace-panel">
            <h3>Technical Structure</h3>
            <div class="content-host">
              <div id="technical-structure" class="ta-grid"></div>
            </div>
          </section>
          <section id="ws-ladder" class="workspace-panel">
            <h3>Forecast Ladder</h3>
            <div class="content-host">
              <div id="forecast-ladder" class="history-list"></div>
            </div>
          </section>
          <section id="ws-order-blocks" class="workspace-panel">
            <h3>Order Blocks</h3>
            <div class="content-host">
              <div id="order-blocks" class="history-list"></div>
            </div>
          </section>
          <section id="ws-fvg" class="workspace-panel">
            <h3>Fair Value Gaps</h3>
            <div class="content-host">
              <div id="fair-value-gaps" class="history-list"></div>
            </div>
          </section>
          <section id="ws-news" class="workspace-panel">
            <h3>Headline Context</h3>
            <div class="content-host">
              <div id="news" class="list"></div>
            </div>
          </section>
          <section id="ws-crowd" class="workspace-panel">
            <h3>Public Discussion Pulse</h3>
            <div class="content-host">
              <div id="crowd" class="list"></div>
            </div>
          </section>
        </div>
      </article>

      <article class="panel workspace-shell">
        <div class="workspace-head">
          <div>
            <h2>Research Workspace</h2>
            <div class="caption">Graphs, local tape, personas, macro state, and history live here as switchable operator views.</div>
          </div>
          <div class="workspace-tabs" data-workspace="research">
            <button class="workspace-tab active" data-target="ws-swarm-graph">Swarm</button>
            <button class="workspace-tab" data-target="ws-branch-graph">Branches</button>
            <button class="workspace-tab" data-target="ws-local-tape">Local Tape</button>
            <button class="workspace-tab" data-target="ws-personas">Personas</button>
            <button class="workspace-tab" data-target="ws-tilts">Tilts</button>
            <button class="workspace-tab" data-target="ws-macro">Macro</button>
            <button class="workspace-tab" data-target="ws-history">History</button>
          </div>
        </div>
        <div class="workspace-body">
          <section id="ws-swarm-graph" class="workspace-panel active">
            <h3>Swarm Influence Graph</h3>
            <div class="caption">Bots, personas, and the simulator laid out as an influence network.</div>
            <div class="plot-host">
              <div id="swarm-graph"></div>
            </div>
          </section>
          <section id="ws-branch-graph" class="workspace-panel">
            <h3>Branch Graph View</h3>
            <div class="caption">Top branch, minority branch, and supporting paths in a branch-oriented view.</div>
            <div class="plot-host">
              <div id="branch-graph-view"></div>
            </div>
          </section>
          <section id="ws-local-tape" class="workspace-panel">
            <h3>Local Simulation Tape</h3>
            <div class="caption">Native local tape remains available here if you want to compare it against TradingView’s live chart desk.</div>
            <div class="plot-host">
              <div id="live-chart"></div>
            </div>
          </section>
          <section id="ws-personas" class="workspace-panel">
            <h3>Persona Impact</h3>
            <div class="content-host">
              <div id="personas" class="persona-list"></div>
            </div>
          </section>
          <section id="ws-tilts" class="workspace-panel">
            <h3>LLM Persona Tilts</h3>
            <div class="content-host">
              <div id="tilts" class="tilt-list"></div>
            </div>
          </section>
          <section id="ws-macro" class="workspace-panel">
            <h3>Macro Pulse</h3>
            <div class="content-host">
              <div id="macro" class="macro"></div>
            </div>
          </section>
          <section id="ws-history" class="workspace-panel">
            <h3>Recent Simulation History</h3>
            <div class="content-host">
              <div id="history" class="history-list"></div>
            </div>
          </section>
        </div>
      </article>
    </section>
  </div>

  <script>
    const state = {
      activeSymbol: 'XAUUSD',
      activeHorizon: 15,
      llmProvider: 'lm_studio',
      stackMode: 'v8_direct',
      lastSimulation: null,
      lastMonitor: null,
      refreshTimer: null,
      lastLatencyMs: null,
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

    function localSessionLabel(date = new Date()) {
      const hour = date.getHours();
      if (hour >= 6 && hour < 13) return 'Asia Session';
      if (hour >= 13 && hour < 18) return 'London Session';
      if (hour >= 18 && hour < 23) return 'New York Session';
      return 'Overnight Session';
    }

    function updateHeaderShell() {
      const now = new Date();
      document.getElementById('header-time').textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      document.getElementById('header-date').textContent = `${now.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' })} • ${localSessionLabel(now)}`;
      document.getElementById('header-session').textContent = `Session: ${localSessionLabel(now)}`;
      document.getElementById('header-market-status').textContent = `Market: ${state.activeSymbol || 'XAUUSD'} Live`;
      document.getElementById('header-model').textContent = `Model: ${state.stackMode === 'v8_direct' ? 'V8 Direct' : 'Hybrid Ensemble'}`;
      document.getElementById('header-latency').textContent = `Latency: ${state.lastLatencyMs === null ? '--' : `${state.lastLatencyMs} ms`}`;
      document.getElementById('header-live-mode').textContent = `Mode: ${state.stackMode === 'v8_direct' ? 'Manual' : 'Hybrid'}`;
      document.getElementById('header-llm').textContent = `LLM: ${state.llmProvider === 'lm_studio' ? 'LM Studio' : 'Ollama'}`;
    }

    function colorForSignal(signal) {
      if (signal === 'bullish' || signal === 'buy') return 'var(--accent-green)';
      if (signal === 'bearish' || signal === 'sell') return 'var(--accent-red)';
      return 'var(--accent-amber)';
    }

    function renderMetrics(simPayload, monitorPayload) {
      const sim = (simPayload || {}).simulation || {};
      const market = (simPayload || {}).market || {};
      const model = (simPayload || {}).model_prediction || {};
      const ensemble = (simPayload || {}).ensemble_prediction || {};
      const judge = (((simPayload || {}).swarm_judge || {}).content || {});
      const activePrediction = (monitorPayload || {}).active_prediction || {};
      const stackMode = String((simPayload || {}).stack_mode || state.stackMode || 'v8_direct');

      updateHeaderShell();

      document.getElementById('metric-price').textContent = fmtPrice(market.current_price);
      const change = Number(market.price_change || 0);
      document.getElementById('metric-price-change').textContent = `${change >= 0 ? '+' : ''}${fmtPrice(change)} vs previous candle`;
      document.getElementById('metric-price-change').style.color = change >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';

      document.getElementById('metric-confidence').textContent = fmtPct(sim.overall_confidence);
      document.getElementById('metric-consensus').textContent = fmtPct(sim.consensus_score);
      document.getElementById('metric-uncertainty').textContent = fmtPct(sim.uncertainty_width);
      document.getElementById('metric-bias').textContent = (sim.scenario_bias || '-').toUpperCase();
      document.getElementById('metric-bias').style.color = colorForSignal(sim.scenario_bias);
      document.getElementById('metric-dominant-driver').textContent = sim.dominant_driver || 'No dominant driver';
      document.getElementById('metric-stack-mode').textContent = stackMode === 'v8_direct' ? 'V8 DIRECT' : 'HYBRID';
      document.getElementById('metric-stack-mode-detail').textContent = stackMode === 'v8_direct'
        ? 'Manual desk mode with direct V8 path projection'
        : 'Hybrid ensemble with external selector stack';
      document.getElementById('metric-judge-stance').textContent = String(judge.manual_stance || 'hold').toUpperCase();
      document.getElementById('metric-judge-stance').style.color = colorForSignal(String(judge.manual_stance || '').toLowerCase());
      document.getElementById('metric-judge-detail').textContent = judge.summary || judge.discussion_takeaway || 'Judge summary waiting';
      document.getElementById('metric-horizon-focus').textContent = `${state.activeHorizon}m`;
      document.getElementById('metric-llm-route').textContent = state.llmProvider === 'lm_studio' ? 'LM STUDIO' : 'OLLAMA';

      if (model && typeof model.bullish_probability === 'number') {
        document.getElementById('metric-neural').textContent = fmtPct(model.bullish_probability);
        document.getElementById('metric-neural').style.color = colorForSignal(model.signal);
        document.getElementById('metric-neural-detail').textContent = `${(model.signal || 'neutral').toUpperCase()} | threshold ${Number(model.threshold || 0.5).toFixed(3)}`;
      } else {
        document.getElementById('metric-neural').textContent = 'N/A';
        document.getElementById('metric-neural').style.color = 'var(--text)';
        document.getElementById('metric-neural-detail').textContent = 'Live model inference unavailable';
      }

      if (ensemble && typeof ensemble.bullish_probability === 'number') {
        document.getElementById('metric-ensemble').textContent = fmtPct(ensemble.bullish_probability);
        document.getElementById('metric-ensemble').style.color = colorForSignal(ensemble.signal);
        const horizon = (ensemble.horizon_predictions || [])[0];
        document.getElementById('metric-ensemble-detail').textContent = horizon
          ? `${(ensemble.signal || 'neutral').toUpperCase()} | 5m ${fmtPrice(horizon.target_price)} | judge ${String(judge.manual_stance || 'hold').toUpperCase()}`
          : `${(ensemble.signal || 'neutral').toUpperCase()} | confidence ${fmtPct(ensemble.confidence)}`;
      } else {
        document.getElementById('metric-ensemble').textContent = 'N/A';
        document.getElementById('metric-ensemble').style.color = 'var(--text)';
        document.getElementById('metric-ensemble-detail').textContent = 'No ensemble output';
      }

      const botAggregate = (simPayload || {}).bot_swarm?.aggregate || {};
      if (typeof botAggregate.bullish_probability === 'number') {
        document.getElementById('metric-bots').textContent = fmtPct(botAggregate.bullish_probability);
        document.getElementById('metric-bots').style.color = colorForSignal(botAggregate.signal);
        document.getElementById('metric-bots-detail').textContent = `${String(botAggregate.signal || 'neutral').toUpperCase()} | disagreement ${fmtPct(botAggregate.disagreement)}`;
      } else {
        document.getElementById('metric-bots').textContent = 'N/A';
        document.getElementById('metric-bots').style.color = 'var(--text)';
        document.getElementById('metric-bots-detail').textContent = 'No bot swarm output';
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
      document.getElementById('stack-badge').textContent = stackMode === 'v8_direct' ? 'V8 Direct' : 'Hybrid Ensemble';
      updateHeaderShell();
    }

    function renderTilts(tilts) {
      const host = document.getElementById('tilts');
      host.innerHTML = '';
      const entries = Object.entries(tilts || {});
      if (!entries.length) {
        host.innerHTML = '<div class="history-item">No LLM persona tilts available.</div>';
        return;
      }
      for (const [name, value] of entries) {
        const node = document.createElement('div');
        node.className = 'history-item';
        node.innerHTML = `<div><strong>${name}</strong>: ${Number(value).toFixed(4)}</div>`;
        host.appendChild(node);
      }
    }

    function renderSpecialistBots(payload) {
      const host = document.getElementById('specialist-bots');
      host.innerHTML = '';
      const bots = (payload.bot_swarm || {}).bots || [];
      if (!bots.length) {
        host.innerHTML = '<div class="history-item">No specialist bot outputs available.</div>';
        return;
      }
      for (const bot of bots) {
        const five = (bot.horizons || []).find(item => item.minutes === 5);
        const ten = (bot.horizons || []).find(item => item.minutes === 10);
        const fifteen = (bot.horizons || []).find(item => item.minutes === 15);
        const node = document.createElement('div');
        node.className = 'bot-item';
        node.innerHTML = `
          <div><strong>${bot.name}</strong> | ${String(bot.direction || 'neutral').toUpperCase()} | conf ${fmtPct(bot.confidence)}</div>
          <div>5m ${five ? fmtPrice(five.target_price) : '-'} | 10m ${ten ? fmtPrice(ten.target_price) : '-'} | 15m ${fifteen ? fmtPrice(fifteen.target_price) : '-'}</div>
          <div>Key ${fmtPrice(bot.key_level)} | Invalid ${fmtPrice(bot.invalidation)} | emotion ${bot.emotion || '-'}</div>
          <div>${bot.reason || ''}</div>
        `;
        host.appendChild(node);
      }
    }

    function renderSwarmJudge(payload) {
      const host = document.getElementById('swarm-judge');
      host.innerHTML = '';
      const judge = ((payload || {}).swarm_judge || {}).content || {};
      if (!Object.keys(judge).length) {
        host.innerHTML = '<div class="history-item">No GPT judge output available.</div>';
        return;
      }
      const summary = document.createElement('div');
      summary.className = 'judge-banner';
      const stance = String(judge.manual_stance || 'hold').toUpperCase();
      const stanceColor = stance === 'BUY' ? 'rgba(46,204,113,0.18)' : stance === 'SELL' ? 'rgba(255,90,95,0.18)' : 'rgba(241,196,15,0.16)';
      summary.innerHTML = `
        <div class="judge-headline">
          <div class="judge-pill" style="background:${stanceColor}; color:${stance === 'SELL' ? '#ff9da1' : stance === 'BUY' ? '#7ef0ac' : '#f7d774'};">${stance}</div>
          <div class="judge-pill mono">Bias ${String(judge.master_bias || '-').toUpperCase()} | conf ${fmtPct(judge.master_confidence)}</div>
        </div>
        <div>${judge.judge_summary || '-'}</div>
        <div class="mini-grid">
          <div class="history-item"><strong>Reason</strong><div>${judge.manual_action_reason || '-'}</div></div>
          <div class="history-item"><strong>Structure</strong><div>${judge.actionable_structure || '-'}</div></div>
          <div class="history-item"><strong>Crowd Lean</strong><div>${judge.crowd_lean || '-'}</div></div>
          <div class="history-item"><strong>Emotion</strong><div>${judge.crowd_emotion || '-'}</div></div>
          <div class="history-item"><strong>Top Bot</strong><div>${judge.top_bot || '-'}</div></div>
          <div class="history-item"><strong>Weakest Bot</strong><div>${judge.weakest_bot || '-'}</div></div>
        </div>
        <div><strong>Minority Case</strong>: ${judge.minority_case || '-'}</div>
      `;
      host.appendChild(summary);
      for (const line of (judge.debate_lines || [])) {
        const node = document.createElement('div');
        node.className = 'history-item';
        node.textContent = line;
        host.appendChild(node);
      }
    }

    function renderForecastLadder(payload) {
      const host = document.getElementById('forecast-ladder');
      host.innerHTML = '';
      const rows = ((payload || {}).final_forecast || {}).horizon_table || [];
      if (!rows.length) {
        host.innerHTML = '<div class="history-item">No multi-horizon forecast available.</div>';
        return;
      }
      for (const row of rows) {
        const node = document.createElement('div');
        node.className = 'history-item';
        node.innerHTML = `
          <div><strong>${row.minutes}m</strong> final ${fmtPrice(row.final_price)}</div>
          <div>sim ${fmtPrice(row.branch_center)} | bots ${fmtPrice(row.bot_target)} | GPT ${fmtPrice(row.llm_target)}</div>
        `;
        host.appendChild(node);
      }
    }

    function renderPublicReaction(payload) {
      const host = document.getElementById('public-reaction');
      host.innerHTML = '';
      const judge = (((payload || {}).swarm_judge || {}).content || {});
      const discussions = (((payload || {}).feeds || {}).public_discussions || []).slice(0, 4);
      const takeaway = document.createElement('div');
      takeaway.className = 'history-item';
      takeaway.innerHTML = `
        <div><strong>Discussion Takeaway</strong></div>
        <div>${judge.discussion_takeaway || 'No judge synthesis available.'}</div>
      `;
      host.appendChild(takeaway);
      for (const line of (judge.public_reaction_lines || [])) {
        const node = document.createElement('div');
        node.className = 'history-item';
        node.textContent = line;
        host.appendChild(node);
      }
      for (const item of discussions) {
        const node = document.createElement('div');
        node.className = 'history-item';
        node.innerHTML = `<strong>${item.classification || 'discussion'}</strong><div>${item.title || '-'}</div><div class="meta">${item.source || ''} | ${fmtTime(item.published_at)}</div>`;
        host.appendChild(node);
      }
    }

    function renderTechnical(payload) {
      const technical = payload.technical_analysis || {};
      const structureHost = document.getElementById('technical-structure');
      const orderBlocksHost = document.getElementById('order-blocks');
      const fvgHost = document.getElementById('fair-value-gaps');

      structureHost.innerHTML = `
        <div class="history-item"><strong>Session</strong><div>${technical.session || '-'}</div></div>
        <div class="history-item"><strong>Structure</strong><div>${String(technical.structure || '-').toUpperCase()}</div></div>
        <div class="history-item"><strong>Location</strong><div>${String(technical.location || '-').toUpperCase()}</div></div>
        <div class="history-item"><strong>RSI 14</strong><div>${technical.rsi_14 ?? '-'}</div></div>
        <div class="history-item"><strong>ATR 14</strong><div>${technical.atr_14 ?? '-'}</div></div>
        <div class="history-item"><strong>Equilibrium</strong><div>${fmtPrice(technical.equilibrium)}</div></div>
        <div class="history-item"><strong>Nearest Support</strong><div>${technical.nearest_support ? fmtPrice(technical.nearest_support.price) : '-'}</div></div>
        <div class="history-item"><strong>Nearest Resistance</strong><div>${technical.nearest_resistance ? fmtPrice(technical.nearest_resistance.price) : '-'}</div></div>
      `;

      orderBlocksHost.innerHTML = '';
      const orderBlocks = technical.order_blocks || [];
      if (!orderBlocks.length) {
        orderBlocksHost.innerHTML = '<div class="history-item">No nearby order blocks detected in the recent window.</div>';
      } else {
        for (const item of orderBlocks) {
          const node = document.createElement('div');
          node.className = 'history-item';
          node.innerHTML = `
            <div><strong>${String(item.type || '').replaceAll('_', ' ').toUpperCase()}</strong></div>
            <div>${fmtPrice(item.low)} -> ${fmtPrice(item.high)}</div>
            <div>Strength ${fmtPct(item.strength)}</div>
            <div>${item.note || ''}</div>
            <div>${fmtTime(item.timestamp)}</div>
          `;
          orderBlocksHost.appendChild(node);
        }
      }

      fvgHost.innerHTML = '';
      const gaps = technical.fair_value_gaps || [];
      if (!gaps.length) {
        fvgHost.innerHTML = '<div class="history-item">No recent fair value gaps detected.</div>';
      } else {
        for (const item of gaps) {
          const node = document.createElement('div');
          node.className = 'history-item';
          node.innerHTML = `
            <div><strong>${String(item.type || '').replaceAll('_', ' ').toUpperCase()}</strong></div>
            <div>${fmtPrice(item.low)} -> ${fmtPrice(item.high)}</div>
            <div>Gap size ${fmtPrice(item.size)}</div>
            <div>${fmtTime(item.timestamp)}</div>
          `;
          fvgHost.appendChild(node);
        }
      }
    }

    function renderConversation(payload) {
      const host = document.getElementById('conversation');
      host.innerHTML = '';
      const convo = payload.branch_conversation || {};
      const llm = (payload.llm_context || {}).content || {};
      if (!convo || (!convo.summary && !convo.debate_lines)) {
        host.innerHTML = '<div class="history-item">No branch conversation available.</div>';
        return;
      }
      const summary = document.createElement('div');
      summary.className = 'history-item';
      summary.innerHTML = `
        <div><strong>LLM Summary</strong></div>
        <div>${convo.summary || llm.explanation || 'No summary available.'}</div>
        <div style="margin-top:6px;">Top: ${convo.top_branch || 'n/a'}</div>
        <div>Minority: ${convo.minority_branch || 'n/a'}</div>
      `;
      host.appendChild(summary);
      for (const line of (convo.supporting_branches || [])) {
        const node = document.createElement('div');
        node.className = 'history-item';
        node.innerHTML = `<strong>Support</strong><div>${line}</div>`;
        host.appendChild(node);
      }
      for (const line of (convo.debate_lines || [])) {
        const node = document.createElement('div');
        node.className = 'history-item';
        node.textContent = line;
        host.appendChild(node);
      }
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
        const meta = [item.source, item.category, fmtTime(item.published_at), item.classification, item.sentiment !== undefined ? `sentiment ${Number(item.sentiment).toFixed(2)}` : '']
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

    function renderSwarmGraph(payload) {
      const graph = (payload.bot_swarm || {}).graph || {};
      const nodes = graph.nodes || [];
      const edges = graph.edges || [];
      if (!nodes.length) {
        Plotly.react('swarm-graph', [], { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(8,16,24,0.22)' }, { responsive: true, displaylogo: false });
        return;
      }

      const lineTraces = edges.map(edge => {
        const source = nodes.find(node => node.id === edge.source);
        const target = nodes.find(node => node.id === edge.target);
        if (!source || !target) return null;
        const positive = Number(edge.agreement || 0) >= 0;
        return {
          type: 'scatter',
          mode: 'lines',
          x: [source.x, target.x],
          y: [source.y, target.y],
          hoverinfo: 'skip',
          showlegend: false,
          line: {
            width: 1 + Number(edge.weight || 0) * 3,
            color: positive ? 'rgba(46,204,113,0.35)' : 'rgba(255,90,95,0.35)',
          },
        };
      }).filter(Boolean);

      const nodeTrace = {
        type: 'scatter',
        mode: 'markers+text',
        x: nodes.map(node => node.x),
        y: nodes.map(node => node.y),
        text: nodes.map(node => node.label),
        textposition: 'top center',
        marker: {
          size: nodes.map(node => node.size),
          color: nodes.map(node => node.color),
          line: { width: 1, color: 'rgba(255,255,255,0.25)' },
        },
        hovertemplate: '%{text}<extra></extra>',
        showlegend: false,
      };

      Plotly.react('swarm-graph', [...lineTraces, nodeTrace], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(8,16,24,0.22)',
        font: { color: '#eef5fb' },
        margin: { l: 10, r: 10, t: 10, b: 10 },
        xaxis: { visible: false },
        yaxis: { visible: false },
        uirevision: `${state.activeSymbol}-swarm-graph`,
      }, { responsive: true, displaylogo: false });
    }

    function renderBranchGraph(payload) {
      const graph = payload.branch_graph || {};
      const traces = [];
      for (const trace of (graph.traces || [])) {
        const highlight = trace.highlight || 'support';
        const style = highlight === 'top'
          ? { color: '#2ecc71', width: 3.5, dash: 'solid', opacity: 0.95, name: `Top ${trace.path_id}` }
          : highlight === 'minority'
          ? { color: '#f39c12', width: 2.5, dash: 'dot', opacity: 0.85, name: `Minority ${trace.path_id}` }
          : { color: 'rgba(77,163,255,0.28)', width: 1.4, dash: 'solid', opacity: 0.45, name: `Branch ${trace.path_id}` };
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: trace.timestamps,
          y: trace.prices,
          line: { color: style.color, width: style.width, dash: style.dash },
          opacity: style.opacity,
          name: style.name,
          hovertemplate: `${style.name}<br>label=${trace.branch_label}<br>prob=${Number(trace.probability || 0).toFixed(3)}<br>fitness=${Number(trace.branch_fitness || 0).toFixed(3)}<extra></extra>`,
        });
      }
      Plotly.react('branch-graph-view', traces, {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(8,16,24,0.22)',
        font: { color: '#eef5fb' },
        margin: { l: 40, r: 12, t: 8, b: 32 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
        hoverlabel: {
          bgcolor: '#0d1b26',
          bordercolor: 'rgba(255,255,255,0.18)',
          font: { color: '#eef5fb' },
        },
        legend: { orientation: 'h', x: 0, y: 1.08 },
        uirevision: `${state.activeSymbol}-branch-graph`,
      }, { responsive: true, displaylogo: false });
    }

    function tradingViewSymbol(symbol) {
      const map = {
        XAUUSD: 'OANDA:XAUUSD',
        EURUSD: 'FX:EURUSD',
        BTCUSD: 'BITSTAMP:BTCUSD',
      };
      return map[symbol] || 'OANDA:XAUUSD';
    }

    function renderTradingView(symbol) {
      const iframe = document.getElementById('tradingview-frame');
      if (!iframe) return;
      const tvSymbol = encodeURIComponent(tradingViewSymbol(symbol));
      const interval = [5, 10, 15, 30].includes(Number(state.activeHorizon)) ? Number(state.activeHorizon) : 15;
      iframe.src = `https://s.tradingview.com/widgetembed/?symbol=${tvSymbol}&interval=${interval}&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1&hide_side_toolbar=0&allow_symbol_change=1&saveimage=1&hideideas=1`;
    }

    function resizeWorkspacePlots(container) {
      if (!container || typeof Plotly === 'undefined') return;
      for (const plotId of ['swarm-graph', 'branch-graph-view', 'live-chart']) {
        const node = container.querySelector(`#${plotId}`);
        if (node) {
          try {
            Plotly.Plots.resize(node);
          } catch (error) {
            console.warn(`Plot resize skipped for ${plotId}`, error);
          }
        }
      }
    }

    function activateWorkspaceTab(button) {
      if (!button) return;
      const tabRow = button.closest('.workspace-tabs');
      if (!tabRow) return;
      const shell = button.closest('.workspace-shell');
      if (!shell) return;
      const targetId = button.dataset.target;
      tabRow.querySelectorAll('.workspace-tab').forEach((item) => item.classList.remove('active'));
      shell.querySelectorAll('.workspace-panel').forEach((panel) => panel.classList.remove('active'));
      button.classList.add('active');
      const target = shell.querySelector(`#${targetId}`);
      if (target) {
        target.classList.add('active');
        requestAnimationFrame(() => resizeWorkspacePlots(target));
      }
    }

    function renderCompareChart(simPayload, monitorPayload) {
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
      const branchPayload = (simPayload || {}).branches || [];
      const highlighted = (simPayload || {}).highlighted_branches || {};
      const finalForecast = ((simPayload || {}).final_forecast || {}).points || [];
      const topBranchId = highlighted.top_branch_id;
      const minorityBranchId = highlighted.minority_branch_id;
      const horizonLimit = Number(state.activeHorizon || 15);

      const traces = [{
        type: 'scatter',
        mode: 'markers',
        x: [anchorTs],
        y: [anchorPrice],
        marker: { size: 10, color: '#f39c12' },
        name: 'Simulation anchor',
      }];

      const topBranches = branchPayload
        .filter(branch => [topBranchId, minorityBranchId].includes(branch.path_id))
        .slice(0, 2);
      for (const branch of topBranches) {
        const isMinority = branch.path_id === minorityBranchId;
        const filteredTimestamps = [anchorTs].concat((branch.timestamps || []).filter((_, index) => ((index + 1) * 5) <= horizonLimit));
        const filteredPrices = [anchorPrice].concat((branch.predicted_prices || []).filter((_, index) => ((index + 1) * 5) <= horizonLimit));
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: filteredTimestamps,
          y: filteredPrices,
          line: {
            color: isMinority ? 'rgba(255,165,0,0.75)' : 'rgba(46,204,113,0.65)',
            width: isMinority ? 2 : 2.5,
            dash: isMinority ? 'dot' : 'solid',
          },
          name: isMinority ? 'Minority ghost path' : 'Top branch path',
          opacity: isMinority ? 0.75 : 0.95,
        });
      }

      const filteredCone = cone.filter(point => Number(point.horizon || 0) * 5 <= horizonLimit);
      if (filteredCone.length) {
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: filteredCone.map(row => row.timestamp),
          y: filteredCone.map(row => row.lower_price),
          line: { color: 'rgba(46,204,113,0.0)' },
          hoverinfo: 'skip',
          showlegend: false,
        });
        traces.push({
          type: 'scatter',
          mode: 'lines',
          x: filteredCone.map(row => row.timestamp),
          y: filteredCone.map(row => row.upper_price),
          line: { color: 'rgba(46,204,113,0.0)' },
          fill: 'tonexty',
          fillcolor: 'rgba(46,204,113,0.18)',
          name: 'Probability cone',
          hoverinfo: 'skip',
        });
      }

      const filteredCenter = center.filter((_, index) => ((index + 1) * 5) <= horizonLimit);
      const filteredFinal = finalForecast.filter(point => Number(point.minutes || 0) <= horizonLimit);
      traces.push({
        type: 'scatter',
        mode: 'lines+markers',
        x: [anchorTs].concat(filteredCenter.map(row => row.timestamp)),
        y: [anchorPrice].concat(filteredCenter.map(row => row.price)),
        line: { color: 'rgba(46,204,113,0.45)', width: 2, dash: 'dash' },
        marker: { size: 5, color: 'rgba(46,204,113,0.55)' },
        name: 'Simulator center',
      });
      if (filteredFinal.length) {
        traces.push({
          type: 'scatter',
          mode: 'lines+markers',
          x: [anchorTs].concat(filteredFinal.map(row => row.timestamp)),
          y: [anchorPrice].concat(filteredFinal.map(row => row.final_price)),
          line: { color: '#f1c40f', width: 3.5 },
          marker: { size: 7, color: '#f1c40f' },
          name: 'Final forecast',
        });
      }

      if (actual.length) {
        const filteredActual = actual.filter((_, index) => ((index + 1) * 5) <= horizonLimit);
        traces.push({
          type: 'scatter',
          mode: 'lines+markers',
          x: [anchorTs].concat(filteredActual.map(row => row.timestamp)),
          y: [anchorPrice].concat(filteredActual.map(row => row.close)),
          line: { color: '#4da3ff', width: 3 },
          marker: {
            size: 7,
            color: filteredActual.map(row => row.inside_cone ? '#4da3ff' : '#ff5a5f'),
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
      if (announce) setStatus(`Simulating ${symbol} in ${state.stackMode === 'v8_direct' ? 'V8 direct manual mode' : 'hybrid mode'} with live market context...`);
      const provider = encodeURIComponent(state.llmProvider || 'lm_studio');
      const stackMode = encodeURIComponent(state.stackMode || 'v8_direct');
      const startedAt = performance.now();
      const response = await fetch(`/api/simulate-live?symbol=${encodeURIComponent(symbol)}&llm_provider=${provider}&stack_mode=${stackMode}`);
      state.lastLatencyMs = Math.max(1, Math.round(performance.now() - startedAt));
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Request failed with ${response.status}`);
      }
      state.lastSimulation = await response.json();
      state.stackMode = String(state.lastSimulation.stack_mode || state.stackMode || 'v8_direct');
      document.getElementById('signal-mode').value = state.stackMode;
      renderPersonas(state.lastSimulation.personas || {});
      renderTilts(state.lastSimulation.persona_weight_tilts || {});
      renderConversation(state.lastSimulation);
      renderSwarmJudge(state.lastSimulation);
      renderPublicReaction(state.lastSimulation);
      renderSpecialistBots(state.lastSimulation);
      renderTechnical(state.lastSimulation);
      renderForecastLadder(state.lastSimulation);
      renderMacro(state.lastSimulation);
      renderList('news', (state.lastSimulation.feeds || {}).news || []);
      renderList('crowd', (state.lastSimulation.feeds || {}).public_discussions || []);
      renderSwarmGraph(state.lastSimulation);
      renderBranchGraph(state.lastSimulation);
      renderTradingView(symbol);
      await refreshMonitor(false);
      updateHeaderShell();
      setStatus(`Simulation frozen at ${fmtTime(state.lastSimulation.generated_at)} for ${symbol}. ${state.stackMode === 'v8_direct' ? 'Manual V8 direct projection is active.' : 'Hybrid ensemble projection is active.'}`);
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
        renderCompareChart(state.lastSimulation, state.lastMonitor);
        renderHistory(state.lastMonitor);
      }
      if (announce) setStatus(`Live market refreshed at ${fmtTime((state.lastMonitor || {}).server_timestamp)}.`);
      updateHeaderShell();
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
    document.getElementById('forecast-horizon').addEventListener('change', (event) => {
      state.activeHorizon = Number(event.target.value || 15);
      if (state.lastSimulation && state.lastMonitor) {
        renderCompareChart(state.lastSimulation, state.lastMonitor);
      }
      renderTradingView(state.activeSymbol || document.getElementById('symbol').value);
    });
    document.getElementById('llm-provider').addEventListener('change', (event) => {
      state.llmProvider = String(event.target.value || 'lm_studio');
      updateHeaderShell();
    });
    document.getElementById('signal-mode').addEventListener('change', (event) => {
      state.stackMode = String(event.target.value || 'v8_direct');
      updateHeaderShell();
    });
    document.querySelectorAll('.workspace-tab').forEach((button) => {
      button.addEventListener('click', () => activateWorkspaceTab(button));
    });

    window.addEventListener('load', async () => {
      try {
        updateHeaderShell();
        setInterval(updateHeaderShell, 1000);
        document.querySelectorAll('.workspace-tab.active').forEach((button) => activateWorkspaceTab(button));
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
