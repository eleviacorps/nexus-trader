from __future__ import annotations


_HTML_HEAD = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Nexus Trader V18</title>
<script src='https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js'></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;700&display=swap');
:root{--bg:#03060a;--bg2:#07111a;--glass:rgba(14,22,31,.82);--line:rgba(173,210,255,.13);--text:#eaf2fb;--muted:#8fa4b8;--bull:#63e6ad;--bear:#ff8f86;--amber:#ffd38a;--cyan:#8ed8ff;--green:#81f1b6;--shadow:0 28px 80px rgba(0,0,0,.62),inset 1px 1px 0 rgba(255,255,255,.04),inset -18px -18px 32px rgba(0,0,0,.22);--neo:inset 12px 12px 24px rgba(0,0,0,.24),inset -10px -10px 18px rgba(255,255,255,.02)}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;color:var(--text);font-family:'Space Grotesk',sans-serif;background:radial-gradient(circle at 12% 16%,rgba(142,216,255,.06),transparent 18%),radial-gradient(circle at 84% 8%,rgba(255,211,138,.05),transparent 22%),radial-gradient(circle at 72% 72%,rgba(99,230,173,.04),transparent 20%),linear-gradient(170deg,#010204 0%,#03070b 34%,#051019 68%,#02060b 100%);min-height:100vh}
body::before{content:'';position:fixed;inset:0;pointer-events:none;background:linear-gradient(transparent 49%,rgba(255,255,255,.014) 50%,transparent 51%),linear-gradient(90deg,transparent 49%,rgba(255,255,255,.01) 50%,transparent 51%);background-size:100% 20px,20px 100%;opacity:.12}
#app{max-width:1720px;margin:0 auto;padding:18px;display:grid;gap:14px}
.grid{display:grid;gap:14px;min-width:0}.top-grid{grid-template-columns:minmax(0,1.5fr) minmax(360px,.94fr)}.bottom-grid{grid-template-columns:minmax(430px,1.08fr) minmax(320px,.88fr) minmax(320px,.96fr)}
.panel{min-width:0;overflow:hidden;border-radius:28px;border:1px solid var(--line);background:linear-gradient(180deg,rgba(17,29,41,.82),rgba(7,15,24,.95));backdrop-filter:blur(22px) saturate(128%);-webkit-backdrop-filter:blur(22px) saturate(128%);box-shadow:var(--shadow),var(--neo)}
.panel-inner{padding:18px}.panel-body{display:flex;flex-direction:column;gap:12px;min-height:0;height:100%}.panel-scroll{overflow:auto;min-height:0;flex:1 1 auto;padding-right:4px}.panel-scroll::-webkit-scrollbar,.table-wrap::-webkit-scrollbar{width:10px;height:10px}.panel-scroll::-webkit-scrollbar-thumb,.table-wrap::-webkit-scrollbar-thumb{background:rgba(142,216,255,.18);border-radius:999px}
.hero{min-height:196px}.h-chart,.h-judge{height:min(74vh,800px)}.h-paper,.h-feed,.h-packet{height:min(62vh,690px)}
.row{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}.stack{display:grid;gap:12px;min-width:0}.split{display:grid;gap:12px;grid-template-columns:repeat(2,minmax(0,1fr))}.triple{display:grid;gap:12px;grid-template-columns:repeat(3,minmax(0,1fr))}.quad{display:grid;gap:12px;grid-template-columns:repeat(4,minmax(0,1fr))}.controls{display:grid;gap:12px;grid-template-columns:repeat(7,minmax(0,1fr))}
.section-title{margin:0;font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:var(--amber)}.headline{margin:0;font-size:clamp(30px,4.5vw,56px);line-height:.95;letter-spacing:-.055em}.muted{color:var(--muted)}.mono{font-family:'JetBrains Mono',monospace}
.stat,.item,.news-item,.legend-item{padding:14px;border-radius:22px;border:1px solid rgba(255,255,255,.08);background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.028));box-shadow:var(--neo);min-width:0}
.label{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}.value{font-size:26px;font-weight:700;line-height:1.05}.value-big{font-size:46px}.tiny{font-size:12px;line-height:1.5;color:var(--muted)}
.status{padding:12px 14px;border-radius:18px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.05);color:var(--muted);overflow-wrap:anywhere}.status.ok{color:var(--bull)}.status.warn{color:var(--amber)}.status.bad{color:var(--bear)}
.badge,.pill{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.05);font-size:12px;letter-spacing:.08em;text-transform:uppercase}
.decision-shell{display:grid;gap:14px;grid-template-columns:minmax(0,1.35fr) minmax(300px,.75fr)}.progress-wrap{height:18px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);overflow:hidden;box-shadow:var(--neo)}.progress-fill{height:100%;width:0;background:linear-gradient(90deg,rgba(142,216,255,.52),rgba(99,230,173,.82));box-shadow:0 0 22px rgba(99,230,173,.22);transition:width .18s ease}
select,input,textarea,button{width:100%;border-radius:16px;border:1px solid rgba(255,255,255,.10);background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.03));color:var(--text);padding:11px 12px;font:inherit;box-shadow:var(--neo)}textarea{min-height:84px;resize:vertical}
button{cursor:pointer;font-weight:700;background:linear-gradient(135deg,rgba(142,216,255,.16),rgba(255,211,138,.12)),linear-gradient(180deg,rgba(255,255,255,.08),rgba(255,255,255,.03))}.btn-secondary{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.025))}
.judge-hero{padding:16px;border-radius:24px;background:linear-gradient(180deg,rgba(255,255,255,.07),rgba(255,255,255,.03));border:1px solid rgba(255,255,255,.08)}.judge-grid{display:grid;gap:12px;grid-template-columns:repeat(4,minmax(0,1fr))}
.table-wrap{overflow:auto;border-radius:20px;border:1px solid rgba(255,255,255,.06);max-height:240px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 9px;text-align:left;border-bottom:1px solid rgba(255,255,255,.06);vertical-align:top}th{position:sticky;top:0;background:rgba(5,12,19,.96);z-index:1}
details{border:1px solid rgba(255,255,255,.08);border-radius:18px;background:rgba(255,255,255,.03);overflow:hidden}summary{cursor:pointer;padding:12px 14px;color:var(--amber);font-size:12px;letter-spacing:.12em;text-transform:uppercase}pre{margin:0;padding:14px;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;font:12px/1.55 'JetBrains Mono',monospace;color:#d7edff}
.collapse-copy{padding:0 14px 14px}.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.10),transparent)}.news-title{font-weight:700;line-height:1.45;overflow-wrap:anywhere}.news-meta{margin-top:6px;font-size:12px;color:var(--muted)}
.summary-card{padding:14px;border-radius:22px;border:1px solid rgba(255,255,255,.08);background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.028));overflow-wrap:anywhere;word-break:break-word}.summary-call{font-size:18px;font-weight:700}
.legend-grid{display:grid;gap:10px;grid-template-columns:repeat(2,minmax(0,1fr))}.legend-line{display:inline-block;width:26px;height:0;border-top:3px solid currentColor;vertical-align:middle;margin-right:8px}.legend-dash{border-top-style:dashed}.legend-dot{border-top-style:dotted}
#chart-wrap{min-height:0;flex:1 1 auto}#chart{height:520px}.no-wrap{white-space:nowrap}.emphasis{color:var(--cyan)}.wrap-anywhere,.item,.stat,td,th{overflow-wrap:anywhere;word-break:break-word}
@media(max-width:1450px){.top-grid,.bottom-grid{grid-template-columns:1fr}.controls{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:900px){#app{padding:12px}.decision-shell,.split,.triple,.quad,.judge-grid,.controls,.legend-grid{grid-template-columns:1fr}.h-chart,.h-judge,.h-paper,.h-feed,.h-packet{height:auto;min-height:unset}.panel-scroll{max-height:none}#chart{height:420px}}
</style>
</head>
<body>
<div id='app'>
  <section class='panel hero'>
    <div class='panel-inner panel-body'>
      <div class='row'>
        <div>
          <div class='section-title'>Nexus Trader V18</div>
          <h1 class='headline'>Dark Glass Terminal For The 15-Minute Decision Window</h1>
          <div class='muted'>V18 now keeps the chart stable, separates the V18 path from the Kimi path, and keeps every glass panel scroll-contained instead of stretching the page.</div>
        </div>
        <div class='stack' style='min-width:280px;text-align:right'>
          <div id='clock' class='value value-big mono'>--:--:--</div>
          <div id='date' class='tiny'>Waiting for local time</div>
          <div id='health' class='status'>Connecting realtime services...</div>
        </div>
      </div>
      <div class='decision-shell'>
        <div class='stack'>
          <div class='row'>
            <div class='badge' id='decision-stance'>HOLD</div>
            <div class='badge' id='decision-action'>WAIT</div>
            <div class='badge'><span class='mono' id='decision-symbol'>XAUUSD</span> / <span id='decision-mode'>Frequency</span></div>
          </div>
          <div class='label'>Decision Readiness</div>
          <div class='progress-wrap'><div id='bar-fill' class='progress-fill'></div></div>
          <div class='tiny' id='bar-note'>Readiness rises when the live price is close to the active entry zone and the current trade call is actionable.</div>
          <div class='triple'>
            <div class='stat'><div class='label'>Bar Countdown</div><div id='bar-countdown' class='value mono'>15m 00s</div><div class='tiny'>Seconds remaining in the current 15-minute decision bar.</div></div>
            <div class='stat'><div class='label'>Entry Zone</div><div id='decision-entry' class='value'>-</div><div class='tiny'>The current Kimi trigger band for this bar.</div></div>
            <div class='stat'><div class='label'>TP / SL</div><div id='decision-targets' class='value'>-</div><div class='tiny'>Current take-profit and invalidation levels for the active idea.</div></div>
          </div>
        </div>
        <div class='stack'>
          <div class='stat'><div class='label'>Current Price</div><div id='hero-price' class='value value-big'>-</div><div class='tiny'>Live quote stream updated over the WebSocket heartbeat.</div></div>
          <div class='quad'>
            <div class='stat'><div class='label'>CABR</div><div id='hero-cabr' class='value'>-</div></div>
            <div class='stat'><div class='label'>SQT</div><div id='hero-sqt' class='value'>-</div></div>
            <div class='stat'><div class='label'>Hurst</div><div id='hero-hurst' class='value'>-</div></div>
            <div class='stat'><div class='label'>Lot</div><div id='hero-lot' class='value'>-</div></div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class='panel'>
    <div class='panel-inner panel-body'>
      <div class='controls'>
        <div><div class='label'>Instrument</div><select id='symbol'><option value='XAUUSD' selected>XAUUSD</option><option value='EURUSD'>EURUSD</option><option value='BTCUSD'>BTCUSD</option></select></div>
        <div><div class='label'>Mode</div><select id='mode'><option value='frequency' selected>Frequency</option><option value='precision'>Precision</option></select></div>
        <div><div class='label'>LLM Provider</div><select id='provider'><option value='nvidia_nim' selected>NVIDIA NIM / Kimi</option><option value='lm_studio'>LM Studio</option><option value='ollama'>Ollama</option></select></div>
        <div><div class='label'>Model Override</div><input id='model' value='moonshotai/kimi-k2-instruct' placeholder='moonshotai/kimi-k2-instruct'></div>
        <div><div class='label'>Refresh Seconds</div><select id='refresh'><option value='15' selected>15</option><option value='30'>30</option><option value='60'>60</option></select></div>
        <div><div class='label'>Auto Refresh</div><select id='auto'><option value='on' selected>On</option><option value='off'>Off</option></select></div>
        <div><div class='label'>Simulation</div><button id='run'>Run V18</button></div>
      </div>
      <div id='status' class='status'>Preparing the first simulation snapshot.</div>
    </div>
  </section>

  <section class='grid top-grid'>
    <section class='panel h-chart'>
      <div class='panel-inner panel-body'>
        <div class='row'>
          <div>
            <div class='section-title'>Realtime Chart</div>
            <div class='muted'>Yellow is the V18 consensus path, blue dashed is the V18 minority path, red dashed rails are the outer cone, and green dotted is the separate Kimi projection.</div>
          </div>
          <div class='row'>
            <div class='pill'>Tier <strong id='chart-tier'>-</strong></div>
            <div class='pill'>Cone <strong id='chart-cone'>-</strong></div>
            <div class='pill'>MFG <strong id='chart-mfg'>-</strong></div>
          </div>
        </div>
        <div class='legend-grid'>
          <div class='legend-item'><span class='legend-line' style='color:var(--amber)'></span>V18 consensus path: the main simulator path used for the desk bias.</div>
          <div class='legend-item'><span class='legend-line legend-dash' style='color:var(--cyan)'></span>V18 minority path: the valid alternative branch, not the opposite of consensus by design.</div>
          <div class='legend-item'><span class='legend-line legend-dash' style='color:var(--bear)'></span>Outer cone rails: the hard V18 volatility envelope for the next 15 minutes.</div>
          <div class='legend-item'><span class='legend-line legend-dot' style='color:var(--green)'></span>Kimi path: Kimi's separate projection, shown independently and not fed back into V18.</div>
        </div>
        <div id='chart-wrap' class='panel-scroll'><div id='chart'></div></div>
      </div>
    </section>

    <section class='panel h-judge'>
      <div class='panel-inner panel-body'>
        <div class='row'>
          <div>
            <div class='section-title'>AI Judge</div>
            <div class='muted'>The Kimi desk view now shows a final call, a pure market/news read, a pure V18 read, and the merged combined summary.</div>
          </div>
          <button id='apply-kimi' class='btn-secondary' style='max-width:220px'>Apply To Trade Form</button>
        </div>
        <div id='judge-status' class='status'>Waiting for the first Kimi decision.</div>
        <div class='judge-hero'>
          <div class='row'>
            <div class='badge' id='judge-stance'>HOLD</div>
            <div class='badge' id='judge-confidence'>VERY_LOW</div>
            <div class='badge' id='judge-final-call'>SKIP</div>
            <div class='badge' id='judge-rr'>R:R -</div>
          </div>
          <div class='judge-grid' style='margin-top:12px'>
            <div class='stat'><div class='label'>Entry Zone</div><div id='judge-entry' class='value'>-</div></div>
            <div class='stat'><div class='label'>Stop Loss</div><div id='judge-sl' class='value'>-</div></div>
            <div class='stat'><div class='label'>Take Profit</div><div id='judge-tp' class='value'>-</div></div>
            <div class='stat'><div class='label'>Hold Time</div><div id='judge-hold' class='value'>-</div></div>
          </div>
          <div class='summary-card' style='margin-top:12px'>
            <div class='label'>Final Desk Summary</div>
            <div id='summary-final-call' class='summary-call'>SKIP</div>
            <div id='summary-final' class='tiny'>Waiting for Kimi.</div>
          </div>
        </div>
        <div class='panel-scroll stack'>
          <div class='summary-card'><div class='label'>Market-Only Summary</div><div id='summary-market-call' class='summary-call'>-</div><div id='summary-market' class='tiny'>Waiting for Kimi.</div><div id='summary-market-reason' class='tiny emphasis'>-</div></div>
          <div class='summary-card'><div class='label'>V18-Only Summary</div><div id='summary-v18-call' class='summary-call'>-</div><div id='summary-v18' class='tiny'>Waiting for Kimi.</div><div id='summary-v18-reason' class='tiny emphasis'>-</div></div>
          <div class='summary-card'><div class='label'>Combined Summary</div><div id='summary-combined-call' class='summary-call'>-</div><div id='summary-combined' class='tiny'>Waiting for Kimi.</div><div id='summary-combined-reason' class='tiny emphasis'>-</div></div>
          <div class='item'><div class='label'>Reasoning</div><div id='judge-reasoning' class='tiny'>Waiting for Kimi.</div></div>
          <div class='item'><div class='label'>Key Risk</div><div id='judge-risk' class='tiny'>Waiting for Kimi.</div></div>
          <div class='item'><div class='label'>Crowd Note</div><div id='judge-crowd' class='tiny'>Waiting for Kimi.</div></div>
          <div class='item'><div class='label'>Regime Note</div><div id='judge-regime' class='tiny'>Waiting for Kimi.</div></div>
          <details open><summary>Raw Judge JSON</summary><pre id='judge-raw'>{}</pre></details>
          <details><summary>Context Sent To Kimi</summary><pre id='judge-context'>{}</pre></details>
        </div>
      </div>
    </section>
  </section>

  <section class='grid bottom-grid'>
    <section class='panel h-paper'>
      <div class='panel-inner panel-body'>
        <div>
          <div class='section-title'>Paper Trading Terminal</div>
          <div class='muted'>The paper desk now keeps the trade form usable, lets you override lot size manually, and keeps open and closed trades visible inside scroll-safe tables.</div>
        </div>
        <div class='panel-scroll stack'>
          <div class='quad'>
            <div class='stat'><div class='label'>Balance</div><div id='pb' class='value'>-</div></div>
            <div class='stat'><div class='label'>Equity</div><div id='pe' class='value'>-</div></div>
            <div class='stat'><div class='label'>Realized</div><div id='pr' class='value'>-</div></div>
            <div class='stat'><div class='label'>Unrealized</div><div id='pu' class='value'>-</div></div>
          </div>
          <div class='split'>
            <div><div class='label'>Direction</div><select id='tdir'><option value='BUY'>BUY</option><option value='SELL'>SELL</option></select></div>
            <div><div class='label'>Leverage</div><select id='tlev'><option value='50'>1:50</option><option value='100'>1:100</option><option value='200' selected>1:200</option></select></div>
          </div>
          <div class='triple'>
            <div><div class='label'>Stop Pips</div><input id='tstop' type='number' min='1' step='0.1' value='20'></div>
            <div><div class='label'>Take Profit Pips</div><input id='ttp' type='number' min='1' step='0.1' value='30'></div>
            <div><div class='label'>Manual Lot Override</div><input id='tlot' type='number' min='0' step='0.01' placeholder='leave blank for auto'></div>
          </div>
          <div class='split'>
            <div class='item'><div class='label'>Suggested Lot</div><div id='trade-lot' class='value'>-</div><div class='tiny'>Auto lot uses equity, tier, SQT, and leverage. Manual lot is capped by leverage before execution.</div></div>
            <div class='item'><div class='label'>Trade Setup</div><div id='trade-kimi' class='tiny'>Apply the current Kimi decision to prefill direction and distances, or keep your own manual setup.</div></div>
          </div>
          <div><div class='label'>Trade Note</div><textarea id='tnote' placeholder='Why this paper trade exists'></textarea></div>
          <div class='split'><button id='open'>Open Paper Trade</button><button id='reset' class='btn-secondary'>Reset Paper Account</button></div>
          <div id='paper-status' class='status'>Paper terminal ready.</div>
          <div class='divider'></div>
          <div class='section-title'>Open Positions</div>
          <div class='table-wrap'><table><thead><tr><th>ID</th><th>Side</th><th>Lot</th><th>Source</th><th>Entry</th><th>PnL</th><th>SL</th><th>TP</th><th>Status</th><th>Actions</th></tr></thead><tbody id='open-body'></tbody></table></div>
          <div class='section-title'>Closed History</div>
          <div class='table-wrap'><table><thead><tr><th>Time</th><th>Side</th><th>Lot</th><th>Entry</th><th>Exit</th><th>PnL</th></tr></thead><tbody id='closed-body'></tbody></table></div>
        </div>
      </div>
    </section>

    <section class='panel h-feed'>
      <div class='panel-inner panel-body'>
        <div>
          <div class='section-title'>Live Context Feed</div>
          <div class='muted'>This panel blends market structure, crowd state, live headlines, and public discussion items so the desk never looks blank.</div>
        </div>
        <div class='split'>
          <div class='item'><div class='label'>Market Structure</div><div id='feed-structure'>-</div><div id='feed-structure-sub' class='tiny'>-</div></div>
          <div class='item'><div class='label'>Crowd State</div><div id='feed-crowd'>-</div><div id='feed-crowd-sub' class='tiny'>-</div></div>
        </div>
        <div id='feeds' class='panel-scroll stack'></div>
      </div>
    </section>

    <section class='panel h-packet'>
      <div class='panel-inner panel-body'>
        <div>
          <div class='section-title'>Kimi Packet + Numeric Guide</div>
          <div class='muted'>Every 15-minute Kimi packet stays visible here with the exact numeric context and plain-English meaning of the key values.</div>
        </div>
        <div id='kimi-meta' class='status'>No Kimi packet logged yet.</div>
        <div class='panel-scroll stack'>
          <details open><summary>Latest Packet Context</summary><pre id='kimi-context'>{}</pre></details>
          <details open><summary>Numeric Glossary</summary><div id='kimi-glossary' class='collapse-copy stack'></div></details>
          <details><summary>What The Numbers Mean</summary><div id='guide' class='collapse-copy stack'></div></details>
        </div>
      </div>
    </section>
  </section>
</div>
<script>
"""

_HTML_SCRIPT = """const PIP_SIZES={XAUUSD:0.1,EURUSD:0.0001,BTCUSD:1};
const state={
  symbol:'XAUUSD',
  mode:'frequency',
  provider:'nvidia_nim',
  model:'moonshotai/kimi-k2-instruct',
  sim:null,
  paper:null,
  kimiLog:null,
  ws:null,
  timer:null,
  simLoading:false,
  live:{price:null,positions:[],paper_summary:{},bar_countdown:900,bar_progress:0,sqt:{label:'NEUTRAL',rolling_accuracy:0.5},connected:false,lastMessageAt:0,timestamp:null},
  chart:{instance:null,candles:null,consensus:null,minority:null,outerUpper:null,outerLower:null,kimi:null,resizeObserver:null,entryLines:[],positionLines:[],seeded:false,baseCandles:[]},
};

const $=(id)=>document.getElementById(id);
const tone=(value)=>{const v=String(value||'').toUpperCase();if(v==='BUY'||v==='BULLISH'||v==='HOT'||v==='READY')return'var(--bull)';if(v==='SELL'||v==='BEARISH'||v==='COLD'||v==='BAD')return'var(--bear)';return'var(--amber)'};
const fmtNum=(value,digits=2)=>value==null||Number.isNaN(Number(value))?'-':Number(value).toFixed(digits);
const fmtPrice=(value)=>value==null||Number.isNaN(Number(value))?'-':Number(value).toFixed(2);
const fmtPct=(value)=>value==null||Number.isNaN(Number(value))?'-':`${(Number(value)*100).toFixed(1)}%`;
const fmtSigned=(value,digits=2)=>value==null||Number.isNaN(Number(value))?'-':`${Number(value)>=0?'+':''}${Number(value).toFixed(digits)}`;
const fmtMoney=(value)=>value==null||Number.isNaN(Number(value))?'-':`${Number(value)>=0?'+':''}$${Math.abs(Number(value)).toFixed(2)}`;
const cleanText=(raw)=>String(raw||'').replace(/<[^>]*>/g,' ').replace(/&nbsp;/g,' ').replace(/\\s+/g,' ').trim();
const toUnix=(value)=>{if(!value)return null;const stamp=Date.parse(String(value));return Number.isFinite(stamp)?Math.floor(stamp/1000):null;};
const wsOrigin=()=>location.protocol==='https:'?`wss://${location.host}`:`ws://${location.host}`;

function setStatus(id,text,kind=''){const el=$(id);if(!el)return;el.textContent=text;el.className=`status${kind?` ${kind}`:''}`;}
function setStatusLine(text,kind=''){setStatus('status',text,kind);}
function setPaperStatus(text,kind=''){setStatus('paper-status',text,kind);}
function currentSimulation(){return (state.sim||{}).simulation||{};}
function currentMarket(){return (state.sim||{}).market||{};}
function currentTech(){return (state.sim||{}).technical_analysis||{};}
function currentPaper(){return state.paper||((state.sim||{}).paper_trading)||{};}
function currentJudgeEnvelope(){return (state.sim||{}).kimi_judge||{};}
function currentJudge(){return currentJudgeEnvelope().content||{};}
function latestKimiEntry(){const entries=((state.kimiLog||{}).entries)||[];return entries.length?entries[entries.length-1]:null;}
function currentPrice(){const live=Number(state.live.price);const market=Number(currentMarket().current_price);return Number.isFinite(live)&&live>0?live:(Number.isFinite(market)?market:null);}

async function jsonFetch(url,options){
  const response=await fetch(url,options);
  const text=await response.text();
  let payload={};
  try{payload=text?JSON.parse(text):{};}catch(_err){payload={detail:text||response.statusText};}
  if(!response.ok){
    const detail=typeof payload.detail==='string'?payload.detail:JSON.stringify(payload);
    throw new Error(detail||`${response.status} ${response.statusText}`);
  }
  return payload;
}

function updateClock(){
  const now=new Date();
  $('clock').textContent=now.toLocaleTimeString();
  $('date').textContent=now.toLocaleDateString(undefined,{weekday:'short',year:'numeric',month:'short',day:'numeric'});
}

function updateHealth(){
  const liveAge=state.live.lastMessageAt?Math.max(0,Math.round((Date.now()-state.live.lastMessageAt)/1000)):null;
  const wsText=state.live.connected?`WebSocket online | last tick ${liveAge ?? 0}s ago`:`WebSocket offline | waiting for heartbeat`;
  const judge=currentJudgeEnvelope();
  let judgeText='Kimi waiting on first refresh';
  let kind='warn';
  if(judge.available){
    judgeText=`Kimi live | ${judge.model||state.model||'-'}`;
    kind=state.live.connected?'ok':'warn';
  }else if(judge.error){
    judgeText=`Kimi fallback | ${judge.error}`;
    kind='warn';
  }else if(judge.reason){
    judgeText=`Kimi cache | ${judge.reason}`;
  }
  setStatus('health',`${wsText} | ${judgeText}`,kind);
}

function guideItems(){
  const sim=currentSimulation();
  const tech=currentTech();
  const judge=currentJudge();
  return[
    {label:'CABR',value:fmtPct(sim.cabr_score),meaning:'Branch-ranking confidence for the main V18 path. Higher is stronger.'},
    {label:'SQT',value:`${((state.live.sqt||{}).label)||sim.sqt_label||'NEUTRAL'} / ${fmtPct(((state.live.sqt||{}).rolling_accuracy)||sim.sqt_accuracy)}`,meaning:'Recent simulator hit rate. COLD means the simulator has been wrong recently.'},
    {label:'Hurst',value:`${fmtNum(sim.hurst_overall,3)} / ${fmtSigned(sim.hurst_asymmetry,3)}`,meaning:'Overall persistence first, then asymmetry. Positive asymmetry means up moves persist more.'},
    {label:'Cone Width',value:sim.cone_width_pips==null?'-':`${fmtNum(sim.cone_width_pips,1)} pips`,meaning:'Average width of the inner V18 cone for the next 15 minutes.'},
    {label:'RSI / ATR',value:`${fmtNum(tech.rsi_14,1)} / ${fmtNum(tech.atr_14,2)}`,meaning:'Momentum and local volatility at the current anchor.'},
    {label:'Suggested Lot',value:sim.suggested_lot==null?'-':`${fmtNum(sim.suggested_lot,2)} lot`,meaning:'Auto-sized lot from equity, confidence tier, SQT label, and leverage rules.'},
    {label:'Final Call',value:String(judge.final_call||'SKIP').toUpperCase(),meaning:'The final Kimi desk action for this 15-minute bar.'},
    {label:'Market Price',value:fmtPrice(currentPrice()),meaning:'Current live quote driving the desk and paper-trade updates.'},
  ];
}

function renderGuide(){
  const host=$('guide');
  host.innerHTML='';
  guideItems().forEach((item)=>{
    const node=document.createElement('div');
    node.className='item';
    node.innerHTML=`<div class='label'>${item.label}</div><div class='value' style='font-size:20px'>${item.value}</div><div class='tiny'>${item.meaning}</div>`;
    host.appendChild(node);
  });
}

function renderDecisionBar(){
  const sim=currentSimulation();
  const judge=currentJudge();
  const price=currentPrice();
  const countdown=Math.max(0,Number(state.live.bar_countdown||900));
  const call=String(judge.final_call||judge.stance||'SKIP').toUpperCase().replace('HOLD','SKIP');
  const zone=Array.isArray(judge.entry_zone)&&judge.entry_zone.length===2?judge.entry_zone:(Array.isArray(sim.entry_zone)&&sim.entry_zone.length===2?sim.entry_zone:[]);
  const confidenceLabel=String(judge.confidence||sim.confidence_tier||'LOW').toUpperCase();
  const confidenceMap={VERY_LOW:.18,LOW:.35,MODERATE:.62,HIGH:.82,VERY_HIGH:.95};
  let readiness=0.12;
  let note='No active entry zone yet. Readiness stays low until a valid setup forms.';
  if(zone.length===2&&price!=null){
    const low=Math.min(Number(zone[0]),Number(zone[1]));
    const high=Math.max(Number(zone[0]),Number(zone[1]));
    const width=Math.max(high-low,Number(currentTech().atr_14||1)*0.4,0.1);
    const mid=(low+high)/2;
    const distance=Math.abs(price-mid);
    const proximity=Math.max(0,1-Math.min(distance/(width*1.8),1));
    readiness=Math.max(0.08,Math.min(1,(0.72*proximity)+(0.28*(confidenceMap[confidenceLabel]||0.25))));
    note=price>=low&&price<=high?'Live price is inside the active entry zone.':'Readiness reflects how close live price is to the active entry zone.';
  }else if(call==='BUY'||call==='SELL'){
    readiness=0.35+(0.45*(confidenceMap[confidenceLabel]||0.25));
    note='The setup is actionable, but there is no two-sided entry zone to measure proximity against.';
  }
  const readinessLabel=call==='BUY'||call==='SELL'?(readiness>=0.75?'READY':readiness>=0.45?'SETUP':'WAIT'):'WAIT';
  $('decision-stance').textContent=call;
  $('decision-stance').style.color=tone(call);
  $('decision-action').textContent=readinessLabel;
  $('decision-action').style.color=tone(readinessLabel);
  $('decision-symbol').textContent=state.symbol;
  $('decision-mode').textContent=state.mode;
  $('hero-price').textContent=fmtPrice(price);
  $('hero-cabr').textContent=fmtPct(sim.cabr_score);
  $('hero-sqt').textContent=`${((state.live.sqt||{}).label)||sim.sqt_label||'-'}`;
  $('hero-hurst').textContent=`${fmtNum(sim.hurst_overall,3)} / ${fmtSigned(sim.hurst_asymmetry,3)}`;
  $('hero-lot').textContent=$('tlot').value.trim()?`${fmtNum(Number($('tlot').value),2)} lot`:(sim.suggested_lot==null?'-':`${fmtNum(sim.suggested_lot,2)} lot`);
  $('bar-countdown').textContent=`${Math.floor(countdown/60)}m ${String(countdown%60).padStart(2,'0')}s`;
  $('bar-fill').style.width=`${Math.round(readiness*100)}%`;
  $('bar-note').textContent=note;
  $('decision-entry').textContent=zone.length===2?`${fmtPrice(zone[0])} - ${fmtPrice(zone[1])}`:'-';
  $('decision-targets').textContent=judge.take_profit!=null||judge.stop_loss!=null?`TP ${fmtPrice(judge.take_profit)} | SL ${fmtPrice(judge.stop_loss)}`:'-';
  updateHealth();
}

function renderJudge(){
  const envelope=currentJudgeEnvelope();
  const judge=currentJudge();
  const rr=(judge.take_profit!=null&&judge.stop_loss!=null&&Array.isArray(judge.entry_zone)&&judge.entry_zone.length===2)?
    (()=>{const entry=(Number(judge.entry_zone[0])+Number(judge.entry_zone[1]))/2;const reward=Math.abs(Number(judge.take_profit)-entry);const risk=Math.abs(entry-Number(judge.stop_loss));return risk>0?`${(reward/risk).toFixed(2)} : 1`:'-';})():'-';
  $('judge-stance').textContent=String(judge.stance||'HOLD').toUpperCase();
  $('judge-stance').style.color=tone(judge.stance);
  $('judge-confidence').textContent=String(judge.confidence||'VERY_LOW').toUpperCase();
  $('judge-final-call').textContent=String(judge.final_call||'SKIP').toUpperCase();
  $('judge-final-call').style.color=tone(judge.final_call);
  $('judge-rr').textContent=`R:R ${rr}`;
  $('judge-entry').textContent=Array.isArray(judge.entry_zone)&&judge.entry_zone.length===2?`${fmtPrice(judge.entry_zone[0])} - ${fmtPrice(judge.entry_zone[1])}`:'-';
  $('judge-sl').textContent=fmtPrice(judge.stop_loss);
  $('judge-tp').textContent=fmtPrice(judge.take_profit);
  $('judge-hold').textContent=judge.hold_time||'-';
  $('summary-final-call').textContent=String(judge.final_call||'SKIP').toUpperCase();
  $('summary-final-call').style.color=tone(judge.final_call);
  $('summary-final').textContent=judge.final_summary||'Waiting for Kimi.';
  $('judge-reasoning').textContent=judge.reasoning||'Waiting for Kimi.';
  $('judge-risk').textContent=judge.key_risk||'Waiting for Kimi.';
  $('judge-crowd').textContent=judge.crowd_note||'Waiting for Kimi.';
  $('judge-regime').textContent=judge.regime_note||'Waiting for Kimi.';
  const blocks=[
    ['market',judge.market_only_summary||{}],
    ['v18',judge.v18_summary||{}],
    ['combined',judge.combined_summary||{}],
  ];
  blocks.forEach(([prefix,block])=>{
    $(`summary-${prefix}-call`).textContent=String(block.call||'-').toUpperCase();
    $(`summary-${prefix}-call`).style.color=tone(block.call);
    $(`summary-${prefix}`).textContent=block.summary||'No summary available.';
    $(`summary-${prefix}-reason`).textContent=block.reasoning||'No reasoning available.';
  });
  $('judge-raw').textContent=JSON.stringify(judge,null,2);
  const latest=latestKimiEntry();
  $('judge-context').textContent=JSON.stringify((latest&&latest.context)||{},null,2);
  if(envelope.available){
    setStatus('judge-status',`Kimi live on ${envelope.model||state.model||'-'} for the current 15-minute bucket.`,'ok');
  }else if(envelope.error){
    setStatus('judge-status',`Kimi fallback is active: ${envelope.error}`,'warn');
  }else{
    setStatus('judge-status',`Kimi cache status: ${envelope.reason||'waiting for first refresh'}`,'warn');
  }
}

function renderPacket(){
  const latest=latestKimiEntry();
  if(!latest){
    setStatus('kimi-meta','No Kimi packet logged yet.');
    $('kimi-context').textContent='{}';
    $('kimi-glossary').innerHTML='<div class=\"tiny\">No numeric glossary available yet.</div>';
    renderGuide();
    return;
  }
  setStatus('kimi-meta',`${latest.request_kind||'packet'} | ${latest.model||'-'} | ${latest.packet_bucket_15m_utc||'-'} | ${latest.status||'-'}`,latest.status==='ok'?'ok':'warn');
  $('kimi-context').textContent=JSON.stringify(latest.context||{},null,2);
  const host=$('kimi-glossary');
  host.innerHTML='';
  const glossary=latest.numeric_glossary||{};
  const keys=Object.keys(glossary).slice(0,80);
  if(!keys.length){
    host.innerHTML='<div class=\"tiny\">No numeric glossary available.</div>';
  }else{
    keys.forEach((key)=>{
      const row=glossary[key]||{};
      const node=document.createElement('div');
      node.className='item';
      node.innerHTML=`<div class='label mono'>${key}</div><div class='value' style='font-size:18px'>${row.value??'-'}</div><div class='tiny'>${row.meaning||''}</div>`;
      host.appendChild(node);
    });
  }
  renderGuide();
}

function renderFeeds(){
  const feeds=((state.sim||{}).feeds)||{};
  const news=[...(feeds.news||[])].slice(0,6);
  const crowd=[...(feeds.public_discussions||[])].slice(0,6);
  const fear=feeds.fear_greed||{};
  const macro=feeds.macro||{};
  const sim=currentSimulation();
  const tech=currentTech();
  $('feed-structure').textContent=`${String(tech.structure||'-').toUpperCase()} / ${String(tech.location||'-').toUpperCase()}`;
  $('feed-structure-sub').textContent=`Support ${fmtPrice((tech.nearest_support||{}).price)} | Resistance ${fmtPrice((tech.nearest_resistance||{}).price)} | RSI ${fmtNum(tech.rsi_14,1)} | ATR ${fmtNum(tech.atr_14,2)}`;
  $('feed-crowd').textContent=`Crowd ${fmtSigned(sim.crowd_bias,3)} | Extreme ${fmtPct(sim.crowd_extreme)}`;
  $('feed-crowd-sub').textContent=`Fear/Greed ${fear.value ?? '-'} (${fear.classification||'unknown'}) | Macro ${fmtSigned(macro.macro_bias,3)} | Shock ${fmtPct(macro.macro_shock)}`;
  const host=$('feeds');
  host.innerHTML='';
  const cards=[];
  news.forEach((item)=>cards.push({kind:'News',title:cleanText(item.title)||'Untitled headline',meta:`${item.source||'unknown'} | sentiment ${fmtSigned(Number(item.sentiment||0)*100,0)}%`,tone:Number(item.sentiment||0)}));
  crowd.forEach((item)=>cards.push({kind:'Crowd',title:cleanText(item.title)||'Untitled discussion',meta:`${item.source||'unknown'} | sentiment ${fmtSigned(Number(item.sentiment||0)*100,0)}%`,tone:Number(item.sentiment||0)}));
  if(!cards.length){
    cards.push({kind:'Context',title:'No live headlines were available for this refresh.',meta:`Macro bias ${fmtSigned(macro.macro_bias,3)} | News bias ${fmtSigned(sim.news_bias,3)} | Crowd bias ${fmtSigned(sim.crowd_bias,3)}`,tone:0});
  }
  cards.forEach((item)=>{
    const color=item.tone>0.08?'var(--bull)':item.tone<-0.08?'var(--bear)':'var(--muted)';
    const node=document.createElement('div');
    node.className='news-item';
    node.innerHTML=`<div class='label'>${item.kind}</div><div class='news-title'>${item.title}</div><div class='news-meta' style='color:${color}'>${item.meta}</div>`;
    host.appendChild(node);
  });
}

function mergedOpenPositions(){
  const symbol=state.symbol.toUpperCase();
  const paperPositions=[...((currentPaper().open_positions)||[])].filter((item)=>String(item.symbol||'').toUpperCase()===symbol);
  const livePositions=[...((state.live.positions)||[])].filter((item)=>String(item.symbol||'').toUpperCase()===symbol);
  const byId=new Map();
  paperPositions.forEach((item)=>byId.set(String(item.trade_id||''),{...item}));
  livePositions.forEach((item)=>byId.set(String(item.trade_id||''),{...(byId.get(String(item.trade_id||''))||{}),...item}));
  return [...byId.values()];
}

function paperSummary(){
  const live=state.live.paper_summary||{};
  const paper=(currentPaper().summary)||{};
  return Object.keys(live).length?{...paper,...live}:paper;
}

function renderPaper(){
  const summary=paperSummary();
  const sim=currentSimulation();
  const manualLot=$('tlot').value.trim();
  $('pb').textContent=summary.balance==null?'-':`$${fmtNum(summary.balance,2)}`;
  $('pe').textContent=summary.equity==null?'-':`$${fmtNum(summary.equity,2)}`;
  $('pr').textContent=summary.realized_pnl==null?'-':fmtMoney(summary.realized_pnl);
  $('pr').style.color=Number(summary.realized_pnl||0)>=0?'var(--bull)':'var(--bear)';
  $('pu').textContent=summary.unrealized_pnl==null?'-':fmtMoney(summary.unrealized_pnl);
  $('pu').style.color=Number(summary.unrealized_pnl||0)>=0?'var(--bull)':'var(--bear)';
  $('trade-lot').textContent=manualLot?`${fmtNum(Number(manualLot),2)} lot (manual)`:(sim.suggested_lot==null?'-':`${fmtNum(sim.suggested_lot,2)} lot (auto)`);
  const positions=mergedOpenPositions();
  setPaperStatus(positions.length?`${positions.length} open position${positions.length===1?'':'s'} on ${state.symbol}.`:'No open paper positions on this symbol.',positions.length?'ok':'');
  const openBody=$('open-body');
  openBody.innerHTML='';
  if(!positions.length){
    openBody.innerHTML='<tr><td colspan=\"10\" class=\"tiny\">No open paper positions.</td></tr>';
  }else{
    positions.forEach((pos)=>{
      const statusLabel=pos.tp_hit?'TP hit':pos.sl_hit?'SL hit':'live';
      const tr=document.createElement('tr');
      tr.innerHTML=`<td class='mono'>${String(pos.trade_id||'').slice(0,8)}</td><td style='color:${tone(pos.direction)}'>${String(pos.direction||'-').toUpperCase()}</td><td>${fmtNum(pos.lot,2)}</td><td>${String(pos.lot_source||'auto')}</td><td>${fmtPrice(pos.entry_price)}</td><td style='color:${Number(pos.unrealized_pnl_usd||0)>=0?'var(--bull)':'var(--bear)'}'>${fmtMoney(pos.unrealized_pnl_usd)}</td><td><input data-sl='${pos.trade_id}' value='${pos.stop_loss==null?'':fmtNum(pos.stop_loss,2)}'></td><td><input data-tp='${pos.trade_id}' value='${pos.take_profit==null?'':fmtNum(pos.take_profit,2)}'></td><td class='tiny'>${statusLabel}</td><td><button data-mod='${pos.trade_id}' class='btn-secondary'>Save</button><button data-close='${pos.trade_id}' class='btn-secondary'>Close</button></td>`;
      openBody.appendChild(tr);
    });
  }
  const closedBody=$('closed-body');
  const closed=[...((currentPaper().closed_trades)||[])].reverse();
  closedBody.innerHTML='';
  if(!closed.length){
    closedBody.innerHTML='<tr><td colspan=\"6\" class=\"tiny\">No closed trades yet.</td></tr>';
  }else{
    closed.slice(0,120).forEach((item)=>{
      const tr=document.createElement('tr');
      tr.innerHTML=`<td>${new Date(item.exit_time||item.entry_time||Date.now()).toLocaleString()}</td><td style='color:${tone(item.direction)}'>${String(item.direction||'-').toUpperCase()}</td><td>${fmtNum(item.lot,2)}</td><td>${fmtPrice(item.entry_price)}</td><td>${fmtPrice(item.exit_price)}</td><td style='color:${Number(item.pnl_usd||0)>=0?'var(--bull)':'var(--bear)'}'>${fmtMoney(item.pnl_usd)}</td>`;
      closedBody.appendChild(tr);
    });
  }
  openBody.querySelectorAll('button[data-close]').forEach((button)=>{
    button.onclick=async()=>{try{await jsonFetch('/api/paper/close',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trade_id:button.getAttribute('data-close'),exit_price:Number(currentPrice()||0)})});await loadPaper();syncChart();setPaperStatus('Paper trade closed.','ok');}catch(error){setPaperStatus(`Close failed: ${error.message}`,'bad');}};
  });
  openBody.querySelectorAll('button[data-mod]').forEach((button)=>{
    button.onclick=async()=>{const id=button.getAttribute('data-mod');const sl=openBody.querySelector(`input[data-sl="${id}"]`);const tp=openBody.querySelector(`input[data-tp="${id}"]`);try{await jsonFetch('/api/paper/modify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trade_id:id,stop_loss:sl&&sl.value!==''?Number(sl.value):null,take_profit:tp&&tp.value!==''?Number(tp.value):null})});await loadPaper();syncChart();setPaperStatus(`Updated SL/TP for ${id}.`,'ok');}catch(error){setPaperStatus(`Modify failed: ${error.message}`,'bad');}};
  });
}

function chartCandles(){
  const realtime=((state.sim||{}).realtime_chart||{}).candles;
  const fallback=((state.sim||{}).market||{}).candles;
  const rows=(Array.isArray(realtime)&&realtime.length?realtime:(Array.isArray(fallback)?fallback:[]));
  return rows.map((row)=>({time:toUnix(row.timestamp),open:Number(row.open),high:Number(row.high),low:Number(row.low),close:Number(row.close)})).filter((row)=>row.time&&[row.open,row.high,row.low,row.close].every((value)=>Number.isFinite(value)));
}

function projectSeries(pathValues,forecastKey){
  const forecast=((state.sim||{}).final_forecast||{}).points||[];
  const candles=state.chart.baseCandles;
  if(!candles.length)return[];
  const baseTime=candles[candles.length-1].time;
  const points=[{time:baseTime,value:Number((pathValues||[])[0]??candles[candles.length-1].close)}];
  forecast.forEach((item,index)=>{
    const time=toUnix(item.timestamp)||baseTime+((index+1)*300);
    const pathIndex=index+1;
    const fallbackValue=forecastKey&&item[forecastKey]!=null?Number(item[forecastKey]):Number((pathValues||[])[pathIndex]);
    const value=Number((pathValues||[])[pathIndex]??fallbackValue);
    if(Number.isFinite(value))points.push({time,value});
  });
  return points;
}

function kimiSeries(){
  const projection=(currentJudgeEnvelope().projection_path||{}).points||[];
  return projection.map((item)=>({time:toUnix(item.timestamp),value:Number(item.price)})).filter((item)=>item.time&&Number.isFinite(item.value));
}

function ensureChart(){
  if(state.chart.instance)return;
  const host=$('chart');
  const chart=LightweightCharts.createChart(host,{width:host.clientWidth||900,height:host.clientHeight||520,layout:{background:{type:'solid',color:'#08131d'},textColor:'#eaf2fb',fontFamily:'JetBrains Mono, monospace'},grid:{vertLines:{color:'rgba(255,255,255,.05)'},horzLines:{color:'rgba(255,255,255,.05)'}},rightPriceScale:{borderColor:'rgba(255,255,255,.12)'},timeScale:{borderColor:'rgba(255,255,255,.12)',timeVisible:true,secondsVisible:false},crosshair:{mode:0}});
  const candles=chart.addCandlestickSeries({upColor:'#63e6ad',downColor:'#ff8f86',borderUpColor:'#63e6ad',borderDownColor:'#ff8f86',wickUpColor:'#63e6ad',wickDownColor:'#ff8f86'});
  const consensus=chart.addLineSeries({color:'#ffd38a',lineWidth:3});
  const minority=chart.addLineSeries({color:'#8ed8ff',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed});
  const outerUpper=chart.addLineSeries({color:'#ff8f86',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed});
  const outerLower=chart.addLineSeries({color:'#ff8f86',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed});
  const kimi=chart.addLineSeries({color:'#81f1b6',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dotted});
  const resizeObserver=new ResizeObserver(()=>chart.applyOptions({width:host.clientWidth||900,height:host.clientHeight||520}));
  resizeObserver.observe(host);
  state.chart={...state.chart,instance:chart,candles,consensus,minority,outerUpper,outerLower,kimi,resizeObserver};
}

function clearPriceLines(list){list.forEach((item)=>{try{item.series.removePriceLine(item.line);}catch(_err){}});return[];}

function syncChart(){
  ensureChart();
  const candles=chartCandles();
  if(!candles.length)return;
  const sim=currentSimulation();
  const judge=currentJudge();
  const timeScale=state.chart.instance.timeScale();
  const preserved=state.chart.seeded?timeScale.getVisibleLogicalRange():null;
  state.chart.baseCandles=candles.slice();
  state.chart.candles.setData(state.chart.baseCandles);
  state.chart.consensus.setData(projectSeries(sim.consensus_path,'final_price'));
  state.chart.minority.setData(projectSeries(sim.minority_path,'minority_price'));
  state.chart.outerUpper.setData(projectSeries(sim.cone_outer_upper,'outer_upper'));
  state.chart.outerLower.setData(projectSeries(sim.cone_outer_lower,'outer_lower'));
  state.chart.kimi.setData(kimiSeries());
  state.chart.entryLines=clearPriceLines(state.chart.entryLines);
  state.chart.positionLines=clearPriceLines(state.chart.positionLines);
  if(Array.isArray(judge.entry_zone)&&judge.entry_zone.length===2){
    const color=String(judge.stance||'').toUpperCase()==='SELL'?'#ff8f86':'#81f1b6';
    judge.entry_zone.forEach((price,index)=>{
      const line=state.chart.candles.createPriceLine({price:Number(price),color,lineWidth:2,axisLabelVisible:true,title:index===0?'Entry Lo':'Entry Hi'});
      state.chart.entryLines.push({series:state.chart.candles,line});
    });
  }
  mergedOpenPositions().slice(-8).forEach((position)=>{
    const entry=state.chart.candles.createPriceLine({price:Number(position.entry_price),color:'#ffd38a',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed,title:'Entry'});
    state.chart.positionLines.push({series:state.chart.candles,line:entry});
    if(position.stop_loss!=null){
      const stop=state.chart.candles.createPriceLine({price:Number(position.stop_loss),color:'#ff8f86',lineWidth:1,title:'SL'});
      state.chart.positionLines.push({series:state.chart.candles,line:stop});
    }
    if(position.take_profit!=null){
      const tp=state.chart.candles.createPriceLine({price:Number(position.take_profit),color:'#63e6ad',lineWidth:1,title:'TP'});
      state.chart.positionLines.push({series:state.chart.candles,line:tp});
    }
  });
  if(preserved&&Number.isFinite(preserved.from)&&Number.isFinite(preserved.to)){
    timeScale.setVisibleLogicalRange(preserved);
  }else if(!state.chart.seeded){
    timeScale.fitContent();
    state.chart.seeded=true;
  }
  $('chart-tier').textContent=String(sim.confidence_tier||'-').replaceAll('_',' ').toUpperCase();
  $('chart-cone').textContent=sim.cone_width_pips==null?'-':`${fmtNum(sim.cone_width_pips,1)} pips`;
  $('chart-mfg').textContent=fmtNum((((state.sim||{}).mfg)||{}).disagreement,6);
}

function applyLiveTickToChart(){
  if(!state.chart.instance||!state.chart.baseCandles.length)return;
  const price=Number(currentPrice());
  const tickTime=toUnix(state.live.timestamp)||Math.floor(Date.now()/1000);
  if(!Number.isFinite(price)||price<=0||!tickTime)return;
  const bucket=tickTime-(tickTime%60);
  const candles=state.chart.baseCandles;
  const last=candles[candles.length-1];
  if(!last)return;
  if(bucket>last.time){
    const next={time:bucket,open:last.close,high:Math.max(last.close,price),low:Math.min(last.close,price),close:price};
    candles.push(next);
    while(candles.length>320)candles.shift();
    state.chart.candles.setData(candles);
  }else{
    last.high=Math.max(last.high,price);
    last.low=Math.min(last.low,price);
    last.close=price;
    state.chart.candles.update({...last});
  }
}

async function loadPaper(){state.paper=await jsonFetch(`/api/paper/state?symbol=${encodeURIComponent(state.symbol)}`);renderPaper();}
async function loadKimiLog(){state.kimiLog=await jsonFetch('/api/llm/kimi-log?limit=12');renderPacket();renderJudge();}

async function loadKimi(force=false){
  const query=new URLSearchParams({symbol:state.symbol,mode:state.mode,llm_provider:state.provider});
  if(state.model)query.set('llm_model',state.model);
  if(force)query.set('force','1');
  try{
    const payload=await jsonFetch(`/api/llm/kimi-live?${query.toString()}`);
    state.sim=state.sim?{...state.sim,kimi_judge:payload.kimi_judge}:{kimi_judge:payload.kimi_judge};
    renderDecisionBar();
    renderJudge();
    syncChart();
    await loadKimiLog();
    const envelope=currentJudgeEnvelope();
    if(envelope.available){setStatusLine(`Kimi refreshed on ${envelope.model||state.model||'-'}.`,'ok');}
    else{setStatusLine(`Kimi fallback active: ${envelope.error||envelope.reason||'remote unavailable'}`,'warn');}
  }catch(error){
    renderJudge();
    setStatusLine(`Kimi refresh failed: ${error.message}`,'warn');
  }
}

async function refreshDesk(announce=true){
  if(state.simLoading)return;
  state.simLoading=true;
  $('run').disabled=true;
  state.symbol=$('symbol').value;
  state.mode=$('mode').value;
  state.provider=$('provider').value;
  state.model=$('model').value.trim();
  try{
    if(announce)setStatusLine(`Running V18 for ${state.symbol}...`);
    const query=new URLSearchParams({symbol:state.symbol,mode:state.mode,llm_provider:state.provider});
    if(state.model)query.set('llm_model',state.model);
    const payload=await jsonFetch(`/api/dashboard/live?${query.toString()}`);
    state.sim=payload;
    if(payload.paper_trading)state.paper=payload.paper_trading;
    renderDecisionBar();
    renderJudge();
    renderFeeds();
    renderPaper();
    renderPacket();
    syncChart();
    await loadKimi(false);
  }catch(error){
    setStatusLine(`Simulation failed: ${error.message}`,'bad');
  }finally{
    state.simLoading=false;
    $('run').disabled=false;
  }
}

function applyKimiSuggestion(){
  const judge=currentJudge();
  const stance=String(judge.stance||'HOLD').toUpperCase();
  if(!['BUY','SELL'].includes(stance)){setStatusLine(`Kimi is skipping this bar: ${judge.final_summary||judge.reasoning||'No actionable Kimi stance for this bar.'}`,'warn');return;}
  $('tdir').value=stance;
  const zone=Array.isArray(judge.entry_zone)&&judge.entry_zone.length===2?judge.entry_zone:[];
  const mid=zone.length===2?(Number(zone[0])+Number(zone[1]))/2:Number(currentPrice()||0);
  const pip=PIP_SIZES[state.symbol]||0.1;
  if(judge.stop_loss!=null)$('tstop').value=Math.max(1,Math.abs(mid-Number(judge.stop_loss))/pip).toFixed(1);
  if(judge.take_profit!=null)$('ttp').value=Math.max(1,Math.abs(Number(judge.take_profit)-mid)/pip).toFixed(1);
  $('trade-kimi').textContent=`Applied ${stance} | final call ${judge.final_call||stance} | zone ${zone.length===2?`${fmtPrice(zone[0])} - ${fmtPrice(zone[1])}`:'-'} | TP ${fmtPrice(judge.take_profit)} | SL ${fmtPrice(judge.stop_loss)}`;
  renderPaper();
  setStatusLine(`Applied ${stance} Kimi setup to the paper form.`,'ok');
}

async function openTrade(){
  const sim=currentSimulation();
  const judge=currentJudge();
  const manualLot=$('tlot').value.trim();
  const payload={
    symbol:state.symbol,
    direction:$('tdir').value,
    entry_price:Number(currentPrice()||0),
    confidence_tier:String(sim.confidence_tier||'low'),
    sqt_label:String(((state.live.sqt||{}).label)||sim.sqt_label||'NEUTRAL'),
    mode:state.mode,
    leverage:Number($('tlev').value||200),
    stop_pips:Number($('tstop').value||20),
    take_profit_pips:Number($('ttp').value||30),
    stop_loss:judge.stop_loss!=null?Number(judge.stop_loss):null,
    take_profit:judge.take_profit!=null?Number(judge.take_profit):null,
    manual_lot:manualLot?Number(manualLot):null,
    note:$('tnote').value,
  };
  if(!payload.entry_price){setPaperStatus('No live price is available yet.','warn');return;}
  await jsonFetch('/api/paper/open',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  await loadPaper();
  syncChart();
  setPaperStatus(`Opened ${payload.direction} paper trade on ${payload.symbol}.`,'ok');
}

async function resetPaper(){
  await jsonFetch('/api/paper/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({starting_balance:1000})});
  await loadPaper();
  syncChart();
  setPaperStatus('Paper account reset to $1000.','ok');
}

function connectWs(){
  if(state.ws){try{state.ws.close();}catch(_err){}}
  const socket=new WebSocket(`${wsOrigin()}/ws/live?symbol=${encodeURIComponent(state.symbol)}`);
  state.ws=socket;
  state.live.connected=false;
  updateHealth();
  socket.onopen=()=>{state.live.connected=true;state.live.lastMessageAt=Date.now();try{socket.send(state.symbol);}catch(_err){}renderDecisionBar();};
  socket.onmessage=(event)=>{
    try{
      const payload=JSON.parse(event.data);
      const prevCount=(state.live.positions||[]).length;
      state.live={...state.live,...payload,connected:true,lastMessageAt:Date.now()};
      renderDecisionBar();
      renderPaper();
      applyLiveTickToChart();
      if(((state.live.positions||[]).length)!==prevCount)syncChart();
    }catch(error){
      setStatusLine(`WebSocket payload error: ${error.message}`,'warn');
    }
  };
  socket.onerror=()=>{state.live.connected=false;updateHealth();};
  socket.onclose=()=>{state.live.connected=false;updateHealth();setTimeout(()=>connectWs(),1500);};
}

function simulationLoop(){
  if(state.timer)clearInterval(state.timer);
  if($('auto').value!=='on')return;
  const seconds=Math.max(10,Number($('refresh').value||15));
  state.timer=setInterval(()=>refreshDesk(false),seconds*1000);
}

$('run').addEventListener('click',()=>refreshDesk(true));
$('apply-kimi').addEventListener('click',applyKimiSuggestion);
$('open').addEventListener('click',()=>openTrade().catch((error)=>setPaperStatus(`Open failed: ${error.message}`,'bad')));
$('reset').addEventListener('click',()=>resetPaper().catch((error)=>setPaperStatus(`Reset failed: ${error.message}`,'bad')));
$('refresh').addEventListener('change',simulationLoop);
$('auto').addEventListener('change',simulationLoop);
$('tlot').addEventListener('input',()=>{renderDecisionBar();renderPaper();});
['symbol','mode','provider','model'].forEach((id)=>$(id).addEventListener('change',async()=>{if(id==='symbol')state.chart.seeded=false;state.symbol=$('symbol').value;connectWs();await refreshDesk(true);simulationLoop();}));

window.addEventListener('load',async()=>{
  updateClock();
  setInterval(updateClock,1000);
  ensureChart();
  connectWs();
  try{
    await loadPaper();
    await loadKimiLog();
    await refreshDesk(false);
  }catch(error){
    setStatusLine(`Initial load failed: ${error.message}`,'bad');
  }
  simulationLoop();
});
</script>
</body>
</html>
"""


def render_web_app_html() -> str:
    return _HTML_HEAD + _HTML_SCRIPT
