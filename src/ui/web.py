from __future__ import annotations


def render_web_app_html() -> str:
    return """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Nexus Trader V18</title>
<script src='https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js'></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;700&display=swap');
:root{--bg:#02060b;--bg2:#071018;--line:rgba(177,214,255,.14);--text:#edf6ff;--muted:#89a1b8;--bull:#5ce7af;--bear:#ff9387;--amber:#ffd48a;--cyan:#8bd9ff;--shadow:0 26px 80px rgba(0,0,0,.58),inset 1px 1px 0 rgba(255,255,255,.04),inset -18px -18px 30px rgba(0,0,0,.22);--neo:inset 12px 12px 24px rgba(0,0,0,.24),inset -10px -10px 18px rgba(255,255,255,.018)}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;color:var(--text);font-family:'Space Grotesk',sans-serif;background:radial-gradient(circle at 14% 16%,rgba(139,217,255,.07),transparent 18%),radial-gradient(circle at 82% 10%,rgba(255,212,138,.08),transparent 20%),radial-gradient(circle at 72% 74%,rgba(92,231,175,.05),transparent 24%),linear-gradient(165deg,#010205 0%,#02060c 32%,#05101a 68%,#02060b 100%);min-height:100vh}
body::before{content:'';position:fixed;inset:0;pointer-events:none;background:linear-gradient(transparent 49%,rgba(255,255,255,.016) 50%,transparent 51%),linear-gradient(90deg,transparent 49%,rgba(255,255,255,.012) 50%,transparent 51%);background-size:100% 20px,20px 100%;opacity:.12}
#app{max-width:1720px;margin:0 auto;padding:18px;display:grid;gap:14px}.grid{display:grid;gap:14px;min-width:0}.top-grid{grid-template-columns:minmax(0,1.45fr) minmax(360px,.9fr)}.bottom-grid{grid-template-columns:minmax(420px,1.1fr) minmax(300px,.82fr) minmax(320px,.95fr)}
.panel{min-width:0;overflow:hidden;border-radius:28px;border:1px solid var(--line);background:linear-gradient(180deg,rgba(17,31,46,.82),rgba(7,16,24,.95));backdrop-filter:blur(22px) saturate(128%);-webkit-backdrop-filter:blur(22px) saturate(128%);box-shadow:var(--shadow),var(--neo)}
.panel-inner{padding:18px}.panel-body{display:flex;flex-direction:column;gap:12px;min-height:0;height:100%}.panel-scroll{overflow:auto;min-height:0;flex:1 1 auto;padding-right:4px}.panel-scroll::-webkit-scrollbar,.table-wrap::-webkit-scrollbar{width:10px;height:10px}.panel-scroll::-webkit-scrollbar-thumb,.table-wrap::-webkit-scrollbar-thumb{background:rgba(139,217,255,.18);border-radius:999px}
.hero{min-height:184px}.h-chart,.h-judge{height:min(72vh,780px)}.h-paper,.h-feed,.h-packet{height:min(60vh,660px)}
.section-title{margin:0;font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:var(--amber)}.headline{margin:0;font-size:clamp(30px,4.7vw,56px);line-height:.95;letter-spacing:-.055em}.muted{color:var(--muted)}.mono{font-family:'JetBrains Mono',monospace}
.row{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}.stack{display:grid;gap:12px;min-width:0}.split{display:grid;gap:12px;grid-template-columns:repeat(2,minmax(0,1fr))}.triple{display:grid;gap:12px;grid-template-columns:repeat(3,minmax(0,1fr))}.quad{display:grid;gap:12px;grid-template-columns:repeat(4,minmax(0,1fr))}.controls{display:grid;gap:12px;grid-template-columns:repeat(7,minmax(0,1fr))}
.stat,.item,.news-item{padding:14px;border-radius:22px;border:1px solid rgba(255,255,255,.08);background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.028));box-shadow:var(--neo)}.label{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}.value{font-size:26px;font-weight:700;line-height:1.05}.value-big{font-size:44px}.tiny{font-size:12px;line-height:1.5;color:var(--muted)}
.status{padding:12px 14px;border-radius:18px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.05);color:var(--muted)}.status.ok{color:var(--bull)}.status.warn{color:var(--amber)}.status.bad{color:var(--bear)}
.badge,.pill{display:inline-flex;align-items:center;gap:8px;padding:10px 14px;border-radius:999px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.05);font-size:12px;letter-spacing:.08em;text-transform:uppercase}
.decision-shell{display:grid;gap:14px;grid-template-columns:minmax(0,1.4fr) minmax(300px,.7fr)}.progress-wrap{height:18px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);overflow:hidden;box-shadow:var(--neo)}.progress-fill{height:100%;width:0%;background:linear-gradient(90deg,rgba(139,217,255,.52),rgba(92,231,175,.78));box-shadow:0 0 22px rgba(92,231,175,.24);transition:width .18s ease}
select,input,textarea,button{width:100%;border-radius:16px;border:1px solid rgba(255,255,255,.10);background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.03));color:var(--text);padding:11px 12px;font:inherit;box-shadow:var(--neo)}textarea{min-height:86px;resize:vertical}
button{cursor:pointer;font-weight:700;background:linear-gradient(135deg,rgba(139,217,255,.16),rgba(255,212,138,.12)),linear-gradient(180deg,rgba(255,255,255,.08),rgba(255,255,255,.03))}button:hover{transform:translateY(-1px)}.btn-secondary{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.025))}
.judge-hero{padding:16px;border-radius:24px;background:linear-gradient(180deg,rgba(255,255,255,.07),rgba(255,255,255,.03));border:1px solid rgba(255,255,255,.08)}.judge-grid{display:grid;gap:12px;grid-template-columns:repeat(4,minmax(0,1fr))}
.table-wrap{overflow:auto;border-radius:20px;border:1px solid rgba(255,255,255,.06);max-height:220px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 9px;text-align:left;border-bottom:1px solid rgba(255,255,255,.06);vertical-align:top}th{position:sticky;top:0;background:rgba(5,12,19,.96);z-index:1}
details{border:1px solid rgba(255,255,255,.08);border-radius:18px;background:rgba(255,255,255,.03);overflow:hidden}summary{cursor:pointer;padding:12px 14px;color:var(--amber);font-size:12px;letter-spacing:.12em;text-transform:uppercase}pre{margin:0;padding:14px;white-space:pre-wrap;word-break:break-word;font:12px/1.55 'JetBrains Mono',monospace;color:#d7edff}
.collapse-copy{padding:0 14px 14px}.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.10),transparent)}.news-title{font-weight:700;line-height:1.45}.news-meta{margin-top:6px;font-size:12px;color:var(--muted)}#chart{height:100%;min-height:460px}.line-clamp{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
@media(max-width:1450px){.top-grid,.bottom-grid{grid-template-columns:1fr}.controls{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:900px){#app{padding:12px}.decision-shell,.split,.triple,.quad,.judge-grid,.controls{grid-template-columns:1fr}.h-chart,.h-judge,.h-paper,.h-feed,.h-packet{height:auto;min-height:unset}.panel-scroll{max-height:none}}
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
          <div class='muted'>Realtime charting, a working Kimi/NIM route, live paper-trade updates, and scroll-safe glass panels for the full V18 desk.</div>
        </div>
        <div class='stack' style='min-width:280px;text-align:right'>
          <div id='clock' class='value value-big mono'>--:--:--</div>
          <div id='date' class='tiny'>Waiting for local time</div>
          <div id='health' class='status'>Starting realtime services...</div>
        </div>
      </div>
      <div class='decision-shell'>
        <div class='stack'>
          <div class='row'>
            <div class='badge' id='decision-stance'>HOLD</div>
            <div class='badge' id='decision-action'>WAIT</div>
            <div class='badge'><span class='mono' id='decision-symbol'>XAUUSD</span> / <span id='decision-mode'>Frequency</span></div>
          </div>
          <div class='progress-wrap'><div id='bar-fill' class='progress-fill'></div></div>
          <div class='triple'>
            <div class='stat'><div class='label'>Bar Countdown</div><div id='bar-countdown' class='value mono'>15m 00s</div><div class='tiny'>Seconds remaining in the current 15-minute decision bar.</div></div>
            <div class='stat'><div class='label'>Entry Zone</div><div id='decision-entry' class='value'>-</div><div class='tiny'>Preferred Kimi trigger band for the active bar.</div></div>
            <div class='stat'><div class='label'>TP / SL</div><div id='decision-targets' class='value'>-</div><div class='tiny'>Take-profit and invalidation levels for the current idea.</div></div>
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
            <div class='muted'>Lightweight live chart with 1-minute candles, consensus and minority paths, outer cone rails, and trade-level price markers.</div>
          </div>
          <div class='row'>
            <div class='pill'>Tier <strong id='chart-tier'>-</strong></div>
            <div class='pill'>Cone <strong id='chart-cone'>-</strong></div>
            <div class='pill'>MFG <strong id='chart-mfg'>-</strong></div>
          </div>
        </div>
        <div class='panel-scroll'><div id='chart'></div></div>
      </div>
    </section>

    <section class='panel h-judge'>
      <div class='panel-inner panel-body'>
        <div class='row'>
          <div>
            <div class='section-title'>AI Judge</div>
            <div class='muted'>The Kimi/NIM decision layer for the current 15-minute window, with transparent fallback status when the remote model fails.</div>
          </div>
          <button id='apply-kimi' class='btn-secondary' style='max-width:220px'>Apply To Trade Form</button>
        </div>
        <div id='judge-status' class='status'>Waiting for the first Kimi decision.</div>
        <div class='judge-hero'>
          <div class='row'>
            <div class='badge' id='judge-stance'>HOLD</div>
            <div class='badge' id='judge-confidence'>VERY_LOW</div>
            <div class='badge' id='judge-rr'>R:R -</div>
          </div>
          <div class='judge-grid' style='margin-top:12px'>
            <div class='stat'><div class='label'>Entry Zone</div><div id='judge-entry' class='value'>-</div></div>
            <div class='stat'><div class='label'>Stop Loss</div><div id='judge-sl' class='value'>-</div></div>
            <div class='stat'><div class='label'>Take Profit</div><div id='judge-tp' class='value'>-</div></div>
            <div class='stat'><div class='label'>Hold Time</div><div id='judge-hold' class='value'>-</div></div>
          </div>
        </div>
        <div class='panel-scroll stack'>
          <div class='item'><div class='label'>Reasoning</div><div id='judge-reasoning' class='line-clamp'>Waiting for Kimi.</div></div>
          <div class='item'><div class='label'>Key Risk</div><div id='judge-risk'>Waiting for Kimi.</div></div>
          <div class='item'><div class='label'>Crowd Note</div><div id='judge-crowd'>Waiting for Kimi.</div></div>
          <div class='item'><div class='label'>Regime Note</div><div id='judge-regime'>Waiting for Kimi.</div></div>
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
          <div class='muted'>Realtime summary, scroll-safe position history, and inline SL/TP editing without the panel breaking its layout.</div>
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
          <div class='split'>
            <div><div class='label'>Stop Pips</div><input id='tstop' type='number' min='1' step='0.1' value='20'></div>
            <div><div class='label'>Take Profit Pips</div><input id='ttp' type='number' min='1' step='0.1' value='30'></div>
          </div>
          <div class='split'>
            <div class='item'><div class='label'>Suggested Lot</div><div id='trade-lot' class='value'>-</div><div class='tiny'>Equity-scaled lot size after leverage and confidence controls.</div></div>
            <div class='item'><div class='label'>Kimi Setup</div><div id='trade-kimi' class='tiny'>Apply the current Kimi decision to prefill direction and distances.</div></div>
          </div>
          <div><div class='label'>Trade Note</div><textarea id='tnote' placeholder='Why this paper trade exists'></textarea></div>
          <div class='split'><button id='open'>Open Paper Trade</button><button id='reset' class='btn-secondary'>Reset Paper Account</button></div>
          <div class='divider'></div>
          <div class='section-title'>Open Positions</div>
          <div class='table-wrap'><table><thead><tr><th>ID</th><th>Side</th><th>Lot</th><th>Entry</th><th>PnL</th><th>SL</th><th>TP</th><th>Status</th><th></th></tr></thead><tbody id='open-body'></tbody></table></div>
          <div class='section-title'>Closed History</div>
          <div class='table-wrap'><table><thead><tr><th>Time</th><th>Side</th><th>Lot</th><th>Entry</th><th>Exit</th><th>PnL</th></tr></thead><tbody id='closed-body'></tbody></table></div>
        </div>
      </div>
    </section>

    <section class='panel h-feed'>
      <div class='panel-inner panel-body'>
        <div>
          <div class='section-title'>News + Structure</div>
          <div class='muted'>Clean headline cards plus a compact live read on structure, volatility, and crowd state.</div>
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
          <div class='muted'>Latest packet metadata, numeric glossary, and a direct explainer for the core desk metrics.</div>
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
const PIP_SIZES={XAUUSD:0.1,EURUSD:0.0001,BTCUSD:1};
const state={symbol:'XAUUSD',mode:'frequency',provider:'nvidia_nim',model:'moonshotai/kimi-k2-instruct',sim:null,paper:null,kimiLog:null,live:{price:null,positions:[],paper_summary:{},bar_countdown:900,bar_progress:0,sqt:{},connected:false,lastMessageAt:0,timestamp:null},ws:null,timer:null,simLoading:false,chart:{instance:null,resizeObserver:null,candleSeries:null,consensusSeries:null,minoritySeries:null,upperSeries:null,lowerSeries:null,entryLines:[],positionLines:[],baseCandles:[]}};
const $=(id)=>document.getElementById(id);
const tone=(value)=>{const v=String(value||'').toUpperCase();if(v==='BUY'||v==='BULLISH'||v==='HOT'||v==='GOOD')return'var(--bull)';if(v==='SELL'||v==='BEARISH'||v==='BAD')return'var(--bear)';return'var(--amber)'};
const fmtPrice=(v)=>v==null||Number.isNaN(Number(v))?'-':Number(v).toFixed(2);
const fmtNum=(v,d=2)=>v==null||Number.isNaN(Number(v))?'-':Number(v).toFixed(d);
const fmtPct=(v)=>v==null||Number.isNaN(Number(v))?'-':`${(Number(v)*100).toFixed(1)}%`;
const fmtSigned=(v,d=2)=>v==null||Number.isNaN(Number(v))?'-':`${Number(v)>=0?'+':''}${Number(v).toFixed(d)}`;
const fmtMoney=(v)=>v==null||Number.isNaN(Number(v))?'-':`$${fmtSigned(Number(v),2)}`;
const fmtTime=(v)=>{if(!v)return'-';const d=new Date(v);return Number.isNaN(d.getTime())?String(v):d.toLocaleString();};
const cleanHeadline=(raw)=>String(raw||'').replace(/<[^>]*>/g,' ').replace(/&nbsp;/g,' ').replace(/\\s+/g,' ').trim();
const status=(text,kind='')=>{const el=$('status');el.textContent=text;el.className=`status${kind?` ${kind}`:''}`;};
const wsOrigin=()=>window.location.origin.replace(/^http/i,'ws');
const toUnix=(value)=>{if(value==null)return null;if(typeof value==='number')return Math.floor(value);const d=new Date(value);if(Number.isNaN(d.getTime()))return null;return Math.floor(d.getTime()/1000);};
const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
const currentSimulation=()=>(((state.sim||{}).v16||{}).simulation||((state.sim||{}).simulation)||{});
const currentMarket=()=>((state.sim||{}).market||{});
const currentTech=()=>((state.sim||{}).technical_analysis||{});
const currentJudgeEnvelope=()=>((state.sim||{}).kimi_judge||{});
const currentJudge=()=>currentJudgeEnvelope().content||{};
const currentPaper=()=>state.paper||((state.sim||{}).paper_trading)||{};
async function jsonFetch(url,opt){const response=await fetch(url,opt);if(!response.ok){let message=`Request failed: ${response.status}`;try{const text=await response.text();if(text)message=text;}catch(_err){}throw new Error(message);}return response.json();}
function updateClock(){const now=new Date();$('clock').textContent=now.toLocaleTimeString();$('date').textContent=now.toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric',year:'numeric'});}
function decisionAction(judge,price){const stance=String(judge.stance||'HOLD').toUpperCase();const zone=Array.isArray(judge.entry_zone)?judge.entry_zone:[];if(stance==='HOLD')return'WAIT';if(zone.length===2&&price!=null){const low=Math.min(Number(zone[0]),Number(zone[1]));const high=Math.max(Number(zone[0]),Number(zone[1]));if(Number(price)>=low&&Number(price)<=high)return'ENTER';}return'WAIT';}
function rrRatio(judge){const zone=Array.isArray(judge.entry_zone)?judge.entry_zone:[];if(zone.length!==2||judge.stop_loss==null||judge.take_profit==null)return'-';const mid=(Number(zone[0])+Number(zone[1]))/2;const risk=Math.abs(mid-Number(judge.stop_loss));const reward=Math.abs(Number(judge.take_profit)-mid);if(!risk)return'-';return`${(reward/risk).toFixed(2)}R`;}
function buildGuideItems(){const sim=currentSimulation();const tech=currentTech();const mfg=(state.sim||{}).mfg||{};return[{label:'CABR',value:fmtPct(sim.cabr_score),meaning:'Branch-ranking strength for the selected path.'},{label:'CPM',value:fmtPct(sim.cpm_score),meaning:'Local predictability score for the current bar state.'},{label:'SQT',value:fmtPct((state.live.sqt||{}).rolling_accuracy),meaning:'Recent simulator hit rate. COLD means recent misses.'},{label:'Hurst Asym',value:fmtNum(sim.hurst_asymmetry,3),meaning:'Positive means upside persistence is stronger; negative means downside sticks more.'},{label:'Cone Width',value:sim.cone_width_pips==null?'-':`${fmtNum(sim.cone_width_pips,1)} pips`,meaning:'Inner cone width in pips. Wider means less certainty.'},{label:'Retail T',value:fmtNum(((sim.testosterone_index||{}).retail),3),meaning:'Retail winner-cycle intensity from WLTC.'},{label:'ATR(14)',value:fmtNum(tech.atr_14,2),meaning:'Fourteen-bar average true range.'},{label:'MFG Disagreement',value:fmtNum(mfg.disagreement,6),meaning:'Cross-persona disagreement in the mean-field crowd model.'},{label:'MFG Drift',value:fmtNum(mfg.consensus_drift,6),meaning:'Mean-field crowd drift implied by persona belief states.'}];}
function renderHealth(){const live=state.live.connected&&Date.now()-Number(state.live.lastMessageAt||0)<6000;const accuracy=(state.live.sqt||{}).rolling_accuracy;const age=state.live.lastMessageAt?Math.max(0,Math.floor((Date.now()-state.live.lastMessageAt)/1000)):null;$('health').textContent=`WebSocket ${live?'live':'offline'} | ${age==null?'-':`${age}s ago`} | SQT ${fmtPct(accuracy)}`;$('health').className=`status ${live?'ok':'bad'}`.trim();}
function renderDecisionBar(){const sim=currentSimulation();const judge=currentJudge();const price=state.live.price??currentMarket().current_price;const stance=String(judge.stance||'HOLD').toUpperCase();const action=decisionAction(judge,price);$('decision-stance').textContent=stance;$('decision-stance').style.color=tone(stance);$('decision-action').textContent=action;$('decision-action').style.color=action==='ENTER'?'var(--bull)':'var(--amber)';$('decision-symbol').textContent=state.symbol;$('decision-mode').textContent=state.mode.charAt(0).toUpperCase()+state.mode.slice(1);$('hero-price').textContent=fmtPrice(price);$('hero-cabr').textContent=fmtPct(sim.cabr_score);$('hero-cabr').style.color=tone(sim.direction);const sqtLabel=String((state.live.sqt||{}).label||((state.sim||{}).sqt||{}).label||sim.sqt_label||'-');$('hero-sqt').textContent=sqtLabel;$('hero-sqt').style.color=tone(sqtLabel);$('hero-hurst').textContent=fmtNum(sim.hurst_asymmetry,3);$('hero-hurst').style.color=Number(sim.hurst_asymmetry||0)>=0?'var(--bull)':'var(--bear)';$('hero-lot').textContent=sim.suggested_lot==null?'-':`${fmtNum(sim.suggested_lot,2)} lot`;$('decision-entry').textContent=Array.isArray(judge.entry_zone)&&judge.entry_zone.length===2?`${fmtPrice(judge.entry_zone[0])} - ${fmtPrice(judge.entry_zone[1])}`:'-';$('decision-targets').textContent=judge.take_profit!=null&&judge.stop_loss!=null?`TP ${fmtPrice(judge.take_profit)} | SL ${fmtPrice(judge.stop_loss)}`:'-';const remaining=Number(state.live.bar_countdown||900);$('bar-countdown').textContent=`${Math.floor(remaining/60)}m ${String(remaining%60).padStart(2,'0')}s`;$('bar-fill').style.width=`${clamp(Number((state.live.bar_progress||0)*100),0,100)}%`;renderHealth();}
function renderJudge(){const envelope=currentJudgeEnvelope();const judge=currentJudge();const stance=String(judge.stance||'HOLD').toUpperCase();$('judge-stance').textContent=stance;$('judge-stance').style.color=tone(stance);$('judge-confidence').textContent=String(judge.confidence||'VERY_LOW').toUpperCase();$('judge-rr').textContent=`R:R ${rrRatio(judge)}`;$('judge-entry').textContent=Array.isArray(judge.entry_zone)&&judge.entry_zone.length===2?`${fmtPrice(judge.entry_zone[0])} - ${fmtPrice(judge.entry_zone[1])}`:'-';$('judge-sl').textContent=judge.stop_loss==null?'-':fmtPrice(judge.stop_loss);$('judge-tp').textContent=judge.take_profit==null?'-':fmtPrice(judge.take_profit);$('judge-hold').textContent=String(judge.hold_time||'-');$('judge-reasoning').textContent=String(judge.reasoning||'No Kimi reasoning yet.');$('judge-risk').textContent=String(judge.key_risk||'No risk note yet.');$('judge-crowd').textContent=String(judge.crowd_note||'No crowd note yet.');$('judge-regime').textContent=String(judge.regime_note||'No regime note yet.');$('judge-raw').textContent=JSON.stringify(judge,null,2);const latest=(((state.kimiLog||{}).entries)||[]).slice(-1)[0]||{};$('judge-context').textContent=JSON.stringify(latest.context||{},null,2);const remoteOk=Boolean(envelope.available);const modelInfo=envelope.model||latest.model||state.model||'-';$('judge-status').textContent=remoteOk?`Kimi live | ${modelInfo}`:`Kimi fallback active | ${modelInfo} | ${String(envelope.error||latest.error||envelope.reason||'remote unavailable')}`;$('judge-status').className=`status ${remoteOk?'ok':'warn'}`;}
function chartCandlesFromState(){const realtime=((state.sim||{}).realtime_chart||{}).candles;const candles=(Array.isArray(realtime)&&realtime.length?realtime:((state.sim||{}).market||{}).candles)||[];return candles.map((row)=>({time:toUnix(row.timestamp),open:Number(row.open),high:Number(row.high),low:Number(row.low),close:Number(row.close)})).filter((row)=>row.time&&[row.open,row.high,row.low,row.close].every((v)=>Number.isFinite(v)));}
function forecastLine(values,forecast,lastClose){const output=[];const baseTime=state.chart.baseCandles.length?state.chart.baseCandles[state.chart.baseCandles.length-1].time:null;if(baseTime==null)return output;if(Array.isArray(values)&&values.length){output.push({time:baseTime,value:Number(values[0]??lastClose)});forecast.forEach((point,index)=>{const time=toUnix(point.timestamp)||baseTime+((index+1)*300);const value=values[index+1];if(Number.isFinite(Number(value)))output.push({time,value:Number(value)});});}return output;}
function ensureChart(){if(state.chart.instance)return;const host=$('chart');const chart=LightweightCharts.createChart(host,{width:host.clientWidth||800,height:host.clientHeight||520,layout:{background:{type:'solid',color:'#08121c'},textColor:'#edf6ff',fontFamily:'JetBrains Mono, monospace'},grid:{vertLines:{color:'rgba(255,255,255,.05)'},horzLines:{color:'rgba(255,255,255,.05)'}},rightPriceScale:{borderColor:'rgba(255,255,255,.10)'},timeScale:{borderColor:'rgba(255,255,255,.10)',timeVisible:true,secondsVisible:false},crosshair:{mode:0}});const candleSeries=chart.addCandlestickSeries({upColor:'#5ce7af',downColor:'#ff9387',borderUpColor:'#5ce7af',borderDownColor:'#ff9387',wickUpColor:'#5ce7af',wickDownColor:'#ff9387'});const consensusSeries=chart.addLineSeries({color:'#ffd48a',lineWidth:3});const minoritySeries=chart.addLineSeries({color:'#8bd9ff',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed});const upperSeries=chart.addLineSeries({color:'#ff9387',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed});const lowerSeries=chart.addLineSeries({color:'#ff9387',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed});const resizeObserver=new ResizeObserver(()=>chart.applyOptions({width:host.clientWidth||800,height:host.clientHeight||520}));resizeObserver.observe(host);state.chart={...state.chart,instance:chart,resizeObserver,candleSeries,consensusSeries,minoritySeries,upperSeries,lowerSeries};}
function clearPriceLines(lines){lines.forEach((item)=>{try{item.series.removePriceLine(item.line);}catch(_err){}});return[];}
function syncChart(){ensureChart();const sim=currentSimulation();const judge=currentJudge();const forecast=((state.sim||{}).final_forecast||{}).points||[];const candles=chartCandlesFromState();if(!candles.length)return;state.chart.baseCandles=candles.slice();state.chart.candleSeries.setData(state.chart.baseCandles);const lastClose=state.chart.baseCandles[state.chart.baseCandles.length-1].close;state.chart.consensusSeries.setData(forecastLine(sim.consensus_path||[],forecast,lastClose));state.chart.minoritySeries.setData(forecastLine(sim.minority_path||[],forecast,lastClose));state.chart.upperSeries.setData(forecastLine(sim.cone_outer_upper||[],forecast,lastClose));state.chart.lowerSeries.setData(forecastLine(sim.cone_outer_lower||[],forecast,lastClose));state.chart.entryLines=clearPriceLines(state.chart.entryLines);state.chart.positionLines=clearPriceLines(state.chart.positionLines);if(Array.isArray(judge.entry_zone)&&judge.entry_zone.length===2){const entryColor=String(judge.stance||'').toUpperCase()==='SELL'?'#ff9387':'#5ce7af';judge.entry_zone.forEach((price,index)=>{const line=state.chart.candleSeries.createPriceLine({price:Number(price),color:entryColor,lineWidth:2,axisLabelVisible:true,title:index===0?'Entry Lo':'Entry Hi'});state.chart.entryLines.push({series:state.chart.candleSeries,line});});}const positions=((state.live.positions&&state.live.positions.length)?state.live.positions:(currentPaper().open_positions||[])).filter((item)=>String(item.symbol||'').toUpperCase()===state.symbol).slice(-6);positions.forEach((position)=>{const entry=state.chart.candleSeries.createPriceLine({price:Number(position.entry_price),color:'#ffd48a',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed,title:'Entry'});state.chart.positionLines.push({series:state.chart.candleSeries,line:entry});if(position.stop_loss!=null){const stop=state.chart.candleSeries.createPriceLine({price:Number(position.stop_loss),color:'#ff9387',lineWidth:1,title:'SL'});state.chart.positionLines.push({series:state.chart.candleSeries,line:stop});}if(position.take_profit!=null){const target=state.chart.candleSeries.createPriceLine({price:Number(position.take_profit),color:'#5ce7af',lineWidth:1,title:'TP'});state.chart.positionLines.push({series:state.chart.candleSeries,line:target});}});state.chart.instance.timeScale().fitContent();$('chart-tier').textContent=String(sim.confidence_tier||'-').replaceAll('_',' ').toUpperCase();$('chart-cone').textContent=sim.cone_width_pips==null?'-':`${fmtNum(sim.cone_width_pips,1)} pips`;$('chart-mfg').textContent=fmtNum((((state.sim||{}).mfg||{}).disagreement),6);}
function applyLiveTickToChart(){if(!state.chart.instance||!state.chart.baseCandles.length)return;const livePrice=Number(state.live.price||0);const tickTime=toUnix(state.live.timestamp)||Math.floor(Date.now()/1000);if(!Number.isFinite(livePrice)||livePrice<=0||!tickTime)return;const bucket=tickTime-(tickTime%60);const candles=state.chart.baseCandles;const last=candles[candles.length-1];if(bucket>last.time){const next={time:bucket,open:last.close,high:Math.max(last.close,livePrice),low:Math.min(last.close,livePrice),close:livePrice};candles.push(next);while(candles.length>240)candles.shift();state.chart.candleSeries.setData(candles);}else{last.high=Math.max(last.high,livePrice);last.low=Math.min(last.low,livePrice);last.close=livePrice;state.chart.candleSeries.update({...last});}}
function renderFeeds(){const feeds=((state.sim||{}).feeds||{});const items=((feeds.news)||[]).slice(0,8);const tech=currentTech();const sim=currentSimulation();$('feed-structure').textContent=`${String(tech.structure||'-').toUpperCase()} / ${String(tech.location||'-').toUpperCase()}`;$('feed-structure-sub').textContent=`Support ${fmtPrice(((tech.nearest_support||{}).price))} | Resistance ${fmtPrice(((tech.nearest_resistance||{}).price))} | RSI ${fmtNum(tech.rsi_14,1)} | ATR ${fmtNum(tech.atr_14,2)}`;$('feed-crowd').textContent=`Retail T ${fmtNum(((sim.testosterone_index||{}).retail),3)} | Inst T ${fmtNum(((sim.testosterone_index||{}).institutional),3)}`;$('feed-crowd-sub').textContent=`MFG ${fmtNum((((state.sim||{}).mfg||{}).disagreement),6)} | SQT ${fmtPct(((state.live.sqt||{}).rolling_accuracy))}`;const host=$('feeds');host.innerHTML='';if(!items.length){host.innerHTML='<div class=\"news-item\"><div class=\"news-title\">No current feed items.</div><div class=\"news-meta\">The simulator is still available when the news stream is thin.</div></div>';return;}items.forEach((item)=>{const sentiment=Number(item.sentiment||0);const color=sentiment>0.1?'var(--bull)':sentiment<-0.1?'var(--bear)':'var(--muted)';const node=document.createElement('div');node.className='news-item';node.innerHTML=`<div class='news-title'>${cleanHeadline(item.title)||'Untitled headline'}</div><div class='news-meta' style='color:${color}'>${item.source||'unknown'} | ${sentiment>0?'+':''}${(sentiment*100).toFixed(0)}% sentiment</div>`;host.appendChild(node);});}
function renderGuide(){const host=$('guide');host.innerHTML='';buildGuideItems().forEach((item)=>{const node=document.createElement('div');node.className='item';node.innerHTML=`<div class='label'>${item.label}</div><div class='value' style='font-size:20px'>${item.value}</div><div class='tiny'>${item.meaning}</div>`;host.appendChild(node);});}
function renderPacket(){const entries=((state.kimiLog||{}).entries)||[];const latest=entries.length?entries[entries.length-1]:null;if(!latest){$('kimi-meta').textContent='No Kimi packet logged yet.';$('kimi-meta').className='status';$('kimi-context').textContent='{}';$('kimi-glossary').innerHTML='<div class=\"tiny\">No numeric glossary yet.</div>';renderGuide();return;}$('kimi-meta').textContent=`${latest.request_kind||'packet'} | ${latest.model||'-'} | ${latest.packet_bucket_15m_utc||'-'} | ${latest.status||'-'}`;$('kimi-meta').className=`status ${latest.status==='ok'?'ok':latest.status==='error'?'warn':''}`.trim();$('kimi-context').textContent=JSON.stringify(latest.context||{},null,2);const host=$('kimi-glossary');host.innerHTML='';const glossary=latest.numeric_glossary||{};const keys=Object.keys(glossary).slice(0,60);if(!keys.length){host.innerHTML='<div class=\"tiny\">No numeric glossary available.</div>';}else{keys.forEach((key)=>{const row=glossary[key]||{};const node=document.createElement('div');node.className='item';node.innerHTML=`<div class='label mono'>${key}</div><div class='value' style='font-size:18px'>${row.value??'-'}</div><div class='tiny'>${row.meaning||''}</div>`;host.appendChild(node);});}renderGuide();}
function paperSummary(){const liveSummary=state.live.paper_summary||{};const fallback=((currentPaper()||{}).summary)||{};return Object.keys(liveSummary).length?liveSummary:fallback;}
function renderPaper(){const summary=paperSummary();const sim=currentSimulation();$('pb').textContent=summary.balance==null?'-':`$${fmtNum(summary.balance,2)}`;$('pe').textContent=summary.equity==null?'-':`$${fmtNum(summary.equity,2)}`;$('pr').textContent=summary.realized_pnl==null?'-':fmtMoney(summary.realized_pnl);$('pr').style.color=Number(summary.realized_pnl||0)>=0?'var(--bull)':'var(--bear)';$('pu').textContent=summary.unrealized_pnl==null?'-':fmtMoney(summary.unrealized_pnl);$('pu').style.color=Number(summary.unrealized_pnl||0)>=0?'var(--bull)':'var(--bear)';$('trade-lot').textContent=sim.suggested_lot==null?'-':`${fmtNum(sim.suggested_lot,2)} lot`;const positions=((state.live.positions&&state.live.positions.length)?state.live.positions:(currentPaper().open_positions||[])).filter((item)=>String(item.symbol||'').toUpperCase()===state.symbol);const openBody=$('open-body');openBody.innerHTML='';if(!positions.length){openBody.innerHTML='<tr><td colspan=\"9\" class=\"tiny\">No open paper positions.</td></tr>';}else{positions.forEach((pos)=>{const statusLabel=pos.tp_hit?'TP hit':pos.sl_hit?'SL hit':'live';const tr=document.createElement('tr');tr.innerHTML=`<td class='mono'>${String(pos.trade_id||'').slice(0,8)}</td><td style='color:${tone(pos.direction)}'>${pos.direction||'-'}</td><td>${fmtNum(pos.lot,2)}</td><td>${fmtPrice(pos.entry_price)}</td><td style='color:${Number(pos.unrealized_pnl_usd||0)>=0?'var(--bull)':'var(--bear)'}'>${fmtMoney(pos.unrealized_pnl_usd)}</td><td><input data-sl='${pos.trade_id}' value='${pos.stop_loss==null?'':fmtNum(pos.stop_loss,2)}'></td><td><input data-tp='${pos.trade_id}' value='${pos.take_profit==null?'':fmtNum(pos.take_profit,2)}'></td><td class='tiny'>${statusLabel}</td><td><button data-mod='${pos.trade_id}' class='btn-secondary'>Save</button><button data-close='${pos.trade_id}' class='btn-secondary'>Close</button></td>`;openBody.appendChild(tr);});}const closedBody=$('closed-body');const closed=(currentPaper().closed_trades||[]).slice().reverse();closedBody.innerHTML='';if(!closed.length){closedBody.innerHTML='<tr><td colspan=\"6\" class=\"tiny\">No closed trades yet.</td></tr>';}else{closed.slice(0,120).forEach((item)=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${fmtTime(item.exit_time||item.entry_time)}</td><td style='color:${tone(item.direction)}'>${item.direction||'-'}</td><td>${fmtNum(item.lot,2)}</td><td>${fmtPrice(item.entry_price)}</td><td>${fmtPrice(item.exit_price)}</td><td style='color:${Number(item.pnl_usd||0)>=0?'var(--bull)':'var(--bear)'}'>${fmtMoney(item.pnl_usd)}</td>`;closedBody.appendChild(tr);});}openBody.querySelectorAll('button[data-close]').forEach((button)=>{button.onclick=async()=>{try{await jsonFetch('/api/paper/close',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trade_id:button.getAttribute('data-close'),exit_price:Number(state.live.price??currentMarket().current_price??0)})});await loadPaper();syncChart();status('Paper trade closed.','ok');}catch(error){status(`Close failed: ${error.message}`,'bad');}};});openBody.querySelectorAll('button[data-mod]').forEach((button)=>{button.onclick=async()=>{const id=button.getAttribute('data-mod');const sl=openBody.querySelector(`input[data-sl="${id}"]`);const tp=openBody.querySelector(`input[data-tp="${id}"]`);try{await jsonFetch('/api/paper/modify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trade_id:id,stop_loss:sl&&sl.value!==''?Number(sl.value):null,take_profit:tp&&tp.value!==''?Number(tp.value):null})});await loadPaper();syncChart();status(`Updated SL/TP for ${id}.`,'ok');}catch(error){status(`Modify failed: ${error.message}`,'bad');}};});}
async function loadPaper(){state.paper=await jsonFetch(`/api/paper/state?symbol=${encodeURIComponent(state.symbol)}`);renderPaper();}
async function loadKimiLog(){state.kimiLog=await jsonFetch('/api/llm/kimi-log?limit=10');renderPacket();}
async function loadKimiDecision(force=false){const query=new URLSearchParams({symbol:state.symbol,mode:state.mode,llm_provider:state.provider});if(state.model)query.set('llm_model',state.model);if(force)query.set('force','1');$('judge-status').textContent=`Refreshing Kimi on the 15-minute cache for ${state.symbol}...`;$('judge-status').className='status';try{const payload=await jsonFetch(`/api/llm/kimi-live?${query.toString()}`);state.sim=state.sim?({...state.sim,kimi_judge:payload.kimi_judge}):{kimi_judge:payload.kimi_judge};renderJudge();await loadKimiLog();const envelope=currentJudgeEnvelope();if(envelope.available){status(`Kimi refreshed on ${envelope.model||state.model||'default'}.`,'ok');}else{status(`Kimi fallback active: ${envelope.error||envelope.reason||'remote unavailable'}`,'warn');}}catch(error){renderJudge();status(`Kimi refresh failed: ${error.message}`,'warn');}}
function applyKimiSuggestion(){const judge=currentJudge();const stance=String(judge.stance||'HOLD').toUpperCase();if(!['BUY','SELL'].includes(stance)){status(`Kimi is holding this bar: ${judge.reasoning||judge.key_risk||'No actionable Kimi stance for this bar.'}`,'warn');return;}$('tdir').value=stance;const zone=Array.isArray(judge.entry_zone)&&judge.entry_zone.length===2?judge.entry_zone:[currentMarket().current_price,currentMarket().current_price];const mid=(Number(zone[0])+Number(zone[1]))/2;const pip=PIP_SIZES[state.symbol]||0.1;if(judge.stop_loss!=null)$('tstop').value=Math.max(1,Math.abs(mid-Number(judge.stop_loss))/pip).toFixed(1);if(judge.take_profit!=null)$('ttp').value=Math.max(1,Math.abs(Number(judge.take_profit)-mid)/pip).toFixed(1);$('trade-kimi').textContent=`Applied ${stance} | zone ${zone.map(fmtPrice).join(' - ')} | TP ${fmtPrice(judge.take_profit)} | SL ${fmtPrice(judge.stop_loss)}`;status(`Applied ${stance} Kimi setup to the paper form.`,'ok');}
async function loadChartSeed(){try{const payload=await jsonFetch(`/api/chart/realtime?symbol=${encodeURIComponent(state.symbol)}&bars=240`);state.sim=state.sim?({...state.sim,realtime_chart:payload}):{realtime_chart:payload};syncChart();}catch(error){status(`Chart seed failed: ${error.message}`,'warn');}}
async function runSimulation(announce=true){if(state.simLoading)return;state.simLoading=true;$('run').disabled=true;try{state.symbol=$('symbol').value;state.mode=$('mode').value;state.provider=$('provider').value;state.model=$('model').value.trim();if(announce)status(`Running V18 for ${state.symbol}...`);const query=new URLSearchParams({symbol:state.symbol,mode:state.mode,llm_provider:state.provider});if(state.model)query.set('llm_model',state.model);const payload=await jsonFetch(`/api/dashboard/live?${query.toString()}`);state.sim=payload;if(payload.paper_trading)state.paper=payload.paper_trading;renderDecisionBar();renderJudge();renderFeeds();renderPaper();syncChart();const judge=currentJudgeEnvelope();if(judge.available){status(`V18 refreshed for ${state.symbol}. Kimi cache is already live.`,'ok');}else{status(`V18 refreshed for ${state.symbol}. Loading the 15-minute Kimi packet in parallel...`,'warn');}loadKimiDecision(false).catch((error)=>status(`Kimi refresh failed: ${error.message}`,'warn'));}finally{state.simLoading=false;$('run').disabled=false;}}
async function openTrade(){const sim=currentSimulation();const market=currentMarket();const judge=currentJudge();const payload={symbol:state.symbol,direction:$('tdir').value,entry_price:Number(state.live.price??market.current_price??0),confidence_tier:String(sim.confidence_tier||'low'),sqt_label:String(((state.live.sqt||{}).label)||((state.sim||{}).sqt||{}).label||sim.sqt_label||'NEUTRAL'),mode:state.mode,leverage:Number($('tlev').value||200),stop_pips:Number($('tstop').value||20),take_profit_pips:Number($('ttp').value||30),stop_loss:judge.stop_loss!=null?Number(judge.stop_loss):null,take_profit:judge.take_profit!=null?Number(judge.take_profit):null,note:$('tnote').value};if(!payload.entry_price){status('No live price is available yet.','warn');return;}await jsonFetch('/api/paper/open',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});await loadPaper();syncChart();status(`Opened ${payload.direction} paper trade on ${payload.symbol}.`,'ok');}
async function resetPaper(){await jsonFetch('/api/paper/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({starting_balance:1000})});await loadPaper();syncChart();status('Paper account reset to $1000.','ok');}
function connectWs(){if(state.ws){try{state.ws.close();}catch(_err){}}state.live.connected=false;renderHealth();const socket=new WebSocket(`${wsOrigin()}/ws/live?symbol=${encodeURIComponent(state.symbol)}`);state.ws=socket;socket.onopen=()=>{state.live.connected=true;state.live.lastMessageAt=Date.now();try{socket.send(state.symbol);}catch(_err){}renderDecisionBar();};socket.onmessage=(event)=>{try{const payload=JSON.parse(event.data);state.live={...state.live,...payload,connected:true,lastMessageAt:Date.now()};renderDecisionBar();renderPaper();applyLiveTickToChart();}catch(error){status(`WebSocket payload error: ${error.message}`,'warn');}};socket.onerror=()=>{state.live.connected=false;renderDecisionBar();};socket.onclose=()=>{state.live.connected=false;renderDecisionBar();setTimeout(()=>connectWs(),1500);};}
function simulationLoop(){if(state.timer)clearInterval(state.timer);if($('auto').value!=='on')return;const seconds=Math.max(10,Number($('refresh').value||15));state.timer=setInterval(()=>runSimulation(false).catch((error)=>status(`Refresh failed: ${error.message}`,'bad')),seconds*1000);}
$('run').addEventListener('click',()=>runSimulation(true).catch((error)=>status(`Simulation failed: ${error.message}`,'bad')));
$('apply-kimi').addEventListener('click',applyKimiSuggestion);
$('open').addEventListener('click',()=>openTrade().catch((error)=>status(`Open failed: ${error.message}`,'bad')));
$('reset').addEventListener('click',()=>resetPaper().catch((error)=>status(`Reset failed: ${error.message}`,'bad')));
['symbol','mode','provider'].forEach((id)=>$(id).addEventListener('change',()=>{state.symbol=$('symbol').value;loadChartSeed();connectWs();runSimulation(true).catch((error)=>status(`Simulation failed: ${error.message}`,'bad'));}));
['refresh','auto'].forEach((id)=>$(id).addEventListener('change',simulationLoop));
window.addEventListener('load',async()=>{updateClock();setInterval(updateClock,1000);ensureChart();await loadChartSeed();connectWs();try{await loadPaper();await loadKimiLog();await runSimulation(false);}catch(error){status(`Initial load failed: ${error.message}`,'bad');}simulationLoop();});
</script>
</body>
</html>
"""
