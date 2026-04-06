from __future__ import annotations


def render_web_app_html() -> str:
    parts = [
        """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Nexus Trader V17</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;500;600;700;800&display=swap');
:root{
  --bg-0:#07111b;
  --bg-1:#0e1f2d;
  --glass:rgba(255,255,255,.11);
  --glass-2:rgba(255,255,255,.08);
  --line:rgba(255,255,255,.16);
  --text:#edf6ff;
  --muted:#a9bfd1;
  --gold:#ffd089;
  --bull:#49df9f;
  --bear:#ff8375;
  --cyan:#7fdcff;
  --shadow:0 24px 60px rgba(0,0,0,.34), inset 1px 1px 0 rgba(255,255,255,.10), inset -10px -10px 20px rgba(6,14,23,.25);
  --shadow-soft:inset 10px 10px 24px rgba(2,8,15,.22), inset -8px -8px 18px rgba(255,255,255,.04);
}
*{box-sizing:border-box}
body{
  margin:0;
  color:var(--text);
  font-family:'Outfit',sans-serif;
  background:
    radial-gradient(circle at 12% 18%,rgba(255,208,137,.18),transparent 24%),
    radial-gradient(circle at 88% 12%,rgba(127,220,255,.14),transparent 23%),
    radial-gradient(circle at 76% 72%,rgba(73,223,159,.10),transparent 26%),
    linear-gradient(155deg,var(--bg-0),#0a1724 45%,var(--bg-1));
  min-height:100vh;
}
body::before{
  content:'';
  position:fixed;
  inset:0;
  pointer-events:none;
  background:
    linear-gradient(transparent 49%,rgba(255,255,255,.02) 50%,transparent 51%),
    linear-gradient(90deg,transparent 49%,rgba(255,255,255,.015) 50%,transparent 51%);
  background-size:100% 24px,24px 100%;
  opacity:.25;
}
#app{max-width:1540px;margin:0 auto;padding:20px;display:grid;gap:16px}
.glass{
  background:linear-gradient(180deg,rgba(255,255,255,.13),rgba(255,255,255,.06));
  border:1px solid var(--line);
  border-radius:28px;
  backdrop-filter:blur(22px) saturate(130%);
  -webkit-backdrop-filter:blur(22px) saturate(130%);
  box-shadow:var(--shadow);
}
.neo{box-shadow:var(--shadow),var(--shadow-soft)}
.pad{padding:18px}
.grid{display:grid;gap:14px}
.hero{grid-template-columns:1.35fr .95fr}
.controls{grid-template-columns:repeat(7,minmax(0,1fr))}
.metrics{grid-template-columns:repeat(6,minmax(0,1fr))}
.main{grid-template-columns:1.45fr .95fr}
.split{grid-template-columns:1fr 1fr}
.triple{grid-template-columns:repeat(3,minmax(0,1fr))}
.quad{grid-template-columns:repeat(4,minmax(0,1fr))}
.row{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap}
.eyebrow,.lbl{font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--gold)}
.title{margin:0;font-size:clamp(34px,5vw,64px);line-height:.94;letter-spacing:-.06em}
.muted,.tiny{color:var(--muted)}
.tiny{font-size:12px;line-height:1.45}
.mono{font-family:'JetBrains Mono',monospace}
.val{font-size:24px;font-weight:800;line-height:1}
.huge{font-size:42px}
.pill,.item,.metric,.guide{background:var(--glass-2);border:1px solid rgba(255,255,255,.12);border-radius:20px}
.pill{padding:10px 14px;border-radius:999px}
.item,.metric,.guide{padding:14px}
.metric{min-height:124px}
.guide{min-height:152px}
.warn{padding:14px 16px;border-radius:20px;border:1px solid rgba(255,208,137,.28);background:rgba(255,208,137,.11);box-shadow:var(--shadow-soft)}
.status{padding:12px 14px;border-radius:18px;border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.07);font-size:13px;color:var(--muted)}
.list{display:grid;gap:10px;max-height:360px;overflow:auto;padding-right:2px}
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl td,.tbl th{padding:9px 7px;text-align:left;border-bottom:1px solid rgba(255,255,255,.08)}
.clock{text-align:right}
.chipbar{display:flex;gap:10px;flex-wrap:wrap}
select,input,textarea,button{
  width:100%;
  min-height:44px;
  border-radius:16px;
  border:1px solid rgba(255,255,255,.12);
  background:linear-gradient(180deg,rgba(255,255,255,.09),rgba(255,255,255,.05));
  color:var(--text);
  padding:10px 12px;
  font:inherit;
  box-shadow:var(--shadow-soft);
}
textarea{min-height:96px;resize:vertical}
button{cursor:pointer;font-weight:700;background:linear-gradient(135deg,rgba(255,208,137,.24),rgba(127,220,255,.12)),linear-gradient(180deg,rgba(255,255,255,.10),rgba(255,255,255,.04))}
button:hover{transform:translateY(-1px)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.5 'JetBrains Mono',monospace;color:#d7ecfb}
#chart{min-height:590px}
.guide .val{font-size:20px}
.good{color:var(--bull)}
.bad{color:var(--bear)}
.mid{color:var(--gold)}
@media(max-width:1220px){.hero,.main,.split,.triple{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(3,minmax(0,1fr))}.controls{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:760px){#app{padding:12px}.metrics,.controls,.quad{grid-template-columns:1fr}#chart{min-height:420px}}
</style>
</head>
<body>
<div id='app'>
  <section class='glass neo pad grid hero'>
    <div class='grid'>
      <div class='eyebrow'>Nexus Trader V17</div>
      <div class='row'>
        <div>
          <h1 class='title'>Glassmorphism 15-Minute Crowd Simulator</h1>
          <div class='muted'>V17 keeps the 15-minute trade horizon, adds WLTC, MMM, and the relativistic cone, and preserves paper trading instead of replacing it.</div>
        </div>
        <div class='clock'>
          <div id='time' class='val huge mono'>--:--:--</div>
          <div id='date' class='tiny'>Waiting for local time</div>
          <div id='next' class='tiny'>Next 5m bar in --s</div>
        </div>
      </div>
      <div class='grid quad'>
        <div class='metric neo'><div class='lbl'>Current Price</div><div id='m-price' class='val'>-</div><div class='tiny'>Latest market anchor used by the simulator.</div></div>
        <div class='metric neo'><div class='lbl'>15m Direction</div><div id='m-dir' class='val'>-</div><div id='m-dir-sub' class='tiny'>Consensus path over the next 15 minutes.</div></div>
        <div class='metric neo'><div class='lbl'>Confidence Tier</div><div id='m-tier' class='val'>-</div><div id='m-tier-sub' class='tiny'>Execution-confidence bucket.</div></div>
        <div class='metric neo'><div class='lbl'>SQT Status</div><div id='m-sqt' class='val'>-</div><div id='m-sqt-sub' class='tiny'>Rolling simulator hit-rate state.</div></div>
      </div>
    </div>
    <div class='grid'>
      <div class='chipbar'>
        <div class='pill'><strong id='b-sym'>XAUUSD</strong> symbol</div>
        <div class='pill'><strong id='b-prov'>LM Studio</strong> provider</div>
        <div class='pill'><strong id='b-mode'>Frequency</strong> mode</div>
      </div>
      <div id='eci' class='warn'>Economic-calendar context waiting for the first simulation payload.</div>
      <div id='status' class='status'>Loading the V17 simulator...</div>
      <div class='grid triple'>
        <div class='guide neo'><div class='lbl'>Retail Testosterone</div><div id='w-retail' class='val'>-</div><div class='tiny'>WLTC winner-cycle intensity for the retail crowd.</div></div>
        <div class='guide neo'><div class='lbl'>Hurst Asymmetry</div><div id='h-asym' class='val'>-</div><div class='tiny'>Positive means upside is more persistent; negative means downside is stickier.</div></div>
        <div class='guide neo'><div class='lbl'>Relativistic c_m</div><div id='cone-cm' class='val'>-</div><div class='tiny'>The empirical speed limit for the hard cone boundary.</div></div>
      </div>
    </div>
  </section>

  <section class='glass neo pad grid controls'>
    <div><div class='lbl'>Instrument</div><select id='symbol'><option value='XAUUSD' selected>XAUUSD</option><option value='EURUSD'>EURUSD</option><option value='BTCUSD'>BTCUSD</option></select></div>
    <div><div class='lbl'>Mode</div><select id='mode'><option value='frequency' selected>Frequency</option><option value='precision'>Precision</option></select></div>
    <div><div class='lbl'>LLM Provider</div><select id='provider'><option value='lm_studio' selected>LM Studio</option><option value='ollama'>Ollama</option><option value='nvidia_nim'>NVIDIA NIM / Kimi</option></select></div>
    <div><div class='lbl'>Model Override</div><input id='model' placeholder='moonshotai/kimi-k2-instruct'></div>
    <div><div class='lbl'>Refresh Seconds</div><select id='refresh'><option value='15'>15</option><option value='30' selected>30</option><option value='60'>60</option></select></div>
    <div><div class='lbl'>Auto Refresh</div><select id='auto'><option value='on' selected>On</option><option value='off'>Off</option></select></div>
    <div><div class='lbl'>Run</div><button id='run'>Refresh V17</button></div>
  </section>
""",
        """
  <section class='grid metrics'>
    <div class='glass neo pad metric'><div class='lbl'>CABR Proxy</div><div id='cabr' class='val'>-</div><div class='tiny'>Branch-ranking strength after the cross-attention ranker.</div></div>
    <div class='glass neo pad metric'><div class='lbl'>BST / Plausibility</div><div id='bst' class='val'>-</div><div class='tiny'>Legacy stability blended with relativistic branch plausibility.</div></div>
    <div class='glass neo pad metric'><div class='lbl'>CPM Score</div><div id='cpm' class='val'>-</div><div class='tiny'>Conditional predictability estimate carried forward from V15.</div></div>
    <div class='glass neo pad metric'><div class='lbl'>Inner Cone Width</div><div id='cone' class='val'>-</div><div class='tiny'>Average width of the V17 inner execution band in pips.</div></div>
    <div class='glass neo pad metric'><div class='lbl'>Suggested Lot</div><div id='lot' class='val'>-</div><div class='tiny'>Paper lot scaled to equity, tier, and leverage cap.</div></div>
    <div class='glass neo pad metric'><div class='lbl'>Execution Rule</div><div id='exec' class='val'>-</div><div id='exec-sub' class='tiny'>Decision status from mode plus SQT context.</div></div>
  </section>

  <section class='grid main'>
    <div class='glass neo pad grid'>
      <div class='row'>
        <div>
          <h2>V17 Cone Chart</h2>
          <div class='muted'>The shaded band is the inner relativistic cone. The red dashed boundary is the hard physical envelope. The dotted cyan line is the minority path.</div>
        </div>
        <div class='chipbar'>
          <div class='pill'><strong id='chart-tier'>-</strong> tier</div>
          <div class='pill'><strong id='chart-hit'>-</strong> cone hits</div>
          <div class='pill'><strong id='chart-asym'>-</strong> asymmetry</div>
        </div>
      </div>
      <div id='chart'></div>
    </div>

    <div class='glass neo pad grid'>
      <div>
        <h2>Paper Trading</h2>
        <div class='muted'>Manual paper trades sit on top of the live simulator. Leverage is for paper testing only; the research track remains unlevered.</div>
      </div>
      <div class='grid quad'>
        <div class='guide neo'><div class='lbl'>Balance</div><div id='pb' class='val'>-</div></div>
        <div class='guide neo'><div class='lbl'>Equity</div><div id='pe' class='val'>-</div></div>
        <div class='guide neo'><div class='lbl'>Realized PnL</div><div id='pr' class='val'>-</div></div>
        <div class='guide neo'><div class='lbl'>Unrealized PnL</div><div id='pu' class='val'>-</div></div>
      </div>
      <div class='grid split'>
        <div><div class='lbl'>Direction</div><select id='tdir'><option value='BUY'>BUY</option><option value='SELL'>SELL</option></select></div>
        <div><div class='lbl'>Leverage</div><select id='tlev'><option value='50'>1:50</option><option value='100'>1:100</option><option value='200' selected>1:200</option></select></div>
      </div>
      <div class='grid split'>
        <div><div class='lbl'>Stop Pips</div><input id='tstop' type='number' min='1' step='1' value='20'></div>
        <div><div class='lbl'>Take Profit Pips</div><input id='ttp' type='number' min='1' step='1' value='30'></div>
      </div>
      <div><div class='lbl'>Note</div><textarea id='tnote' placeholder='Why this paper trade exists'></textarea></div>
      <div class='grid split'>
        <button id='open'>Open Paper Trade</button>
        <button id='reset'>Reset Paper Account</button>
      </div>
      <div style='overflow:auto'><table class='tbl'><thead><tr><th>ID</th><th>Side</th><th>Lot</th><th>Entry</th><th>PnL</th><th></th></tr></thead><tbody id='open-body'></tbody></table></div>
    </div>
  </section>

  <section class='grid split'>
    <div class='glass neo pad grid'><h2>Recent Simulations</h2><div class='muted'>Direction hit, cone hit rate, confidence tier, and minority rescue context.</div><div id='sims' class='list'></div></div>
    <div class='glass neo pad grid'><h2>Closed Trade History</h2><div class='muted'>Basic paper-trade history for live testing.</div><div style='overflow:auto'><table class='tbl'><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Lot</th><th>PnL</th></tr></thead><tbody id='closed-body'></tbody></table></div></div>
  </section>

  <section class='grid split'>
    <div class='glass neo pad grid'><h2>Judge + Context</h2><div id='judge' class='list'></div></div>
    <div class='glass neo pad grid'><h2>News + Crowd</h2><div id='feeds' class='list'></div></div>
  </section>

  <section class='grid split'>
    <div class='glass neo pad grid'>
      <h2>What The Numbers Mean</h2>
      <div class='muted'>These are the live V17 values on the page, with the meaning written beside them so the dashboard is self-explaining.</div>
      <div id='guide' class='grid quad'></div>
    </div>
    <div class='glass neo pad grid'>
      <div>
        <h2>Kimi 15m Packet</h2>
        <div class='muted'>Every NVIDIA NIM / Kimi request is logged with the exact 15-minute V17 context plus a numeric glossary.</div>
      </div>
      <div id='kimi-meta' class='status'>No Kimi packet logged yet. Switch the provider to NVIDIA NIM and refresh once the API key is present.</div>
      <div class='grid'>
        <div class='guide neo'><div class='lbl'>Context Sent To Kimi</div><pre id='kimi-context'>{}</pre></div>
        <div class='guide neo'><div class='lbl'>Numeric Glossary</div><div id='kimi-glossary' class='list'></div></div>
      </div>
    </div>
  </section>
</div>

<script>
const st={symbol:'XAUUSD',mode:'frequency',provider:'lm_studio',model:'',sim:null,mon:null,paper:null,kimi:null,timer:null};
const $=(id)=>document.getElementById(id);
const pct=(v)=>v==null||Number.isNaN(Number(v))?'-':`${(Number(v)*100).toFixed(1)}%`;
const price=(v)=>v==null||Number.isNaN(Number(v))?'-':Number(v).toFixed(2);
const qty=(v,d=2)=>v==null||Number.isNaN(Number(v))?'-':Number(v).toFixed(d);
const sgn=(v)=>v==null||Number.isNaN(Number(v))?'-':`${Number(v)>=0?'+':''}${Number(v).toFixed(2)}`;
const tm=(v)=>{if(!v)return'-';const d=new Date(v);return Number.isNaN(d.getTime())?String(v):d.toLocaleString();};
function tone(direction){const d=String(direction||'').toUpperCase();if(d==='BUY'||d==='BULLISH')return'var(--bull)';if(d==='SELL'||d==='BEARISH')return'var(--bear)';return'var(--gold)';}
function stat(message){$('status').textContent=message;}
async function j(url,opt){const r=await fetch(url,opt);if(!r.ok){const t=await r.text();throw new Error(t||`Request failed with ${r.status}`);}return r.json();}
function tick(){const now=new Date(),next=new Date(now);next.setSeconds(0,0);const mod=now.getMinutes()%5;next.setMinutes(now.getMinutes()+(mod===0?5:5-mod));$('time').textContent=now.toLocaleTimeString();$('date').textContent=now.toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric',year:'numeric'});$('next').textContent=`Next 5m bar in ${Math.max(0,Math.floor((next-now)/1000))}s`;}
""",
        """
function head(){
  const sim=st.sim||{},market=sim.market||{},v=((sim.v16||{}).simulation||sim.simulation||{}),s=sim.sqt||{},mmm=sim.mmm||{},w=v.testosterone_index||sim.wltc||{};
  $('m-price').textContent=price(market.current_price);
  $('m-dir').textContent=String(v.direction||'-');
  $('m-dir').style.color=tone(v.direction);
  $('m-dir-sub').textContent=`${String(st.mode).toUpperCase()} mode on a 15m horizon`;
  $('m-tier').textContent=String(v.confidence_tier||'-').replaceAll('_',' ').toUpperCase();
  $('m-tier').style.color=v.tier_color||'var(--gold)';
  $('m-tier-sub').textContent=v.tier_label||'No tier label available';
  $('m-sqt').textContent=String(s.label||v.sqt_label||'-');
  $('m-sqt').style.color=tone(String(s.label||'').toUpperCase()==='COLD'?'SELL':'BUY');
  $('m-sqt-sub').textContent=`Rolling accuracy ${pct(s.rolling_accuracy||v.sqt_accuracy)}`;
  $('b-sym').textContent=st.symbol;
  $('b-prov').textContent=st.provider==='nvidia_nim'?'NVIDIA NIM':st.provider==='ollama'?'Ollama':'LM Studio';
  $('b-mode').textContent=String(st.mode).replace(/^./,(c)=>c.toUpperCase());
  $('eci').textContent=(sim.eci||{}).note||'No event-pressure note available.';
  $('w-retail').textContent=qty((w.retail&&w.retail.testosterone_index)!=null?w.retail.testosterone_index:v.testosterone_index&&v.testosterone_index.retail,3);
  $('h-asym').textContent=qty(mmm.hurst_asymmetry ?? v.hurst_asymmetry,3);
  $('cone-cm').textContent=qty(v.cone_c_m,6);
  $('chart-tier').textContent=String(v.confidence_tier||'-').replaceAll('_',' ').toUpperCase();
  $('chart-tier').style.color=v.tier_color||'var(--gold)';
  const rec=((st.mon||{}).recent_simulations||[]).filter((x)=>typeof x.hit_rate==='number');
  const avg=rec.length?rec.reduce((a,x)=>a+Number(x.hit_rate||0),0)/rec.length:null;
  $('chart-hit').textContent=avg==null?'-':pct(avg);
  $('chart-asym').textContent=(Number(v.cone_h_minus||0)-Number(v.cone_h_plus||0))>0.02?'Downside wider':'Near symmetric';
}

function metrics(){
  const v=(((st.sim||{}).v16||{}).simulation||((st.sim||{}).simulation)||{});
  $('cabr').textContent=pct(v.cabr_score);
  $('bst').textContent=pct(v.bst_score);
  $('cpm').textContent=pct(v.cpm_score);
  $('cone').textContent=v.cone_width_pips==null?'-':`${qty(v.cone_width_pips,1)} pips`;
  $('lot').textContent=v.suggested_lot==null?'-':`${qty(v.suggested_lot,2)} lot`;
  $('exec').textContent=v.should_execute?'EXECUTE':'WAIT';
  $('exec').className='val '+(v.should_execute?'good':'mid');
  $('exec-sub').textContent=String(v.execution_reason||'No execution reason available');
}

function chart(){
  const sim=st.sim||{},market=sim.market||{},cand=market.candles||[],v=((sim.v16||{}).simulation||sim.simulation||{}),forecast=(sim.final_forecast||{}).points||[];
  const ops=((st.paper||{}).open_positions||[]).filter((x)=>String(x.symbol||'').toUpperCase()===st.symbol);
  const traces=[{type:'candlestick',x:cand.map(x=>x.timestamp),open:cand.map(x=>x.open),high:cand.map(x=>x.high),low:cand.map(x=>x.low),close:cand.map(x=>x.close),increasing:{line:{color:'#49df9f'}},decreasing:{line:{color:'#ff8375'}},name:'Live market'}];
  const anchor=cand.length?cand[cand.length-1].timestamp:null;
  const fx=[anchor].concat(forecast.map(x=>x.timestamp||null));
  const consensus=v.consensus_path||[],innerUpper=v.cone_upper||[],innerLower=v.cone_lower||[],outerUpper=v.cone_outer_upper||[],outerLower=v.cone_outer_lower||[],minority=v.minority_path||[];
  if(forecast.length&&consensus.length===forecast.length+1){
    traces.push({type:'scatter',mode:'lines',x:fx,y:innerLower,line:{color:'rgba(0,0,0,0)'},hoverinfo:'skip',showlegend:false});
    traces.push({type:'scatter',mode:'lines',x:fx,y:innerUpper,line:{color:'rgba(0,0,0,0)'},fill:'tonexty',fillcolor:'rgba(127,220,255,.18)',name:'Inner cone',hoverinfo:'skip'});
    traces.push({type:'scatter',mode:'lines',x:fx,y:[consensus[0]].concat(outerUpper.slice(1)),line:{color:'#ff8375',width:2,dash:'dash'},name:'Outer upper',hovertemplate:'Physical price boundary (relativistic limit)<extra></extra>'});
    traces.push({type:'scatter',mode:'lines',x:fx,y:[consensus[0]].concat(outerLower.slice(1)),line:{color:'#ff8375',width:2,dash:'dash'},name:'Outer lower',hovertemplate:'Physical price boundary (relativistic limit)<extra></extra>'});
    traces.push({type:'scatter',mode:'lines+markers',x:fx,y:consensus,line:{color:v.tier_color||'#ffd089',width:4},marker:{size:7,color:v.tier_color||'#ffd089'},name:'Consensus path'});
    traces.push({type:'scatter',mode:'lines+markers',x:fx.slice(0,minority.length),y:minority,line:{color:'#7fdcff',width:2,dash:'dot'},marker:{size:5,color:'#7fdcff'},name:'Minority path'});
  }
  if(ops.length){
    traces.push({type:'scatter',mode:'markers+text',x:ops.map(x=>x.entry_time),y:ops.map(x=>x.entry_price),text:ops.map(x=>`${x.direction} ${x.lot} lot`),textposition:'top center',marker:{size:12,color:ops.map(x=>String(x.direction).toUpperCase()==='BUY'?'#49df9f':'#ff8375'),symbol:ops.map(x=>String(x.direction).toUpperCase()==='BUY'?'triangle-up':'triangle-down')},name:'Open paper trades'});
  }
  Plotly.react('chart',traces,{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(255,255,255,.03)',font:{color:'#edf6ff',family:'JetBrains Mono, monospace'},margin:{l:46,r:18,t:16,b:38},hovermode:'x unified',xaxis:{gridcolor:'rgba(255,255,255,.06)',rangeslider:{visible:false}},yaxis:{gridcolor:'rgba(255,255,255,.06)'},legend:{orientation:'h',x:0,y:1.08}},{responsive:true,displaylogo:false});
}

function sims(){
  const host=$('sims');host.innerHTML='';
  const arr=(st.mon||{}).recent_simulations||[];
  if(!arr.length){host.innerHTML='<div class="item"><strong>No completed simulations yet.</strong><div class="tiny">Completed 15-minute outcomes will appear here once a forecast matures.</div></div>';return;}
  for(const x of arr){
    const n=document.createElement('div');n.className='item neo';
    n.innerHTML=`<strong>${String(x.scenario_bias||'-').toUpperCase()} -> ${String(x.realized_direction||'pending').toUpperCase()}</strong><div class='tiny'>Tier ${String(x.confidence_tier||'uncertain').toUpperCase()} | cone hit ${x.hit_rate==null?'pending':pct(x.hit_rate)}</div><div class='tiny'>Direction ${x.direction_match==null?'pending':x.direction_match?'correct':'wrong'} | minority rescue ${x.minority_was_closer==null?'pending':x.minority_was_closer?'yes':'no'}</div><div class='tiny'>${tm(x.anchor_timestamp)}</div>`;
    host.appendChild(n);
  }
}

function ctx(){
  const sim=st.sim||{},judge=((sim.swarm_judge||{}).content||{}),jh=$('judge'),fh=$('feeds');
  jh.innerHTML='';fh.innerHTML='';
  for(const row of[{t:`Judge stance: ${String(judge.manual_stance||'hold').toUpperCase()}`,d:judge.manual_action_reason||'Manual stance unavailable.'},{t:`Crowd lean: ${judge.crowd_lean||'unknown'}`,d:judge.crowd_emotion||'No crowd-emotion summary.'},{t:'Minority case',d:judge.minority_case||'Minority case unavailable.'},{t:'Takeaway',d:judge.judge_summary||judge.discussion_takeaway||'Judge summary unavailable.'}]){
    const n=document.createElement('div');n.className='item neo';n.innerHTML=`<strong>${row.t}</strong><div class='tiny'>${row.d}</div>`;jh.appendChild(n);
  }
  const items=(((sim.feeds||{}).news)||[]).slice(0,4).concat((((sim.feeds||{}).public_discussions)||[]).slice(0,4));
  if(!items.length){fh.innerHTML='<div class="item"><strong>No live feed items available.</strong><div class="tiny">The simulator still runs even when the sidecar feed is thin.</div></div>';return;}
  for(const it of items){
    const n=document.createElement('div');n.className='item neo';n.innerHTML=`<strong>${it.title||'Untitled feed item'}</strong><div class='tiny'>${it.source||'unknown'} | ${tm(it.published_at)}</div>`;fh.appendChild(n);
  }
}

function paper(){
  const p=st.paper||{},s=p.summary||{};
  $('pb').textContent=s.balance==null?'-':`$${qty(s.balance,2)}`;
  $('pe').textContent=s.equity==null?'-':`$${qty(s.equity,2)}`;
  $('pr').textContent=s.realized_pnl==null?'-':`$${sgn(s.realized_pnl)}`;
  $('pr').className='val '+(Number(s.realized_pnl||0)>=0?'good':'bad');
  $('pu').textContent=s.unrealized_pnl==null?'-':`$${sgn(s.unrealized_pnl)}`;
  $('pu').className='val '+(Number(s.unrealized_pnl||0)>=0?'good':'bad');
  const ob=$('open-body');ob.innerHTML='';const ops=p.open_positions||[];
  if(!ops.length){ob.innerHTML='<tr><td colspan="6" class="tiny">No open paper positions.</td></tr>';}
  else for(const x of ops){const r=document.createElement('tr');r.innerHTML=`<td class='mono'>${String(x.trade_id||'').slice(0,8)}</td><td style='color:${tone(x.direction)};'>${x.direction}</td><td>${qty(x.lot,2)}</td><td>${price(x.entry_price)}</td><td style='color:${Number(x.unrealized_pnl_usd||0)>=0?'var(--bull)':'var(--bear)'};'>$${sgn(x.unrealized_pnl_usd)}</td><td><button data-close='${x.trade_id}'>Close</button></td>`;ob.appendChild(r);}
  const cb=$('closed-body');cb.innerHTML='';const closed=(p.closed_trades||[]).slice().reverse();
  if(!closed.length){cb.innerHTML='<tr><td colspan="5" class="tiny">No closed trades yet.</td></tr>';}
  else for(const x of closed){const r=document.createElement('tr');r.innerHTML=`<td>${tm(x.exit_time||x.entry_time)}</td><td>${x.symbol||'-'}</td><td style='color:${tone(x.direction)};'>${x.direction||'-'}</td><td>${qty(x.lot,2)}</td><td style='color:${Number(x.pnl_usd||0)>=0?'var(--bull)':'var(--bear)'};'>$${sgn(x.pnl_usd)}</td>`;cb.appendChild(r);}
  ob.querySelectorAll('button[data-close]').forEach((b)=>b.addEventListener('click',async()=>{try{const id=b.getAttribute('data-close'),px=Number((((st.sim||{}).market||{}).current_price)||0);if(!id||!px)return;await j('/api/paper/close',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trade_id:id,exit_price:px})});await loadPaper();chart();stat(`Closed paper trade ${id}.`);}catch(e){stat(`Paper close failed: ${e.message}`);}}));
}
""",
        """
function numericGuide(){
  const sim=st.sim||{},market=sim.market||{},v=((sim.v16||{}).simulation||sim.simulation||{}),mmm=sim.mmm||{},w=sim.wltc||{},guide=$('guide');
  const retailW=(w.retail||{}),noiseW=(w.noise||{});
  const items=[
    {label:'Current Price',value:price(market.current_price),meaning:'Latest live market price used as the anchor for the 15-minute simulation.'},
    {label:'CABR Proxy',value:pct(v.cabr_score),meaning:'Branch-ranking confidence. Higher means the selected branch is structurally stronger.'},
    {label:'BST / Plausibility',value:pct(v.bst_score),meaning:'Blend of legacy branch survival and the V17 relativistic envelope plausibility.'},
    {label:'CPM Score',value:pct(v.cpm_score),meaning:'Predictability estimate. Higher means the local feature state historically behaves more consistently.'},
    {label:'Inner Cone Width',value:v.cone_width_pips==null?'-':`${qty(v.cone_width_pips,1)} pips`,meaning:'Average width of the inner 15-minute execution band.'},
    {label:'Relativistic c_m',value:qty(v.cone_c_m,6),meaning:'Empirical market speed limit. It defines the hard outer cone boundary.'},
    {label:'H+',value:qty(v.cone_h_plus ?? mmm.hurst_positive,4),meaning:'Persistence of upward moves. Above 0.5 means upside continuation is more likely than random.'},
    {label:'H-',value:qty(v.cone_h_minus ?? mmm.hurst_negative,4),meaning:'Persistence of downward moves. Higher than H+ means downside is structurally stickier.'},
    {label:'H Asymmetry',value:qty(mmm.hurst_asymmetry ?? v.hurst_asymmetry,4),meaning:'H+ minus H-. Negative values mean downside persistence dominates.'},
    {label:'Retail Testosterone',value:qty(retailW.testosterone_index,4),meaning:'WLTC winner-cycle intensity for retail participants.'},
    {label:'Noise Testosterone',value:qty(noiseW.testosterone_index,4),meaning:'WLTC winner-cycle intensity for the noisiest impulsive crowd slice.'},
    {label:'Suggested Lot',value:v.suggested_lot==null?'-':`${qty(v.suggested_lot,2)} lot`,meaning:'Paper lot size scaled to equity, stop distance, confidence tier, and leverage cap.'}
  ];
  guide.innerHTML='';
  for(const item of items){const n=document.createElement('div');n.className='guide neo';n.innerHTML=`<div class='lbl'>${item.label}</div><div class='val'>${item.value}</div><div class='tiny'>${item.meaning}</div>`;guide.appendChild(n);}
}

function kimiPanel(){
  const entries=((st.kimi||{}).entries)||[],latest=entries.length?entries[entries.length-1]:null;
  if(!latest){$('kimi-meta').textContent='No Kimi packet logged yet. Switch to NVIDIA NIM and refresh after the API key is configured.';$('kimi-context').textContent='{}';$('kimi-glossary').innerHTML='<div class="item"><strong>No glossary yet.</strong><div class="tiny">A Kimi request needs to run first.</div></div>';return;}
  $('kimi-meta').textContent=`${latest.request_kind} | ${latest.model} | bucket ${latest.packet_bucket_15m_utc} | ${latest.status}`;
  $('kimi-context').textContent=JSON.stringify(latest.context||{},null,2);
  const host=$('kimi-glossary');host.innerHTML='';const glossary=latest.numeric_glossary||{},keys=Object.keys(glossary);
  if(!keys.length){host.innerHTML='<div class="item"><strong>No numeric glossary available.</strong></div>';return;}
  for(const key of keys.slice(0,40)){const item=glossary[key]||{};const n=document.createElement('div');n.className='item neo';n.innerHTML=`<strong class='mono'>${key}</strong><div class='tiny'>Value: ${item.value}</div><div class='tiny'>${item.meaning||''}</div>`;host.appendChild(n);}
}

async function loadPaper(){st.paper=await j(`/api/paper/state?symbol=${encodeURIComponent(st.symbol)}`);paper();}
async function refreshMon(){st.mon=await j(`/api/live-monitor?symbol=${encodeURIComponent(st.symbol)}`);sims();head();}
async function loadKimi(){st.kimi=await j('/api/llm/kimi-log?limit=8');kimiPanel();}

async function run(announce=true){
  st.symbol=$('symbol').value;st.mode=$('mode').value;st.provider=$('provider').value;st.model=$('model').value.trim();
  if(announce)stat(`Running ${st.symbol} through the V17 simulator...`);
  const q=new URLSearchParams({symbol:st.symbol,mode:st.mode,llm_provider:st.provider});if(st.model)q.set('llm_model',st.model);
  st.sim=await j(`/api/simulate-live?${q.toString()}`);
  head();metrics();ctx();numericGuide();await refreshMon();await loadPaper();await loadKimi();chart();
  stat(`V17 simulation refreshed for ${st.symbol}.`);
}

async function openTrade(){
  const sim=st.sim||{},market=sim.market||{},v=((sim.v16||{}).simulation||sim.simulation||{});
  const payload={symbol:st.symbol,direction:$('tdir').value,entry_price:Number(market.current_price||0),confidence_tier:String(v.confidence_tier||'uncertain'),sqt_label:String(((sim.sqt||{}).label)||v.sqt_label||'NEUTRAL'),mode:st.mode,leverage:Number($('tlev').value||200),stop_pips:Number($('tstop').value||20),take_profit_pips:Number($('ttp').value||30),note:$('tnote').value};
  if(!payload.entry_price){stat('No live price available for paper trading.');return;}
  await j('/api/paper/open',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  await loadPaper();chart();stat(`Opened ${payload.direction} paper trade on ${payload.symbol}.`);
}

async function resetPaper(){await j('/api/paper/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({starting_balance:1000})});await loadPaper();chart();stat('Paper account reset to $1000.');}
function loop(){if(st.timer)clearInterval(st.timer);if($('auto').value!=='on')return;const s=Number($('refresh').value||30);st.timer=setInterval(()=>run(false).catch((e)=>stat(`Refresh failed: ${e.message}`)),s*1000);}

$('run').addEventListener('click',()=>run(true).catch((e)=>stat(`Simulation failed: ${e.message}`)));
$('symbol').addEventListener('change',()=>run(true).catch((e)=>stat(`Simulation failed: ${e.message}`)));
$('mode').addEventListener('change',()=>run(true).catch((e)=>stat(`Simulation failed: ${e.message}`)));
$('provider').addEventListener('change',()=>run(true).catch((e)=>stat(`Simulation failed: ${e.message}`)));
$('refresh').addEventListener('change',loop);$('auto').addEventListener('change',loop);
$('open').addEventListener('click',()=>openTrade().catch((e)=>stat(`Paper open failed: ${e.message}`)));
$('reset').addEventListener('click',()=>resetPaper().catch((e)=>stat(`Paper reset failed: ${e.message}`)));
window.addEventListener('load',async()=>{tick();setInterval(tick,1000);try{await loadPaper();await loadKimi();await run(false);loop();}catch(e){stat(`Initial load failed: ${e.message}`);}});
</script>
</body>
</html>
""",
    ]
    return "".join(parts)
