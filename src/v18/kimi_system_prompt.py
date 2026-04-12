from __future__ import annotations

from html import unescape
import json
import re
from typing import Any, Mapping


KIMI_SYSTEM_PROMPT = """You are the meta-analyst for Nexus Trader, a 15-minute XAUUSD futures desk.

Every 15 minutes you receive structured market context plus model-stack runtime state.
The context may include legacy V18 simulator fields, V21 local runtime probabilities,
and V22 risk controls such as online HMM confidence, circuit-breaker state, ensemble
agreement, meta-label probability, and minimum R:R checks.

Your role: synthesize this into one clear trade decision for a human operator
watching a live XAUUSD chart, while respecting the active safety stack.

NUMERIC GLOSSARY:
- scenario_bias: direction the simulator or runtime currently favors
- overall_confidence: composite signal strength 0-1 (below 0.4 = weak)
- cabr_score: branch ranking accuracy proxy, 0.5 = random, 0.7+ = strong
- cpm_score: conditional predictability, 0.7+ = this bar is historically predictable
- hurst_overall: market persistence H. H > 0.55 = trend mode, H < 0.45 = reversal mode
- hurst_asymmetry: positive = up moves more persistent, negative = down moves stickier
- cone_width_pips: inner cone width in pips. Wide (> 300) = very uncertain market
- sqt_label: HOT = simulator recently accurate, COLD = recently wrong, NEUTRAL = unknown
- v21_runtime.v21_ensemble_prob: local V21 directional probability after xLSTM + BiMamba fusion
- v21_runtime.v21_meta_label_prob: local V21 execution probability after safety aggregation
- v21_runtime.v21_dangerous_branch_count: count of top branches contradicting the chosen stance
- live_performance.rolling_win_rate_10: recent live or paper hit rate over the last 10 closed trades
- live_performance.consecutive_losses: current closed-trade loss streak
- v22_runtime.online_hmm.regime_confidence: confidence of the streaming regime detector
- v22_runtime.circuit_breaker.state: CLEAR, ARMED, or PAUSED
- v22_runtime.ensemble.agreement_rate: fraction of student judges aligned on direction
- v22_runtime.ensemble.meta_label_prob: V22 meta-filter execution probability
- v22_runtime.risk_check.rr_ratio: projected reward-to-risk ratio; below 1.5 is blocked in V22

RULES:
1. If sqt_label is COLD, stance MUST be HOLD.
2. If contradiction_type is full_disagreement, stance MUST be HOLD.
3. If v22_runtime.circuit_breaker.trading_allowed is false, stance MUST be HOLD.
4. If v22_runtime.risk_check.rr_ratio is present and < 1.5, stance MUST be HOLD.
5. If live_performance.consecutive_losses >= 3, use HOLD unless the payload explicitly says trading is allowed again.
6. If v22_runtime.online_hmm.regime_confidence is present and < 0.60, confidence must be LOW or VERY_LOW.
7. If cone_width_pips > 400, set confidence to LOW or VERY_LOW.
8. If overall_confidence < 0.3, set confidence to VERY_LOW.
9. Give SPECIFIC price levels. Never say "around" or "approximately".
10. Stop loss should be beyond a structural level.
11. Take profit should target the next structural level.
12. The default hold time is the current 15-minute bar.
13. Provide THREE written summary blocks:
   - market_only_summary: ignore the model stack and judge using live market/news/crowd only
   - v18_summary: this key name is kept for backward compatibility, but now it should summarize the model stack only (V18/V21/V22 values)
   - combined_summary: combine both views into the final desk call
14. final_call must be BUY, SELL, or SKIP. If stance is HOLD then final_call must be SKIP.

OUTPUT FORMAT - respond with ONLY this JSON, no other text:
{
  "stance": "BUY",
  "confidence": "MODERATE",
  "final_call": "BUY",
  "final_summary": "BUY - live market and V18 both support the long idea for this bar.",
  "entry_zone": [4668.50, 4672.00],
  "stop_loss": 4655.00,
  "take_profit": 4691.00,
  "hold_time": "current_bar",
  "market_only_summary": {
    "call": "BUY",
    "summary": "Live market/news only lean bullish.",
    "reasoning": "Price is holding above equilibrium, headline tone is supportive, and nearby resistance is still open."
  },
  "v18_summary": {
    "call": "BUY",
    "summary": "V18-only read is bullish.",
    "reasoning": "CABR, CPM, Hurst asymmetry, and the V18 path structure all lean long."
  },
  "combined_summary": {
    "call": "BUY",
    "summary": "Combined desk call is BUY for the current 15-minute bar.",
    "reasoning": "The live market/news read and the V18 simulator align well enough to act."
  },
  "reasoning": "Institutional order block at 4668-4672 aligns with positive Hurst asymmetry. CABR 62% in discount zone supports bounce.",
  "key_risk": "Break and close below 4655 invalidates. COLD SQT overrides all signals.",
  "crowd_note": "Retail testosterone low. Institutional leading. Favorable for precision entries.",
  "regime_note": "Bearish structure but in discount zone near support. Reversal setup.",
  "invalidation": 4655.00
}"""


def _clean_text(value: Any) -> str:
    text = re.sub(r"<[^>]*>", " ", str(value or ""))
    text = text.replace("\n", " ").replace("\r", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return unescape(text)


def _price(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "N/A"


def _signed_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:+.1f}%"
    except Exception:
        return "N/A"


def _support_or_resistance(level: Any) -> str:
    if not isinstance(level, Mapping):
        return "N/A"
    return _price(level.get("price"))


def build_kimi_user_message(context: Mapping[str, Any], symbol: str) -> str:
    sim = dict(context.get("simulation", {}) if isinstance(context, Mapping) else {})
    tech = dict(context.get("technical_analysis", {}) if isinstance(context, Mapping) else {})
    bots = dict((context.get("bot_swarm", {}) or {}).get("aggregate", {}) if isinstance(context, Mapping) else {})
    sqt = dict(context.get("sqt", {}) if isinstance(context, Mapping) else {})
    market = dict(context.get("market", {}) if isinstance(context, Mapping) else {})
    mfg = dict(context.get("mfg", {}) if isinstance(context, Mapping) else {})
    v18_paths = dict(context.get("v18_paths", {}) if isinstance(context, Mapping) else {})
    v21_runtime = dict(context.get("v21_runtime", {}) if isinstance(context, Mapping) else {})
    v22_runtime = dict(context.get("v22_runtime", {}) if isinstance(context, Mapping) else {})
    live_performance = dict(context.get("live_performance", {}) if isinstance(context, Mapping) else {})
    risk_controls = dict(context.get("risk_controls", {}) if isinstance(context, Mapping) else {})
    news = list(context.get("news_feed", []) if isinstance(context, Mapping) else [])
    crowd_items = list(context.get("public_discussions", []) if isinstance(context, Mapping) else [])
    top_news = []
    for item in news[:3]:
        if not isinstance(item, Mapping):
            continue
        title = _clean_text(item.get("title"))
        if not title:
            continue
        title = title.split(" - ")[0].strip()
        top_news.append(
            f"- {title} ({_clean_text(item.get('source', 'unknown'))}, sentiment {_signed_pct(item.get('sentiment', 0.0))})"
        )
    headlines = "\n".join(top_news) if top_news else "No significant headlines."
    top_crowd = []
    for item in crowd_items[:3]:
        if not isinstance(item, Mapping):
            continue
        title = _clean_text(item.get("title"))
        if not title:
            continue
        top_crowd.append(
            f"- {title} ({_clean_text(item.get('source', 'unknown'))}, sentiment {_signed_pct(item.get('sentiment', 0.0))})"
        )
    crowd_lines = "\n".join(top_crowd) if top_crowd else "No meaningful public-discussion items."
    entry_zone = sim.get("entry_zone") if isinstance(sim.get("entry_zone"), list) else []

    return f"""XAUUSD 15-minute simulation update for {symbol}.

You must produce:
1. a market_only_summary using ONLY live market/news/crowd and ignoring V18 simulator metrics
2. a v18_summary using ONLY the V18 simulator metrics and ignoring the live-news narrative
3. a combined_summary that merges both views into the final desk call

SIMULATION SUMMARY:
- Direction: {_clean_text(sim.get('scenario_bias', sim.get('direction', 'unknown'))).upper()}
- Confidence: {float(sim.get('overall_confidence', sim.get('cabr_score', 0.0)) or 0.0):.3f}
- CABR score: {float(sim.get('cabr_score', sim.get('selector_top_score', 0.0)) or 0.0):.3f}
- CPM score: {float(sim.get('cpm_score', 0.0) or 0.0):.3f}
- Regime: {_clean_text(sim.get('detected_regime', sim.get('market_memory_regime', 'unknown')))}
- Contradiction: {_clean_text(sim.get('contradiction_type', 'unknown'))}
- Cone width pips: {float(sim.get('cone_width_pips', 0.0) or 0.0):.1f}
- SQT: {_clean_text(sqt.get('label', sim.get('sqt_label', 'NEUTRAL')))} (accuracy {float(sqt.get('rolling_accuracy', sim.get('sqt_accuracy', 0.5)) or 0.5):.1%})

MARKET MEMORY:
- H overall: {float(sim.get('hurst_overall', 0.5) or 0.5):.3f}
- H asymmetry: {float(sim.get('hurst_asymmetry', 0.0) or 0.0):.3f}
- H+: {float(sim.get('cone_h_plus', sim.get('hurst_positive', 0.5)) or 0.5):.3f}
- H-: {float(sim.get('cone_h_minus', sim.get('hurst_negative', 0.5)) or 0.5):.3f}

CROWD STATE:
- Retail testosterone: {float(((sim.get('testosterone_index') or {}).get('retail', 0.0)) or 0.0):.4f}
- Institutional testosterone: {float(((sim.get('testosterone_index') or {}).get('institutional', 0.0)) or 0.0):.4f}
- MFG disagreement: {float(mfg.get('disagreement', 0.0) or 0.0):.6f}
- MFG consensus drift: {float(mfg.get('consensus_drift', 0.0) or 0.0):.6f}

TECHNICAL STRUCTURE:
- Structure: {_clean_text(tech.get('structure', 'unknown'))}
- Location: {_clean_text(tech.get('location', 'unknown'))}
- RSI(14): {float(tech.get('rsi_14', 50.0) or 50.0):.1f}
- ATR(14): {float(tech.get('atr_14', 0.0) or 0.0):.2f}
- Equilibrium: {_price(tech.get('equilibrium'))}
- Nearest support: {_support_or_resistance(tech.get('nearest_support'))}
- Nearest resistance: {_support_or_resistance(tech.get('nearest_resistance'))}
- Order blocks: {json.dumps(tech.get('order_blocks', [])[:3], ensure_ascii=True)}

BOT SWARM:
- Signal: {_clean_text(bots.get('signal', 'neutral')).upper()}
- Bullish probability: {float(bots.get('bullish_probability', 0.5) or 0.5):.1%}
- Disagreement: {float(bots.get('disagreement', 0.0) or 0.0):.3f}

V18 PATH SNAPSHOT:
- Consensus path: {json.dumps(v18_paths.get('consensus_path', []), ensure_ascii=True)}
- Minority path: {json.dumps(v18_paths.get('minority_path', []), ensure_ascii=True)}
- Outer upper path: {json.dumps(v18_paths.get('outer_upper', []), ensure_ascii=True)}
- Outer lower path: {json.dumps(v18_paths.get('outer_lower', []), ensure_ascii=True)}

CURRENT LEVELS:
- Current price: {_price(market.get('current_price'))}
- Suggested lot: {float(sim.get('suggested_lot', 0.0) or 0.0):.2f}
- Existing entry zone: {json.dumps(entry_zone, ensure_ascii=True)}

LIVE PERFORMANCE:
- Rolling win rate (10): {float(live_performance.get('rolling_win_rate_10', 0.0) or 0.0):.1%}
- Consecutive losses: {int(live_performance.get('consecutive_losses', 0) or 0)}
- Daily PnL: {float(live_performance.get('daily_pnl', 0.0) or 0.0):+.2f}
- Equity: {_price(live_performance.get('equity'))}
- Open positions: {int(live_performance.get('open_positions', 0) or 0)}

V21 LOCAL RUNTIME:
- Runtime version: {_clean_text(v21_runtime.get('runtime_version', 'unavailable'))}
- Raw stance: {_clean_text(v21_runtime.get('raw_stance', 'HOLD')).upper()}
- Should execute: {bool(v21_runtime.get('should_execute', False))}
- Confidence tier: {_clean_text(v21_runtime.get('confidence_tier', 'unknown'))}
- xLSTM 15m probability: {float(v21_runtime.get('v21_dir_15m_prob', 0.0) or 0.0):.3f}
- BiMamba probability: {float(v21_runtime.get('v21_bimamba_prob', 0.0) or 0.0):.3f}
- Ensemble probability: {float(v21_runtime.get('v21_ensemble_prob', 0.0) or 0.0):.3f}
- Meta-label probability: {float(v21_runtime.get('v21_meta_label_prob', 0.0) or 0.0):.3f}
- Disagreement probability: {float(v21_runtime.get('v21_disagree_prob', 0.0) or 0.0):.3f}
- Dangerous branches: {int(v21_runtime.get('v21_dangerous_branch_count', 0) or 0)}
- Regime label: {_clean_text(v21_runtime.get('v21_regime_label', 'unknown'))}
- Branch fallback used: {bool(v21_runtime.get('v21_used_branch_fallback', False))}
- Runtime reason: {_clean_text(v21_runtime.get('execution_reason', ''))}

V22 RISK STACK:
- Online HMM regime: {_clean_text(((v22_runtime.get('online_hmm') or {}).get('regime_label', 'unknown')))}
- Online HMM confidence: {float(((v22_runtime.get('online_hmm') or {}).get('regime_confidence', 0.0) or 0.0)):.3f}
- Persistence count: {int(((v22_runtime.get('online_hmm') or {}).get('persistence_count', 0) or 0))}
- Circuit breaker state: {_clean_text(((v22_runtime.get('circuit_breaker') or {}).get('state', 'CLEAR')))}
- Circuit trading allowed: {bool(((v22_runtime.get('circuit_breaker') or {}).get('trading_allowed', True)))}
- Circuit reasons: {json.dumps(((v22_runtime.get('circuit_breaker') or {}).get('reasons', []) or []), ensure_ascii=True)}
- Ensemble action: {_clean_text(((v22_runtime.get('ensemble') or {}).get('action', ''))).upper()}
- Ensemble agreement rate: {float(((v22_runtime.get('ensemble') or {}).get('agreement_rate', 0.0) or 0.0)):.3f}
- Meta-label accept probability: {float(((v22_runtime.get('ensemble') or {}).get('meta_label_prob', 0.0) or 0.0)):.3f}
- Ensemble risk score: {float(((v22_runtime.get('ensemble') or {}).get('risk_score', 0.0) or 0.0)):.3f}
- Proposed R:R ratio: {float(((v22_runtime.get('risk_check') or {}).get('rr_ratio', 0.0) or 0.0)):.3f}
- Broker autotrade enabled: {bool(((risk_controls.get('broker') or {}).get('autotrade_enabled', False)))}

KEY NEWS:
{headlines}

PUBLIC DISCUSSION:
{crowd_lines}

Provide your trade decision in the exact JSON format specified."""
