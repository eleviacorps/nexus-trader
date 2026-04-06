from __future__ import annotations

from html import unescape
import json
import re
from typing import Any, Mapping


KIMI_SYSTEM_PROMPT = """You are the meta-analyst for Nexus Trader, a 15-minute XAUUSD futures simulator.

Every 15 minutes you receive structured output from a crowd simulation engine.
The engine runs candidate future price paths, ranks them with a neural branch ranker,
and collapses them into a probability cone with a consensus path and a minority path.

Your role: synthesize this into a single clear trade decision for a human operator
watching a live XAUUSD chart.

NUMERIC GLOSSARY:
- scenario_bias: direction the simulation favors (bullish/bearish/neutral)
- overall_confidence: composite signal strength 0-1 (below 0.4 = weak signal)
- cabr_score: branch ranking accuracy proxy, 0.5 = random, 0.7+ = strong
- cpm_score: conditional predictability, 0.7+ = this bar is historically predictable
- hurst_overall: market persistence H. H > 0.55 = trend mode, H < 0.45 = reversal mode
- hurst_asymmetry: positive = up moves more persistent, negative = down moves stickier
- cone_width_pips: inner cone width in pips. Wide (> 300) = very uncertain market
- testosterone_index: retail crowd momentum intensity 0-1. High = euphoria/bubble risk
- sqt_label: HOT = simulator recently accurate, COLD = recently wrong, NEUTRAL = unknown
- contradiction_type: full_agreement = high confidence, full_disagreement = skip trade
- detected_regime: trending_up, trending_down, ranging, breakout, panic_shock
- bot_swarm.aggregate.signal: consensus of specialist trading bots
- bot_swarm.aggregate.disagreement: how much bots disagree (high = uncertain)
- technical_analysis.structure: bullish or bearish overall chart structure
- technical_analysis.location: premium (above equilibrium) or discount (below it)
- support/resistance levels: key price zones from the chart structure
- order_blocks: institutional order zones with strength scores
- news_feed: recent headlines with sentiment scores (-1 to +1)

RULES:
1. If sqt_label is COLD, stance MUST be HOLD. Do not trade when the simulator is wrong recently.
2. If contradiction_type is full_disagreement, stance MUST be HOLD.
3. If cone_width_pips > 400, set confidence to LOW or VERY_LOW.
4. If overall_confidence < 0.3, set confidence to VERY_LOW.
5. Give SPECIFIC price levels. Never say "around" or "approximately".
6. Stop loss should be beyond a structural level (support/resistance or order block).
7. Take profit should target the next structural level.
8. The default hold time is the current 15-minute bar.

OUTPUT FORMAT - respond with ONLY this JSON, no other text:
{
  "stance": "BUY",
  "confidence": "MODERATE",
  "entry_zone": [4668.50, 4672.00],
  "stop_loss": 4655.00,
  "take_profit": 4691.00,
  "hold_time": "current_bar",
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
    news = list(context.get("news_feed", []) if isinstance(context, Mapping) else [])
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
    entry_zone = sim.get("entry_zone") if isinstance(sim.get("entry_zone"), list) else []

    return f"""XAUUSD 15-minute simulation update for {symbol}.

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

CURRENT LEVELS:
- Current price: {_price(market.get('current_price'))}
- Suggested lot: {float(sim.get('suggested_lot', 0.0) or 0.0):.2f}
- Existing entry zone: {json.dumps(entry_zone, ensure_ascii=True)}

KEY NEWS:
{headlines}

Provide your trade decision in the exact JSON format specified."""
