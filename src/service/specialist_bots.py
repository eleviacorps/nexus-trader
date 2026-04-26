from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SpecialistBotResult:
    bot_id: str
    name: str
    style: str
    direction: str
    confidence: float
    bullish_probability: float
    key_level: float
    invalidation: float
    emotion: str
    reason: str
    horizons: list[dict[str, Any]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return default
        return number
    except Exception:
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _signal_from_score(score: float) -> tuple[str, float]:
    bullish_probability = _clamp(0.5 + 0.5 * np.tanh(score))
    direction = "bullish" if bullish_probability >= 0.53 else "bearish" if bullish_probability <= 0.47 else "neutral"
    confidence = abs(bullish_probability - 0.5) * 2.0
    return direction, confidence


def _build_horizons(current_price: float, atr: float, score: float, confidence: float) -> list[dict[str, Any]]:
    horizons = []
    for minutes, scale in [(5, 0.55), (10, 0.95), (15, 1.25), (30, 1.90)]:
        move = atr * scale * np.tanh(score) * (0.45 + 0.55 * confidence)
        target = current_price + move
        bullish_probability = _clamp(0.5 + 0.5 * np.tanh(score * scale))
        horizons.append(
            {
                "minutes": minutes,
                "target_price": round(float(target), 5),
                "bullish_probability": round(float(bullish_probability), 6),
            }
        )
    return horizons


def _bot_result(
    bot_id: str,
    name: str,
    style: str,
    score: float,
    current_price: float,
    atr: float,
    key_level: float,
    invalidation: float,
    emotion: str,
    reason: str,
) -> SpecialistBotResult:
    direction, confidence = _signal_from_score(score)
    probability = _clamp(0.5 + 0.5 * np.tanh(score))
    return SpecialistBotResult(
        bot_id=bot_id,
        name=name,
        style=style,
        direction=direction,
        confidence=round(float(confidence), 6),
        bullish_probability=round(float(probability), 6),
        key_level=round(float(key_level), 5),
        invalidation=round(float(invalidation), 5),
        emotion=emotion,
        reason=reason,
        horizons=_build_horizons(current_price, atr, score, confidence),
    )


def _nearest_level(level: Mapping[str, Any] | None, fallback: float) -> float:
    if not level:
        return fallback
    return _safe_float(level.get("price"), fallback)


def run_specialist_bots(
    symbol: str,
    current_row: Mapping[str, Any],
    technical_analysis: Mapping[str, Any],
    feeds: Mapping[str, Any],
    llm_content: Mapping[str, Any],
) -> dict[str, Any]:
    current_price = _safe_float(current_row.get("close"), 0.0)
    atr = max(_safe_float(current_row.get("atr_14"), current_price * 0.0015), max(current_price * 0.001, 0.25))
    ema_cross = _safe_float(current_row.get("ema_cross"), 0.0)
    rsi_14 = _safe_float(current_row.get("rsi_14"), 50.0)
    rsi_7 = _safe_float(current_row.get("rsi_7"), 50.0)
    macd_hist = _safe_float(current_row.get("macd_hist"), 0.0)
    bb_pct = _safe_float(current_row.get("bb_pct"), 0.5)
    macro_bias = _safe_float(current_row.get("macro_bias"), 0.0)
    macro_shock = _safe_float(current_row.get("macro_shock"), 0.0)
    news_bias = _safe_float(current_row.get("news_bias"), 0.0)
    crowd_bias = _safe_float(current_row.get("crowd_bias"), 0.0)
    crowd_extreme = _safe_float(current_row.get("crowd_extreme"), 0.0)
    llm_institutional = _safe_float(llm_content.get("institutional_bias"), 0.0)
    llm_whale = _safe_float(llm_content.get("whale_bias"), 0.0)
    llm_retail = _safe_float(llm_content.get("retail_bias"), 0.0)

    nearest_support = _nearest_level(technical_analysis.get("nearest_support"), current_price - atr)
    nearest_resistance = _nearest_level(technical_analysis.get("nearest_resistance"), current_price + atr)
    equilibrium = _safe_float(technical_analysis.get("equilibrium"), current_price)
    order_blocks = list(technical_analysis.get("order_blocks", []) or [])
    fair_value_gaps = list(technical_analysis.get("fair_value_gaps", []) or [])
    headlines = list(feeds.get("news", []) or [])
    public_discussions = list(feeds.get("public_discussions", []) or [])

    latest_bullish_ob = next((item for item in reversed(order_blocks) if str(item.get("type", "")).startswith("bullish")), None)
    latest_bearish_ob = next((item for item in reversed(order_blocks) if str(item.get("type", "")).startswith("bearish")), None)
    latest_bullish_fvg = next((item for item in reversed(fair_value_gaps) if str(item.get("type", "")).startswith("bullish")), None)
    latest_bearish_fvg = next((item for item in reversed(fair_value_gaps) if str(item.get("type", "")).startswith("bearish")), None)

    bots: list[SpecialistBotResult] = []

    trend_score = (1.20 * ema_cross) + (0.80 * macd_hist) + ((rsi_14 - 50.0) / 18.0)
    bots.append(
        _bot_result(
            "trend",
            "Trend Bot",
            "trend_following",
            trend_score,
            current_price,
            atr,
            current_price,
            current_price - atr if trend_score >= 0 else current_price + atr,
            "momentum",
            "EMA structure and momentum alignment are driving this trend view.",
        )
    )

    breakout_score = ((current_price - nearest_resistance) / atr) * -1.2 + ((current_price - nearest_support) / atr) * 0.3 + abs(macd_hist) * np.sign(ema_cross or macd_hist)
    bots.append(
        _bot_result(
            "breakout",
            "Breakout Bot",
            "breakout",
            breakout_score,
            current_price,
            atr,
            nearest_resistance if breakout_score >= 0 else nearest_support,
            nearest_support if breakout_score >= 0 else nearest_resistance,
            "anticipation",
            "This bot watches compression near swing levels for a range expansion move.",
        )
    )

    mean_rev_score = ((0.5 - bb_pct) * 2.8) + ((50.0 - rsi_14) / 28.0)
    bots.append(
        _bot_result(
            "mean_reversion",
            "Mean Reversion Bot",
            "mean_reversion",
            mean_rev_score,
            current_price,
            atr,
            equilibrium,
            current_price + atr if mean_rev_score < 0 else current_price - atr,
            "patience",
            "This bot leans toward snapback behavior when price stretches away from balance.",
        )
    )

    orderblock_score = 0.0
    orderblock_level = equilibrium
    invalidation = equilibrium
    if latest_bullish_ob:
        orderblock_score = 0.55 + _safe_float(latest_bullish_ob.get("strength"), 0.0)
        orderblock_level = _safe_float(latest_bullish_ob.get("high"), equilibrium)
        invalidation = _safe_float(latest_bullish_ob.get("low"), orderblock_level - atr)
    if latest_bearish_ob and (
        not latest_bullish_ob
        or abs(current_price - _safe_float(latest_bearish_ob.get("high"), current_price)) < abs(current_price - orderblock_level)
    ):
        orderblock_score = -(0.55 + _safe_float(latest_bearish_ob.get("strength"), 0.0))
        orderblock_level = _safe_float(latest_bearish_ob.get("low"), equilibrium)
        invalidation = _safe_float(latest_bearish_ob.get("high"), orderblock_level + atr)
    bots.append(
        _bot_result(
            "order_block",
            "Order Block Bot",
            "order_block",
            orderblock_score,
            current_price,
            atr,
            orderblock_level,
            invalidation,
            "discipline",
            "This bot reacts to the nearest validated order block zone and displacement behavior.",
        )
    )

    liquidity_score = ((nearest_resistance - current_price) - (current_price - nearest_support)) / max(atr * 2.0, 1e-6)
    liquidity_score *= -1.0 if crowd_bias > 0 else 1.0
    bots.append(
        _bot_result(
            "liquidity_sweep",
            "Liquidity Sweep Bot",
            "liquidity_sweep",
            liquidity_score,
            current_price,
            atr,
            nearest_resistance if liquidity_score >= 0 else nearest_support,
            nearest_support if liquidity_score >= 0 else nearest_resistance,
            "contrarian",
            "This bot looks for stop runs and the side of the book most likely to be swept first.",
        )
    )

    fvg_score = 0.0
    fvg_level = equilibrium
    fvg_invalidation = equilibrium
    if latest_bullish_fvg:
        fvg_score = 0.45 + (_safe_float(latest_bullish_fvg.get("size"), 0.0) / max(atr, 1e-6))
        fvg_level = _safe_float(latest_bullish_fvg.get("high"), equilibrium)
        fvg_invalidation = _safe_float(latest_bullish_fvg.get("low"), fvg_level - atr)
    if latest_bearish_fvg and (
        not latest_bullish_fvg
        or abs(current_price - _safe_float(latest_bearish_fvg.get("low"), current_price)) < abs(current_price - fvg_level)
    ):
        fvg_score = -(0.45 + (_safe_float(latest_bearish_fvg.get("size"), 0.0) / max(atr, 1e-6)))
        fvg_level = _safe_float(latest_bearish_fvg.get("low"), equilibrium)
        fvg_invalidation = _safe_float(latest_bearish_fvg.get("high"), fvg_level + atr)
    bots.append(
        _bot_result(
            "fvg",
            "Fair Value Gap Bot",
            "fair_value_gap",
            fvg_score,
            current_price,
            atr,
            fvg_level,
            fvg_invalidation,
            "imbalance",
            "This bot tracks imbalance gaps and expects price to respect or refill them.",
        )
    )

    macro_score = (1.2 * macro_bias) + (0.6 * llm_institutional) - (0.25 * macro_shock * np.sign(macro_bias or 1.0))
    bots.append(
        _bot_result(
            "macro",
            "Macro Regime Bot",
            "macro_regime",
            macro_score,
            current_price,
            atr,
            equilibrium,
            equilibrium - atr if macro_score >= 0 else equilibrium + atr,
            "conviction",
            "This bot aligns with the prevailing macro regime and institutional narrative bias.",
        )
    )

    average_news_sentiment = np.mean([_safe_float(item.get("sentiment"), 0.0) for item in headlines]) if headlines else news_bias
    news_score = (0.9 * average_news_sentiment) + (0.55 * news_bias) + (0.20 * macro_shock * np.sign(average_news_sentiment or 1.0))
    bots.append(
        _bot_result(
            "news",
            "News Shock Bot",
            "news_shock",
            news_score,
            current_price,
            atr,
            current_price,
            current_price - atr if news_score >= 0 else current_price + atr,
            "urgency",
            "This bot reacts to headline tone and shock intensity in the current tape.",
        )
    )

    discussion_sentiment = np.mean([_safe_float(item.get("sentiment"), 0.0) for item in public_discussions]) if public_discussions else crowd_bias
    crowd_score = -(0.95 * discussion_sentiment * max(crowd_extreme, 0.35)) + (0.25 * llm_retail)
    bots.append(
        _bot_result(
            "crowd",
            "Crowd Extremes Bot",
            "crowd_extremes",
            crowd_score,
            current_price,
            atr,
            nearest_support if crowd_score >= 0 else nearest_resistance,
            nearest_resistance if crowd_score >= 0 else nearest_support,
            "emotion",
            "This bot fades one-sided crowd emotion and watches for exhaustion or squeeze behavior.",
        )
    )

    skeptic_score = -(0.45 * abs(trend_score)) + (0.60 * macro_shock) - (0.35 * crowd_extreme)
    bots.append(
        _bot_result(
            "skeptic",
            "Risk Skeptic Bot",
            "skeptic_filter",
            skeptic_score,
            current_price,
            atr,
            equilibrium,
            current_price + atr if skeptic_score < 0 else current_price - atr,
            "caution",
            "This bot resists crowded conviction and looks for reasons the move could fail.",
        )
    )

    results = [bot.__dict__ for bot in bots]
    weights = {
        "trend": 1.00,
        "breakout": 0.90,
        "mean_reversion": 0.85,
        "order_block": 1.10,
        "liquidity_sweep": 1.05,
        "fvg": 0.95,
        "macro": 1.10,
        "news": 0.95,
        "crowd": 0.90,
        "skeptic": 0.75,
    }

    total_weight = sum(weights.get(bot["bot_id"], 1.0) * max(bot["confidence"], 0.15) for bot in results)
    weighted_probability = sum(
        weights.get(bot["bot_id"], 1.0) * max(bot["confidence"], 0.15) * bot["bullish_probability"] for bot in results
    ) / max(total_weight, 1e-6)
    disagreement = float(np.std([bot["bullish_probability"] for bot in results], ddof=0)) if len(results) > 1 else 0.0
    signal = "bullish" if weighted_probability >= 0.53 else "bearish" if weighted_probability <= 0.47 else "neutral"

    horizon_predictions = []
    for minutes in [5, 10, 15, 30]:
        horizon_probs = [next(item["bullish_probability"] for item in bot["horizons"] if item["minutes"] == minutes) for bot in results]
        horizon_targets = [next(item["target_price"] for item in bot["horizons"] if item["minutes"] == minutes) for bot in results]
        horizon_predictions.append(
            {
                "minutes": minutes,
                "bullish_probability": round(float(np.mean(horizon_probs)), 6),
                "target_price": round(float(np.mean(horizon_targets)), 5),
            }
        )

    style_groups = {
        "trend": {"trend_following", "breakout"},
        "reversal": {"mean_reversion", "crowd_extremes", "skeptic_filter"},
        "structure": {"order_block", "fair_value_gap", "liquidity_sweep"},
        "macro": {"macro_regime", "news_shock"},
        "shock": {"news_shock", "skeptic_filter"},
        "crowd": {"crowd_extremes"},
    }
    style_biases: dict[str, dict[str, float]] = {}
    for group_name, preferred_styles in style_groups.items():
        members = [bot for bot in results if bot["style"] in preferred_styles]
        if not members:
            style_biases[group_name] = {"bullish_probability": 0.5, "confidence": 0.0, "bias": 0.0}
            continue
        member_weight = np.asarray([max(_safe_float(bot["confidence"], 0.0), 0.15) for bot in members], dtype=np.float32)
        member_prob = np.asarray([_safe_float(bot["bullish_probability"], 0.5) for bot in members], dtype=np.float32)
        weighted_member_prob = float(np.average(member_prob, weights=member_weight))
        style_biases[group_name] = {
            "bullish_probability": round(weighted_member_prob, 6),
            "confidence": round(float(np.average(member_weight)), 6),
            "bias": round(float((weighted_member_prob - 0.5) * 2.0), 6),
        }

    regime_affinity = {
        "trend": round(float((0.50 * style_biases["trend"]["confidence"]) + (0.30 * abs(style_biases["trend"]["bias"])) + (0.20 * abs(style_biases["structure"]["bias"]))), 6),
        "reversal": round(float((0.55 * style_biases["reversal"]["confidence"]) + (0.25 * abs(style_biases["reversal"]["bias"])) + (0.20 * style_biases["crowd"]["confidence"])), 6),
        "macro_shock": round(float((0.55 * style_biases["macro"]["confidence"]) + (0.45 * style_biases["shock"]["confidence"])), 6),
        "balanced": round(float(1.0 - min(1.0, abs(weighted_probability - 0.5) * 1.8 + disagreement)), 6),
    }

    style_map = {
        "retail": {"trend_following", "breakout", "crowd_extremes", "news_shock"},
        "institutional": {"macro_regime", "order_block", "liquidity_sweep"},
        "algo": {"breakout", "mean_reversion", "fair_value_gap", "trend_following"},
        "whale": {"liquidity_sweep", "order_block", "macro_regime"},
        "noise": {"crowd_extremes", "news_shock"},
    }
    persona_reactions: dict[str, list[dict[str, Any]]] = {}
    for persona, preferred_styles in style_map.items():
        persona_reactions[persona] = [
            {
                "bot_id": bot["bot_id"],
                "name": bot["name"],
                "weight": round(1.0 if bot["style"] in preferred_styles else 0.35, 4),
                "direction": bot["direction"],
                "confidence": bot["confidence"],
            }
            for bot in results
        ]

    graph_nodes = [
        {
            "id": "simulator",
            "label": "Simulator",
            "group": "core",
            "x": 0.0,
            "y": 0.0,
            "size": 26,
            "color": "#4da3ff",
        }
    ]
    edges = []
    left_x = -1.4
    right_x = 1.35
    for index, bot in enumerate(results):
        y = 1.25 - (index * 0.28)
        color = "#2ecc71" if bot["direction"] == "bullish" else "#ff5a5f" if bot["direction"] == "bearish" else "#f1c40f"
        graph_nodes.append(
            {
                "id": bot["bot_id"],
                "label": bot["name"],
                "group": "bot",
                "x": left_x,
                "y": y,
                "size": 12 + (bot["confidence"] * 14),
                "color": color,
            }
        )
        edges.append(
            {
                "source": bot["bot_id"],
                "target": "simulator",
                "weight": round(max(bot["confidence"], 0.15), 4),
                "agreement": round((bot["bullish_probability"] - 0.5) * 2.0, 4),
            }
        )
    persona_positions = [("institutional", 0.65), ("whale", 0.25), ("algo", -0.15), ("retail", -0.55), ("noise", -0.95)]
    for persona, y in persona_positions:
        graph_nodes.append(
            {
                "id": persona,
                "label": persona.title(),
                "group": "persona",
                "x": right_x,
                "y": y,
                "size": 18,
                "color": "#9b59b6" if persona in {"institutional", "whale"} else "#16a085",
            }
        )
        for reaction in persona_reactions[persona][:4]:
            edges.append(
                {
                    "source": reaction["bot_id"],
                    "target": persona,
                    "weight": reaction["weight"],
                    "agreement": 1.0 if reaction["direction"] != "neutral" else 0.0,
                }
            )
        edges.append({"source": persona, "target": "simulator", "weight": 0.65, "agreement": 0.6})

    return {
        "symbol": symbol,
        "bots": results,
        "aggregate": {
            "bullish_probability": round(float(weighted_probability), 6),
            "bearish_probability": round(float(1.0 - weighted_probability), 6),
            "signal": signal,
            "confidence": round(float(max(0.0, min(1.0, (1.0 - min(disagreement / 0.25, 1.0)) * np.mean([bot["confidence"] for bot in results])))), 6),
            "disagreement": round(float(disagreement), 6),
            "horizon_predictions": horizon_predictions,
            "style_biases": style_biases,
            "regime_affinity": regime_affinity,
        },
        "persona_reactions": persona_reactions,
        "graph": {"nodes": graph_nodes, "edges": edges},
    }
