from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from src.v20.runtime import build_v20_live_bundle
from src.v21.inference import run_v21_bimamba_inference, run_v21_xlstm_inference
from src.v21.runtime_v21 import V21Runtime


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return float(default)
        return float(number)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _confidence_tier(ensemble_prob: float, meta_label_prob: float, disagree_prob: float, cone_width_pips: float) -> str:
    edge = abs(float(ensemble_prob) - 0.5) * 2.0
    if edge >= 0.38 and meta_label_prob >= 0.62 and disagree_prob <= 0.10 and cone_width_pips <= 180.0:
        return "very_high"
    if edge >= 0.28 and meta_label_prob >= 0.52 and disagree_prob <= 0.16 and cone_width_pips <= 260.0:
        return "high"
    if edge >= 0.18 and meta_label_prob >= 0.42 and disagree_prob <= 0.24 and cone_width_pips <= 380.0:
        return "moderate"
    if edge >= 0.10 and meta_label_prob >= 0.34:
        return "low"
    return "very_low"


def _stance_from_prob(probability: float) -> str:
    if probability >= 0.56:
        return "BUY"
    if probability <= 0.44:
        return "SELL"
    return "HOLD"


def _regime_label(regime_probs: list[float]) -> str:
    labels = [
        "balanced_range",
        "trend_up",
        "trend_down",
        "volatile_breakout",
        "mean_reversion",
        "macro_shock",
    ]
    if not regime_probs:
        return "unknown"
    best_index = int(np.argmax(np.asarray(regime_probs, dtype=np.float32)))
    return labels[best_index] if 0 <= best_index < len(labels) else "unknown"


def build_v21_runtime_state(payload: Mapping[str, Any], *, mode: str = "frequency") -> dict[str, Any]:
    bundle = build_v20_live_bundle(payload)
    if bundle is None or bundle.branches.empty:
        return {"available": False, "runtime_version": "v21_local", "execution_reason": "insufficient_live_bars"}

    x_result = run_v21_xlstm_inference(payload)
    b_result = run_v21_bimamba_inference(payload)
    if not x_result.get("available", False) and not b_result.get("available", False):
        return {
            "available": False,
            "runtime_version": "v21_local",
            "execution_reason": f"xLSTM={x_result.get('error', 'unavailable')} | BiMamba={b_result.get('error', 'unavailable')}",
        }

    latest = dict(bundle.feature_row)
    current_price = _safe_float(latest.get("close"), _safe_float(((payload.get("market") or {}).get("current_price")), 0.0))
    x_prob = _safe_float(x_result.get("dir_15m_prob"), 0.5)
    b_prob = _safe_float(b_result.get("dir_15m_prob"), x_prob)
    ensemble_prob = _clip01((0.65 * x_prob) + (0.35 * b_prob))
    disagree_prob = _clip01(abs(x_prob - b_prob))
    raw_stance = _stance_from_prob(ensemble_prob)
    top_branch = bundle.branches.iloc[0].to_dict()
    branch_stance = str(top_branch.get("decision_direction", "HOLD")).upper()
    frequency_mode = str(mode).lower() == "frequency"
    used_branch_fallback = False
    if frequency_mode and raw_stance == "HOLD" and branch_stance in {"BUY", "SELL"}:
        raw_stance = branch_stance
        used_branch_fallback = True
    top_cabr = _safe_float(top_branch.get("v20_cabr_score"), 0.5)
    cone_width = abs(bundle.cone_upper[-1] - bundle.cone_lower[-1]) / 0.1 if bundle.cone_upper and bundle.cone_lower else 0.0
    conformal_confidence = _clip01(bundle.confidence)
    atr_value = max(_safe_float(latest.get("atr_14"), current_price * 0.0015), current_price * 0.0005, 0.1)
    top_branch_prices = bundle.branches.head(10)["predicted_price_15m"].astype(float).tolist()
    dangerous_branch_count = 0
    for price in top_branch_prices:
        displacement = price - current_price
        if raw_stance == "BUY" and displacement < -(0.80 * atr_value):
            dangerous_branch_count += 1
        elif raw_stance == "SELL" and displacement > (0.80 * atr_value):
            dangerous_branch_count += 1
        elif raw_stance == "HOLD" and abs(displacement) > (1.20 * atr_value):
            dangerous_branch_count += 1
    meta_label_prob = _clip01(
        (0.30 * (abs(ensemble_prob - 0.5) * 2.0))
        + (0.25 * conformal_confidence)
        + (0.20 * top_cabr)
        + (0.15 * (1.0 - disagree_prob))
        + (0.10 * (1.0 - min(dangerous_branch_count, 5) / 5.0))
    )
    runtime = V21Runtime(mode="research" if frequency_mode else "production")
    should_execute, failed_gates = runtime.should_trade(
        sjd_output={"stance": raw_stance, "disagree_prob": disagree_prob},
        conformal_confidence=conformal_confidence,
        dangerous_branch_count=dangerous_branch_count,
        meta_label_prob=meta_label_prob,
    )
    confidence_tier = _confidence_tier(ensemble_prob, meta_label_prob, disagree_prob, cone_width)
    kelly_fraction = _clip01((abs(ensemble_prob - 0.5) * 2.0) * conformal_confidence * (1.0 - disagree_prob) * 0.35)
    paper_state = payload.get("paper_trading") or {}
    paper_summary = paper_state.get("summary", {}) if isinstance(paper_state, Mapping) else {}
    account_equity = max(_safe_float(paper_summary.get("equity"), 1000.0), 100.0)
    suggested_lot = runtime.get_size(kelly_fraction=kelly_fraction, account_balance=account_equity, price=max(current_price, 1.0))
    cpm_score = _clip01(
        (0.28 * _safe_float(latest.get("quant_route_confidence"), 0.5))
        + (0.18 * _safe_float(latest.get("quant_regime_strength"), 0.5))
        + (0.16 * conformal_confidence)
        + (0.14 * (1.0 - min(abs(_safe_float(latest.get("mfg_disagreement"), 0.0)), 1.0)))
        + (0.24 * (abs(ensemble_prob - 0.5) * 2.0))
    )
    final_direction = raw_stance if should_execute else "HOLD"
    regime_probs = [float(item) for item in (x_result.get("regime_probs") or bundle.regime_probs or [])]
    sqt_label = "GOOD" if meta_label_prob >= 0.58 and confidence_tier in {"high", "very_high"} else "NEUTRAL" if meta_label_prob >= 0.38 else "CAUTION"
    fallback_note = f", branch_fallback={branch_stance}" if used_branch_fallback else ""
    execution_reason = (
        f"V21 local execution cleared: stance={raw_stance}, ensemble={ensemble_prob:.3f}, conformal={conformal_confidence:.3f}, meta={meta_label_prob:.3f}, dangerous_branches={dangerous_branch_count}{fallback_note}."
        if should_execute
        else f"V21 local hold: raw_stance={raw_stance}, failed_gates={','.join(failed_gates) if failed_gates else 'none'}, ensemble={ensemble_prob:.3f}, conformal={conformal_confidence:.3f}, meta={meta_label_prob:.3f}, disagree={disagree_prob:.3f}{fallback_note}."
    )
    return {
        "available": True,
        "runtime_version": "v21_local",
        "selected_branch_id": int(top_branch.get("branch_id", 0)),
        "selected_branch_label": str(top_branch.get("branch_label", "v21_branch")),
        "decision_direction": final_direction,
        "raw_stance": raw_stance,
        "cabr_score": round(top_cabr, 6),
        "cabr_raw_score": round(_safe_float(top_branch.get("v20_cabr_raw_score"), top_cabr), 6),
        "cpm_score": round(cpm_score, 6),
        "confidence_tier": confidence_tier,
        "sqt_label": sqt_label,
        "cone_width_pips": round(float(cone_width), 3),
        "lepl_action": raw_stance,
        "lepl_probabilities": {
            "execute": round(meta_label_prob, 6),
            "hold": round(1.0 - meta_label_prob, 6),
        },
        "lepl_features": {
            "kelly_fraction": round(kelly_fraction, 6),
            "suggested_lot": round(float(suggested_lot), 4),
            "conformal_confidence": round(conformal_confidence, 6),
            "dangerous_branch_count": int(dangerous_branch_count),
            "paper_equity": round(account_equity, 2),
        },
        "should_execute": bool(should_execute),
        "execution_reason": execution_reason,
        "branch_scores": [
            {
                "branch_id": int(item.get("branch_id", 0)),
                "branch_label": str(item.get("branch_label", "")),
                "decision_direction": str(item.get("decision_direction", "HOLD")),
                "cabr_score": round(_safe_float(item.get("v20_cabr_score"), 0.0), 6),
                "cabr_raw_score": round(_safe_float(item.get("v20_cabr_raw_score"), 0.0), 6),
            }
            for item in bundle.branches.head(8).to_dict(orient="records")
        ],
        "consensus_path": bundle.consensus_path,
        "minority_path": bundle.minority_path,
        "cone_upper": bundle.cone_upper,
        "cone_lower": bundle.cone_lower,
        "regime_probs": [round(float(value), 6) for value in regime_probs],
        "v21_mode": runtime.mode,
        "v21_dir_15m_prob": round(x_prob, 6),
        "v21_bimamba_prob": round(b_prob, 6),
        "v21_ensemble_prob": round(ensemble_prob, 6),
        "v21_disagree_prob": round(disagree_prob, 6),
        "v21_meta_label_prob": round(meta_label_prob, 6),
        "v21_dangerous_branch_count": int(dangerous_branch_count),
        "v21_top_vsn_features": list(x_result.get("top_vsn_features", []) or []),
        "v21_regime_label": _regime_label(regime_probs),
        "v21_used_branch_fallback": bool(used_branch_fallback),
    }


def build_v21_local_judge(payload: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    if not runtime.get("available", False):
        return {
            "available": False,
            "provider": "local_v21",
            "model": "v21_xlstm_bimamba",
            "error": runtime.get("execution_reason", "missing_v21_runtime"),
            "content": {
                "stance": "HOLD",
                "confidence": "VERY_LOW",
                "final_call": "SKIP",
                "final_summary": "SKIP - local V21 is unavailable for this bar.",
                "entry_zone": [],
                "stop_loss": None,
                "take_profit": None,
                "hold_time": "15m",
                "market_only_summary": {"call": "SKIP", "summary": "Local V21 is unavailable.", "reasoning": str(runtime.get("execution_reason", "Runtime unavailable."))},
                "v18_summary": {"call": "SKIP", "summary": "V21 local runtime unavailable.", "reasoning": "The local V21 models could not score this bar."},
                "combined_summary": {"call": "SKIP", "summary": "No local V21 decision.", "reasoning": "Wait for more bars or valid checkpoints."},
                "reasoning": str(runtime.get("execution_reason", "Runtime unavailable.")),
                "key_risk": "Local V21 could not generate a stable score for the current bar.",
                "crowd_note": "Crowd note unavailable.",
                "regime_note": "Regime note unavailable.",
                "invalidation": None,
                "kelly_fraction": 0.0,
            },
        }

    market = payload.get("market") or {}
    technical = payload.get("technical_analysis") or {}
    feeds = payload.get("feeds") or {}
    current_price = _safe_float(market.get("current_price"), 0.0)
    raw_stance = str(runtime.get("raw_stance", "HOLD")).upper()
    final_call = raw_stance if runtime.get("should_execute", False) and raw_stance in {"BUY", "SELL"} else "SKIP"
    confidence = str(runtime.get("confidence_tier", "very_low")).upper()
    atr_14 = max(_safe_float(technical.get("atr_14"), current_price * 0.0015), 0.1)
    stop_distance = max(atr_14 * 0.65, 1.8)
    target_distance = max(atr_14 * 1.10, 3.0)
    if raw_stance == "BUY":
        stop_loss = round(current_price - stop_distance, 2)
        take_profit = round(current_price + target_distance, 2)
        entry_zone = [round(current_price - (atr_14 * 0.10), 2), round(current_price + (atr_14 * 0.15), 2)]
    elif raw_stance == "SELL":
        stop_loss = round(current_price + stop_distance, 2)
        take_profit = round(current_price - target_distance, 2)
        entry_zone = [round(current_price - (atr_14 * 0.15), 2), round(current_price + (atr_14 * 0.10), 2)]
    else:
        stop_loss = None
        take_profit = None
        entry_zone = []
    fear_greed = feeds.get("fear_greed", {}) if isinstance(feeds, Mapping) else {}
    vsn_notes = list(runtime.get("v21_top_vsn_features", []) or [])[:3]
    top_features = ", ".join(f"{item.get('feature')} {float(item.get('weight', 0.0)):.3f}" for item in vsn_notes if item.get("feature")) or "top features unavailable"
    regime_note = f"Local V21 regime reads {runtime.get('v21_regime_label', 'unknown')} with xLSTM regime probabilities {runtime.get('regime_probs', [])}."
    crowd_note = f"Fear/Greed is {fear_greed.get('classification', 'unknown')} ({fear_greed.get('value', 'n/a')}) and VSN focus is {top_features}."
    reasoning = (
        f"xLSTM={_safe_float(runtime.get('v21_dir_15m_prob'), 0.5):.3f}, "
        f"BiMamba={_safe_float(runtime.get('v21_bimamba_prob'), 0.5):.3f}, "
        f"ensemble={_safe_float(runtime.get('v21_ensemble_prob'), 0.5):.3f}, "
        f"conformal={_safe_float((runtime.get('lepl_features') or {}).get('conformal_confidence'), 0.0):.3f}, "
        f"meta={_safe_float(runtime.get('v21_meta_label_prob'), 0.0):.3f}, "
        f"dangerous_branches={int(runtime.get('v21_dangerous_branch_count', 0))}."
    )
    combined_summary_text = (
        f"Local V21 would execute {raw_stance}." if final_call in {"BUY", "SELL"} else f"Local V21 prefers {raw_stance} structurally but skips execution on this bar."
    )
    return {
        "available": True,
        "provider": "local_v21",
        "model": "v21_xlstm_bimamba",
        "content": {
            "stance": raw_stance,
            "confidence": confidence,
            "final_call": final_call,
            "final_summary": f"{final_call} - {combined_summary_text}",
            "entry_zone": entry_zone,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "hold_time": "15m",
            "market_only_summary": {
                "call": raw_stance if raw_stance in {"BUY", "SELL"} else "SKIP",
                "summary": f"Pure local V21 market-model read is {raw_stance}.",
                "reasoning": reasoning,
            },
            "v18_summary": {
                "call": raw_stance if raw_stance in {"BUY", "SELL"} else "SKIP",
                "summary": "This block is now backed by the local V21 stack, not Kimi.",
                "reasoning": f"Top branch CABR is {_safe_float(runtime.get('cabr_score'), 0.0):.3f}, CPM is {_safe_float(runtime.get('cpm_score'), 0.0):.3f}, and cone width is {_safe_float(runtime.get('cone_width_pips'), 0.0):.1f} pips.",
            },
            "combined_summary": {
                "call": final_call,
                "summary": combined_summary_text,
                "reasoning": f"Execution gate says should_execute={bool(runtime.get('should_execute', False))}. {str(runtime.get('execution_reason', ''))}",
            },
            "reasoning": reasoning,
            "key_risk": (
                f"Model disagreement is {_safe_float(runtime.get('v21_disagree_prob'), 0.0):.3f} and dangerous branch count is {int(runtime.get('v21_dangerous_branch_count', 0))}."
            ),
            "crowd_note": crowd_note,
            "regime_note": regime_note,
            "invalidation": stop_loss,
            "kelly_fraction": round(_safe_float((runtime.get("lepl_features") or {}).get("kelly_fraction"), 0.0), 6),
        },
    }


__all__ = ["build_v21_local_judge", "build_v21_runtime_state"]
