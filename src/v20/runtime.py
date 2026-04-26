from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.project_config import V20_CONFORMAL_CONE_PATH, V20_HMM_MODEL_PATH
from src.v16.confidence_tier import ConfidenceTier, classify_confidence
from src.v16.sel import sel_lot_size
from src.v20.cabr_v20 import heuristic_branch_scores
from src.v20.conformal_cone import ConformalCone
from src.v20.feature_builder import build_v20_feature_frame
from src.v20.regime_detector import RegimeDetector
from src.v20.rl_executor import HierarchicalExecutor
from src.v20.sjd_v20 import latest_sjd_decision


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _price_frame_from_payload(payload: Mapping[str, Any]) -> pd.DataFrame:
    candles = list(((payload.get("market") or {}).get("candles") or []))
    if not candles:
        candles = list(((payload.get("realtime_chart") or {}).get("candles") or []))
    frame = pd.DataFrame(candles)
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce").ffill().bfill().fillna(0.0)
    return frame[["open", "high", "low", "close", "volume"]]


def _mode_multiplier(mode: str) -> float:
    return 0.90 if str(mode).lower() == "precision" else 1.00


def _direction_signal(row: Mapping[str, Any]) -> float:
    components = [
        0.24 * np.tanh(_safe_float(row.get("macro_trend_strength"), 0.0) * 5.0),
        0.20 * np.tanh(_safe_float(row.get("quant_trend_score"), 0.0) * 3.0),
        0.18 * ((_safe_float(row.get("rsi_14"), 50.0) - 50.0) / 50.0),
        0.16 * (_safe_float(row.get("hurst_overall"), 0.5) - 0.5) * 2.0,
        0.12 * (_safe_float(row.get("mfg_mean_belief"), 0.0) * 250.0),
        0.10 * (_safe_float(row.get("roc_15m"), 0.0) * 50.0),
    ]
    return float(np.clip(sum(components), -1.0, 1.0))


def _build_branch_candidates(latest: Mapping[str, Any], branch_count: int = 64) -> pd.DataFrame:
    anchor_price = _safe_float(latest.get("close"), _safe_float(latest.get("anchor_price"), 0.0))
    atr_pct = max(abs(_safe_float(latest.get("atr_pct"), 0.001)), 1e-5)
    direction_signal = _direction_signal(latest)
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    for branch_id in range(1, int(branch_count) + 1):
        noise = rng.normal(0.0, atr_pct * 2.2)
        drift = direction_signal * atr_pct * 1.8
        predicted_price = anchor_price * (1.0 + drift + noise)
        rows.append(
            {
                "branch_id": branch_id,
                "branch_label": f"v20_branch_{branch_id}",
                "anchor_price": anchor_price,
                "predicted_price_15m": predicted_price,
                "generator_probability": float(np.clip(0.5 + drift * 20.0 - abs(noise) * 10.0, 0.01, 0.99)),
                "analog_similarity": float(np.clip(0.55 + direction_signal * 0.15 - abs(noise) * 20.0, 0.01, 0.99)),
                "quant_regime_strength": _safe_float(latest.get("quant_regime_strength"), 0.5),
                "quant_route_confidence": _safe_float(latest.get("quant_route_confidence"), 0.5),
                "branch_confidence": float(np.clip(0.55 - abs(noise) * 12.0 + branch_id / max(branch_count * 20.0, 1.0), 0.01, 0.99)),
                "consensus_score": float(np.clip(0.60 - abs(noise) * 10.0, 0.01, 0.99)),
                "mfg_disagreement": _safe_float(latest.get("mfg_disagreement"), 0.4),
                "macro_alignment": float(np.clip(0.5 + np.sign(direction_signal or 1.0) * np.sign(_safe_float(latest.get("macro_trend_strength"), 0.0)) * 0.25, 0.0, 1.0)),
                "cone_realism": _safe_float(latest.get("quant_vol_realism"), 0.5),
                "hmm_regime_match": 1.0 if int(_safe_float(latest.get("hmm_state"), 0.0)) in {1, 2, 3, 4} else 0.5,
            }
        )
    candidates = pd.DataFrame(rows)
    return heuristic_branch_scores(candidates)


@lru_cache(maxsize=1)
def _runtime_executor() -> HierarchicalExecutor:
    return HierarchicalExecutor()


def _load_conformal() -> ConformalCone:
    if V20_CONFORMAL_CONE_PATH.exists():
        try:
            return ConformalCone.load(V20_CONFORMAL_CONE_PATH)
        except Exception:
            pass
    return ConformalCone(alpha=0.15)


def _load_hmm_detector() -> RegimeDetector | None:
    if not V20_HMM_MODEL_PATH.exists():
        return None
    try:
        return RegimeDetector.load(V20_HMM_MODEL_PATH)
    except Exception:
        return None


@dataclass
class V20RuntimeBundle:
    feature_row: dict[str, Any]
    branches: pd.DataFrame
    consensus_path: list[float]
    minority_path: list[float]
    cone_upper: list[float]
    cone_lower: list[float]
    confidence: float
    regime_probs: list[float]


def build_v20_live_bundle(payload: Mapping[str, Any]) -> V20RuntimeBundle | None:
    price_frame = _price_frame_from_payload(payload)
    if price_frame.empty or len(price_frame) < 30:
        return None
    features, _ = build_v20_feature_frame(price_frame, hmm_detector=_load_hmm_detector())
    latest = features.iloc[-1].to_dict()
    latest["direction_signal"] = _direction_signal(latest)
    branches = _build_branch_candidates(latest)
    top = branches.sort_values("v20_cabr_score", ascending=False).reset_index(drop=True)
    consensus_price = float(top.head(5)["predicted_price_15m"].mean()) if not top.empty else _safe_float(latest.get("close"), 0.0)
    minority_price = float(top.tail(5)["predicted_price_15m"].mean()) if len(top) >= 5 else consensus_price
    current_price = _safe_float(latest.get("close"), 0.0)
    regime = int(_safe_float(latest.get("hmm_state"), 0.0))
    cone = _load_conformal()
    path = np.asarray([current_price, (current_price + consensus_price) / 2.0, consensus_price], dtype=np.float64)
    upper, lower, confidence = cone.predict(path, regime)
    regime_probs = [float(_safe_float(latest.get(f"hmm_prob_{idx}"), 0.0)) for idx in range(6)]
    return V20RuntimeBundle(
        feature_row=latest,
        branches=top,
        consensus_path=[round(float(item), 5) for item in path.tolist()],
        minority_path=[round(float(item), 5) for item in [current_price, (current_price + minority_price) / 2.0, minority_price]],
        cone_upper=[round(float(item), 5) for item in upper.tolist()],
        cone_lower=[round(float(item), 5) for item in lower.tolist()],
        confidence=float(confidence),
        regime_probs=regime_probs,
    )


def build_v20_local_judge(payload: Mapping[str, Any], bundle: V20RuntimeBundle | None) -> dict[str, Any]:
    if bundle is None:
        return {
            "available": False,
            "provider": "local_sjd_v20",
            "model": "v20_rule_judge",
            "error": "insufficient_live_bars",
            "content": {
                "final_call": "SKIP",
                "confidence": "LOW",
                "final_summary": "SKIP - V20 local judge needs more live bars.",
                "reasoning": "The V20 feature stack needs at least 30 recent bars before judging.",
                "entry_zone": [],
                "stop_loss": None,
                "take_profit": None,
                "hold_time": "skip",
                "market_only_summary": {"call": "SKIP", "summary": "Insufficient data.", "reasoning": "Need more recent bars."},
                "v18_summary": {"call": "SKIP", "summary": "Insufficient data.", "reasoning": "Need more recent bars."},
                "combined_summary": {"call": "SKIP", "summary": "Insufficient data.", "reasoning": "Need more recent bars."},
                "key_risk": "Not enough bars for the V20 feature stack.",
                "crowd_note": "Crowd context unavailable.",
                "regime_note": "Regime unavailable.",
                "invalidation": None,
            },
        }
    decision = latest_sjd_decision(bundle.feature_row)
    current_price = _safe_float(bundle.feature_row.get("close"), 0.0)
    stop_loss = current_price - (decision.sl_offset * 0.1) if decision.final_call == "BUY" else current_price + (decision.sl_offset * 0.1) if decision.final_call == "SELL" else None
    take_profit = current_price + (decision.tp_offset * 0.1) if decision.final_call == "BUY" else current_price - (decision.tp_offset * 0.1) if decision.final_call == "SELL" else None
    return {
        "available": True,
        "provider": "local_sjd_v20",
        "model": "v20_rule_judge",
        "content": {
            "stance": decision.final_call if decision.final_call in {"BUY", "SELL"} else "HOLD",
            "confidence": decision.confidence,
            "final_call": decision.final_call,
            "final_summary": f"{decision.final_call} - V20 local judge is using macro-conditioned rule distillation.",
            "entry_zone": [round(current_price - 0.15, 2), round(current_price + 0.15, 2)] if decision.final_call in {"BUY", "SELL"} else [],
            "stop_loss": round(float(stop_loss), 2) if stop_loss is not None else None,
            "take_profit": round(float(take_profit), 2) if take_profit is not None else None,
            "hold_time": "15m",
            "market_only_summary": {
                "call": decision.final_call,
                "summary": f"Market-only V20 read is {decision.final_call}.",
                "reasoning": decision.reasoning,
            },
            "v18_summary": {
                "call": decision.final_call,
                "summary": "V20 simulator is independent from Kimi.",
                "reasoning": f"Direction signal {bundle.feature_row.get('direction_signal', 0.0):.3f} and CABR {bundle.branches['v20_cabr_score'].iloc[0]:.1%} drive the local call.",
            },
            "combined_summary": {
                "call": decision.final_call,
                "summary": f"Combined V20 local read is {decision.final_call}.",
                "reasoning": f"Kelly {decision.kelly:.3f}, conformal confidence {bundle.confidence:.1%}, and HMM routing are all included.",
            },
            "reasoning": decision.reasoning,
            "key_risk": "Rule-distilled V20 local judge is a fallback until the large SJD V20 model is trained.",
            "crowd_note": f"MFG disagreement is {bundle.feature_row.get('mfg_disagreement', 0.0):.3f}.",
            "regime_note": f"HMM regime is {bundle.feature_row.get('hmm_state_name', 'unknown')}.",
            "invalidation": round(float(stop_loss), 2) if stop_loss is not None else None,
            "kelly_fraction": round(decision.kelly, 4),
        },
    }


def build_v20_runtime_state(payload: Mapping[str, Any], *, mode: str = "frequency") -> dict[str, Any]:
    bundle = build_v20_live_bundle(payload)
    if bundle is None or bundle.branches.empty:
        return {"available": False, "execution_reason": "insufficient_live_bars"}
    best = bundle.branches.iloc[0].to_dict()
    latest = bundle.feature_row
    best_direction = str(best.get("decision_direction", "HOLD")).upper()
    sjd = latest_sjd_decision(latest)
    decision_signal = _safe_float(latest.get("direction_signal"), 0.0)
    executor = _runtime_executor()
    macro_state = [
        _safe_float(latest.get("macro_trend_strength"), 0.0),
        _safe_float(latest.get("macro_dxy_zscore_20d"), 0.0),
        _safe_float(latest.get("macro_dxy_zscore_60d"), 0.0),
        _safe_float(latest.get("macro_realized_vol_20"), 0.0),
        _safe_float(latest.get("macro_realized_vol_60"), 0.0),
        _safe_float(latest.get("mfg_disagreement"), 0.0),
        _safe_float(latest.get("hurst_overall"), 0.5),
        _safe_float(latest.get("spectral_entropy"), 0.0),
    ]
    branch_rewards = [float((value - _safe_float(latest.get("close"), 0.0)) * (1.0 if best_direction == "BUY" else -1.0)) for value in bundle.branches["predicted_price_15m"].head(16).tolist()]
    exec_decision = executor.decide(
        regime_probs=bundle.regime_probs,
        direction_signal=decision_signal,
        confidence=float(best.get("v20_cabr_score", 0.5)),
        macro_state=macro_state,
        volatility=_safe_float(latest.get("macro_realized_vol_20"), 0.0),
        kelly_fraction=sjd.kelly,
        branch_rewards=branch_rewards,
    )
    cpm_score = float(
        np.clip(
            0.30 * _safe_float(latest.get("quant_route_confidence"), 0.5)
            + 0.20 * _safe_float(latest.get("quant_regime_strength"), 0.5)
            + 0.15 * (1.0 - min(1.0, abs(_safe_float(latest.get("mfg_disagreement"), 0.5))))
            + 0.15 * np.clip(abs(decision_signal), 0.0, 1.0)
            + 0.20 * _safe_float(best.get("generator_probability"), 0.5),
            0.0,
            1.0,
        )
    )
    cone_width = abs(bundle.cone_upper[-1] - bundle.cone_lower[-1]) / 0.1 if len(bundle.cone_upper) == len(bundle.cone_lower) and bundle.cone_upper else 0.0
    tier = classify_confidence(float(best.get("v20_cabr_score", 0.5)), max(bundle.confidence, 0.5), cone_width, cpm_score=cpm_score)
    alignment_ok = sjd.final_call == exec_decision.action
    should_execute = (
        exec_decision.action in {"BUY", "SELL"}
        and tier in {ConfidenceTier.VERY_HIGH, ConfidenceTier.HIGH, ConfidenceTier.MODERATE}
        and float(best.get("v20_cabr_score", 0.5)) >= 0.64
        and cpm_score >= 0.52
        and abs(decision_signal) >= 0.14
        and alignment_ok
    )
    lot = sel_lot_size(
        equity=1000.0,
        confidence_tier=tier,
        sqt_label="GOOD" if bundle.confidence >= 0.85 else "NEUTRAL",
        mode=mode,
        stop_pips=max(sjd.sl_offset, 12.0),
        pip_value_per_lot=10.0,
        max_lot=0.20,
    )
    final_action = exec_decision.action if should_execute else "HOLD"
    executor.record_outcome(regime_probs=bundle.regime_probs, action=final_action, reward=exec_decision.expected_reward)
    return {
        "available": True,
        "runtime_version": "v20",
        "selected_branch_id": int(best.get("branch_id", 0)),
        "selected_branch_label": str(best.get("branch_label", "v20_branch")),
        "decision_direction": final_action,
        "cabr_score": round(float(best.get("v20_cabr_score", 0.5)), 6),
        "cabr_raw_score": round(float(best.get("v20_cabr_raw_score", 0.5)), 6),
        "cpm_score": round(cpm_score, 6),
        "confidence_tier": tier.value,
        "sqt_label": "GOOD" if bundle.confidence >= 0.85 else "NEUTRAL",
        "cone_width_pips": round(float(cone_width), 3),
        "lepl_action": exec_decision.action,
        "lepl_probabilities": exec_decision.hyper_weights,
        "lepl_features": {
            "active_sub_agent": exec_decision.active_sub_agent,
            "kelly_fraction": round(float(sjd.kelly), 6),
            "direction_signal": round(float(decision_signal), 6),
            "conformal_confidence": round(float(bundle.confidence), 6),
            "mode_multiplier": round(_mode_multiplier(mode), 4),
            "suggested_lot": round(float(lot), 4),
        },
        "should_execute": bool(should_execute),
        "execution_reason": (
            f"V20 chose {exec_decision.active_sub_agent} with Kelly {sjd.kelly:.3f} and conformal confidence {bundle.confidence:.1%}."
            if should_execute
            else f"V20 held because tier={tier.value}, cabr={float(best.get('v20_cabr_score', 0.5)):.3f}, cpm={cpm_score:.3f}, signal={decision_signal:.3f}, and sjd_alignment={alignment_ok} did not clear the execution gate."
        ),
        "branch_scores": [
            {
                "branch_id": int(item.get("branch_id", 0)),
                "branch_label": str(item.get("branch_label", "")),
                "decision_direction": str(item.get("decision_direction", "HOLD")),
                "cabr_score": round(float(item.get("v20_cabr_score", 0.0)), 6),
                "cabr_raw_score": round(float(item.get("v20_cabr_raw_score", 0.0)), 6),
            }
            for item in bundle.branches.head(8).to_dict(orient="records")
        ],
        "consensus_path": bundle.consensus_path,
        "minority_path": bundle.minority_path,
        "cone_upper": bundle.cone_upper,
        "cone_lower": bundle.cone_lower,
        "regime_probs": [round(float(value), 6) for value in bundle.regime_probs],
    }
