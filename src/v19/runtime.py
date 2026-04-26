from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.project_config import V19_BRANCH_ARCHIVE_PATH, V19_CABR_MODEL_PATH, V19_LEPL_MODEL_PATH
from src.v13.cabr import load_cabr_model, score_cabr_model
from src.v16.confidence_tier import ConfidenceTier, classify_confidence
from src.v16.sel import sel_lot_size
from src.v19.lepl import LiveExecutionPolicy

_V19_CABR_CACHE: tuple[Any, tuple[str, ...], tuple[str, ...], dict[str, Any]] | None = None
_V19_LEPL_CACHE: LiveExecutionPolicy | None = None
_V19_LEPL_LOAD_ERROR: str | None = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _clip01(value: Any, default: float = 0.5) -> float:
    return float(np.clip(_safe_float(value, default), 0.0, 1.0))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value))))


def _binary_entropy(probability: float) -> float:
    p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0))


def _normalized_location_from_fair_value(fair_value_z: float) -> float:
    return float(np.clip(0.5 + (0.18 * _safe_float(fair_value_z, 0.0)), 0.0, 1.0))


def _contradiction_flags(raw: Any) -> dict[str, float]:
    label = str(raw or "").strip().lower()
    return {
        "contradiction_full_agreement": 1.0 if "full_agreement" in label or label.startswith("agreement_") else 0.0,
        "contradiction_partial_disagreement": 1.0 if "partial" in label else 0.0,
        "contradiction_full_disagreement": 1.0 if "full_disagreement" in label else 0.0,
        "contradiction_mixed": 1.0 if "mixed" in label or label == "balanced" else 0.0,
    }


def _regime_flags(raw: Any) -> dict[str, float]:
    label = str(raw or "").strip().lower()
    return {
        "context_regime_trending_up": 1.0 if label in {"trending_up", "trend_following", "bullish_trend"} else 0.0,
        "context_regime_trending_down": 1.0 if label in {"trending_down", "bearish_trend"} else 0.0,
        "context_regime_ranging": 1.0 if label in {"ranging", "balanced_range", "random_walk"} else 0.0,
        "context_regime_breakout": 1.0 if label in {"breakout", "trend_breakout"} else 0.0,
        "context_regime_panic_shock": 1.0 if label in {"panic_shock", "macro_shock"} else 0.0,
        "context_regime_low_volatility": 1.0 if label in {"low_volatility", "compression"} else 0.0,
    }


def _wltc_flags(retail_t: float, institutional_t: float, noise_t: float) -> dict[str, float]:
    values = {
        "retail": _safe_float(retail_t, 0.0),
        "institutional": _safe_float(institutional_t, 0.0),
        "noise": _safe_float(noise_t, 0.0),
    }
    top_name = max(values, key=values.get)
    ordered = sorted(values.values(), reverse=True)
    balanced = 1.0 if abs(ordered[0] - ordered[1]) < 0.08 else 0.0
    return {
        "wltc_state_retail_dominant": 1.0 if top_name == "retail" and balanced == 0.0 else 0.0,
        "wltc_state_institutional_dominant": 1.0 if top_name == "institutional" and balanced == 0.0 else 0.0,
        "wltc_state_balanced": balanced,
    }


def _confidence_scalar(raw: Any) -> float:
    return {
        "VERY_LOW": 0.0,
        "LOW": 1.0,
        "MODERATE": 2.0,
        "HIGH": 3.0,
    }.get(str(raw or "LOW").strip().upper(), 1.0)


def infer_sqt_label(cpm_score: float, cabr_score: float) -> str:
    signal = min(_clip01(cpm_score), _clip01(cabr_score))
    if signal < 0.42:
        return "COLD"
    if signal < 0.56:
        return "NEUTRAL"
    if signal < 0.72:
        return "GOOD"
    return "HOT"


def mode_allows_trade(mode: str, tier: ConfidenceTier) -> bool:
    normalized = str(mode).strip().lower()
    if normalized == "precision":
        return tier in {ConfidenceTier.HIGH, ConfidenceTier.VERY_HIGH}
    return tier in {ConfidenceTier.LOW, ConfidenceTier.MODERATE, ConfidenceTier.HIGH, ConfidenceTier.VERY_HIGH}


def load_v19_cabr_runtime() -> tuple[Any, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    global _V19_CABR_CACHE
    if _V19_CABR_CACHE is None:
        _V19_CABR_CACHE = load_cabr_model(V19_CABR_MODEL_PATH, map_location="cpu")
    return _V19_CABR_CACHE


def load_v19_lepl_policy() -> LiveExecutionPolicy | None:
    global _V19_LEPL_CACHE, _V19_LEPL_LOAD_ERROR
    if _V19_LEPL_CACHE is not None:
        return _V19_LEPL_CACHE
    if _V19_LEPL_LOAD_ERROR is not None:
        return None
    if not V19_LEPL_MODEL_PATH.exists():
        _V19_LEPL_LOAD_ERROR = "missing_v19_lepl_checkpoint"
        return None
    try:
        _V19_LEPL_CACHE = LiveExecutionPolicy.load(V19_LEPL_MODEL_PATH)
        return _V19_LEPL_CACHE
    except Exception as exc:
        _V19_LEPL_LOAD_ERROR = str(exc)
        return None


def _live_candidate_row(
    payload: Mapping[str, Any],
    branch: Mapping[str, Any],
) -> dict[str, Any]:
    simulation = dict(payload.get("simulation", {}) if isinstance(payload, Mapping) else {})
    technical = dict(payload.get("technical_analysis", {}) if isinstance(payload, Mapping) else {})
    current_row = dict(payload.get("current_row", {}) if isinstance(payload, Mapping) else {})
    mfg = dict(payload.get("mfg", {}) if isinstance(payload, Mapping) else {})
    contradiction = str(simulation.get("contradiction_type", payload.get("contradiction", {}).get("type", "mixed")))
    symbol = str(payload.get("symbol", "XAUUSD")).upper()
    current_price = _safe_float(payload.get("market", {}).get("current_price"), 0.0)
    predicted_prices = list(branch.get("predicted_prices", []))
    predicted_terminal = _safe_float(predicted_prices[-1] if predicted_prices else current_price, current_price)
    branch_probability = _clip01(branch.get("probability"), _clip01(branch.get("selector_score"), 0.5))
    branch_direction = 1.0 if predicted_terminal >= current_price else -1.0
    retail_t = _safe_float((payload.get("wltc", {}) or {}).get("retail", {}).get("testosterone_index"), _safe_float(current_row.get("wltc_testosterone_retail"), 0.0))
    institutional_t = _safe_float((payload.get("wltc", {}) or {}).get("institutional", {}).get("testosterone_index"), 0.0)
    noise_t = _safe_float((payload.get("wltc", {}) or {}).get("noise", {}).get("testosterone_index"), _safe_float(current_row.get("wltc_testosterone_noise"), 0.0))
    fair_value_z = _safe_float(technical.get("quant_fair_value_z"), 0.0)
    volatility_realism = _clip01(technical.get("quant_vol_realism"), 0.5)
    cpm_score = _clip01(simulation.get("cpm_score"), 0.5)
    route_confidence = _clip01(current_row.get("quant_route_confidence"), 0.5)
    path_error = abs(predicted_terminal - current_price) / max(current_price, 1e-6)

    row: dict[str, Any] = {
        "symbol": symbol,
        "sample_id": int(branch.get("path_id", 0) or 0),
        "branch_id": int(branch.get("path_id", 0) or 0),
        "timestamp": payload.get("generated_at"),
        "anchor_price": current_price,
        "entry_open_price": current_price,
        "actual_price_15m": predicted_terminal,
        "exit_close_price_15m": predicted_terminal,
        "predicted_price_15m": predicted_terminal,
        "generator_probability": branch_probability,
        "branch_confidence": branch_probability,
        "branch_entropy": _binary_entropy(branch_probability),
        "path_entropy": _binary_entropy(branch_probability),
        "path_smoothness": float(np.clip(1.0 - path_error, 0.0, 1.0)),
        "reversal_likelihood": _clip01(technical.get("quant_transition_risk"), 0.5),
        "mean_reversion_likelihood": float(np.clip(abs(fair_value_z) / 3.0, 0.0, 1.0)),
        "v10_diversity_score": _clip01(mfg.get("disagreement"), 0.5),
        "analog_similarity": _clip01(simulation.get("analog_confidence"), 0.5),
        "leaf_analog_confidence": _clip01(simulation.get("analog_confidence"), 0.5),
        "consensus_strength": _clip01(simulation.get("consensus_score"), 0.5),
        "analog_confidence": _clip01(simulation.get("analog_confidence"), 0.5),
        "cone_realism": volatility_realism,
        "mfg_disagreement": _clip01(mfg.get("disagreement"), 0.0),
        "volatility_realism": volatility_realism,
        "fair_value_dislocation": abs(fair_value_z),
        "context_regime_confidence": _clip01(simulation.get("regime_confidence"), _clip01(technical.get("quant_regime_strength"), 0.5)),
        "context_atr_percentile_30d": float(np.clip(_safe_float(current_row.get("atr_14"), 0.0) / max(current_price * 0.003, 1e-6), 0.0, 1.0)),
        "context_rsi_14": _safe_float(technical.get("rsi_14"), 50.0),
        "context_macd_hist": _safe_float(current_row.get("ema_cross"), 0.0),
        "context_bb_pct": _normalized_location_from_fair_value(fair_value_z),
        "context_days_since_regime_change": round(30.0 * _clip01(route_confidence, 0.5), 4),
        "context_emotional_momentum": float(np.clip(abs(_safe_float(simulation.get("crowd_bias"), 0.0)) * _clip01(simulation.get("crowd_extreme"), 0.0), 0.0, 1.0)),
        "context_emotional_fragility": float(np.clip(_clip01(simulation.get("crowd_extreme"), 0.0) * (1.0 - _clip01(simulation.get("overall_confidence"), 0.5)), 0.0, 1.0)),
        "context_emotional_conviction": float(np.clip(abs(_safe_float(simulation.get("crowd_bias"), 0.0)), 0.0, 1.0)),
        "context_narrative_age": float(np.clip(1.0 - _clip01(payload.get("eci", {}).get("cone_width_modifier"), 0.0), 0.0, 1.0)),
        "context_hurst_overall": _safe_float(simulation.get("hurst_overall"), 0.5),
        "context_hurst_positive": _safe_float(simulation.get("hurst_positive"), 0.5),
        "context_hurst_negative": _safe_float(simulation.get("hurst_negative"), 0.5),
        "context_hurst_asymmetry": _safe_float(simulation.get("hurst_asymmetry"), 0.0),
        "quant_regime_strength": _clip01(technical.get("quant_regime_strength"), 0.5),
        "quant_transition_risk": _clip01(technical.get("quant_transition_risk"), 0.5),
        "quant_vol_realism": volatility_realism,
        "quant_fair_value_z": fair_value_z,
        "quant_route_confidence": route_confidence,
        "quant_trend_score": float(np.clip(_safe_float(simulation.get("mean_probability"), 0.5) - 0.5, -1.0, 1.0)),
        "hurst_overall": _safe_float(simulation.get("hurst_overall"), 0.5),
        "hurst_positive": _safe_float(simulation.get("hurst_positive"), 0.5),
        "hurst_negative": _safe_float(simulation.get("hurst_negative"), 0.5),
        "hurst_asymmetry": _safe_float(simulation.get("hurst_asymmetry"), 0.0),
        "cpm_score": cpm_score,
        "cone_width_pips": _safe_float(simulation.get("cone_width_pips"), 0.0),
        "branch_label": str(branch.get("branch_label", "live_branch")),
        "decision_direction": "BUY" if branch_direction > 0 else "SELL",
        "dominant_regime": str(simulation.get("detected_regime", "ranging")),
        "mfg_consensus_drift": _safe_float(mfg.get("consensus_drift"), 0.0),
    }
    row.update(_contradiction_flags(contradiction))
    row.update(_regime_flags(row["dominant_regime"]))
    row.update(_wltc_flags(retail_t, institutional_t, noise_t))
    return row


def build_live_v19_candidate_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    branches = list(payload.get("branches", []) if isinstance(payload, Mapping) else [])
    if not branches:
        simulation = dict(payload.get("simulation", {}) if isinstance(payload, Mapping) else {})
        fallback_branch = {
            "path_id": 1,
            "probability": simulation.get("mean_probability", 0.5),
            "selector_score": simulation.get("overall_confidence", 0.5),
            "predicted_prices": list(simulation.get("consensus_path", []))[1:] or [payload.get("market", {}).get("current_price", 0.0)],
            "branch_label": "consensus_path",
        }
        branches = [fallback_branch]
    rows = [_live_candidate_row(payload, branch) for branch in branches if isinstance(branch, Mapping)]
    return pd.DataFrame(rows)


def _archive_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    contradiction_flags = _contradiction_flags(row.get("contradiction_type"))
    regime_flags = _regime_flags(row.get("dominant_regime", row.get("v10_regime_label", "ranging")))
    retail_impact = _safe_float(row.get("retail_impact"), 0.0)
    institutional_impact = _safe_float(row.get("institutional_impact"), 0.0)
    algo_impact = _safe_float(row.get("algo_impact"), 0.0)
    wltc_flags = _wltc_flags(abs(retail_impact), abs(institutional_impact), abs(algo_impact))
    fair_value_z = _safe_float(row.get("quant_fair_value_z"), _safe_float(row.get("fair_value_dislocation"), 0.0))
    cpm_score = float(
        np.clip(
            (0.35 * _clip01(row.get("model_confidence_prob_15m"), 0.5))
            + (0.25 * _clip01(row.get("generator_probability"), 0.5))
            + (0.20 * _clip01(row.get("branch_confidence"), 0.5))
            + (0.20 * (1.0 - np.clip(abs(_safe_float(row.get("path_error"), 0.0)), 0.0, 1.0))),
            0.0,
            1.0,
        )
    )
    payload = {
        "symbol": str(row.get("symbol", "XAUUSD")).upper(),
        "sample_id": int(_safe_float(row.get("sample_id"), 0)),
        "branch_id": int(_safe_float(row.get("branch_id"), 0)),
        "timestamp": row.get("timestamp"),
        "anchor_price": _safe_float(row.get("anchor_price"), _safe_float(row.get("entry_open_price"), 0.0)),
        "entry_open_price": _safe_float(row.get("entry_open_price"), _safe_float(row.get("anchor_price"), 0.0)),
        "actual_price_15m": _safe_float(row.get("actual_price_15m"), 0.0),
        "exit_close_price_15m": _safe_float(row.get("exit_close_price_15m"), _safe_float(row.get("actual_price_15m"), 0.0)),
        "predicted_price_15m": _safe_float(row.get("predicted_price_15m"), _safe_float(row.get("anchor_price"), 0.0)),
        "generator_probability": _clip01(row.get("generator_probability"), 0.5),
        "branch_confidence": _clip01(row.get("branch_confidence"), 0.5),
        "branch_entropy": _clip01(row.get("branch_entropy"), 0.5),
        "path_entropy": _clip01(row.get("path_entropy"), _clip01(row.get("branch_disagreement"), 0.5)),
        "path_smoothness": _clip01(row.get("path_smoothness"), 1.0 - np.clip(abs(_safe_float(row.get("path_error"), 0.0)), 0.0, 1.0)),
        "reversal_likelihood": _clip01(row.get("reversal_likelihood"), row.get("hmm_transition_risk")),
        "mean_reversion_likelihood": _clip01(row.get("mean_reversion_likelihood"), row.get("mean_reversion_pressure")),
        "v10_diversity_score": _clip01(row.get("v10_diversity_score"), 0.5),
        "analog_similarity": _clip01(row.get("analog_similarity"), 0.5),
        "leaf_analog_confidence": _clip01(row.get("leaf_analog_confidence"), row.get("analog_confidence")),
        "consensus_strength": _clip01(row.get("consensus_strength"), 0.5),
        "analog_confidence": _clip01(row.get("analog_confidence"), 0.5),
        "cone_realism": _clip01(row.get("cone_realism"), row.get("quant_vol_realism")),
        "mfg_disagreement": _clip01(row.get("mfg_disagreement"), 0.0),
        "volatility_realism": _clip01(row.get("volatility_realism"), row.get("quant_vol_realism")),
        "fair_value_dislocation": abs(_safe_float(row.get("fair_value_dislocation"), fair_value_z)),
        "context_regime_confidence": _clip01(row.get("context_regime_confidence"), row.get("quant_regime_strength", row.get("hmm_regime_match", 0.5))),
        "context_atr_percentile_30d": _clip01(row.get("context_atr_percentile_30d"), row.get("branch_volatility")),
        "context_rsi_14": _safe_float(row.get("context_rsi_14"), 50.0 + 25.0 * _safe_float(row.get("quant_trend_score"), 0.0)),
        "context_macd_hist": _safe_float(row.get("context_macd_hist"), row.get("quant_trend_score", 0.0)),
        "context_bb_pct": _clip01(row.get("context_bb_pct"), _normalized_location_from_fair_value(fair_value_z)),
        "context_days_since_regime_change": _safe_float(row.get("context_days_since_regime_change"), 30.0 * _clip01(row.get("hmm_persistence"), 0.5)),
        "context_emotional_momentum": _clip01(row.get("context_emotional_momentum"), abs(_safe_float(row.get("crowd_bias"), 0.0)) * _clip01(row.get("crowd_extreme"), 0.0)),
        "context_emotional_fragility": _clip01(row.get("context_emotional_fragility"), _clip01(row.get("crowd_extreme"), 0.0) * (1.0 - _clip01(row.get("crowd_consistency"), 0.5))),
        "context_emotional_conviction": _clip01(row.get("context_emotional_conviction"), abs(_safe_float(row.get("crowd_bias"), 0.0)) * _clip01(row.get("crowd_consistency"), 0.5)),
        "context_narrative_age": _clip01(row.get("context_narrative_age"), 1.0 - _clip01(row.get("news_intensity"), 0.0)),
        "context_hurst_overall": _safe_float(row.get("context_hurst_overall"), row.get("hurst_overall", 0.5)),
        "context_hurst_positive": _safe_float(row.get("context_hurst_positive"), row.get("hurst_positive", 0.5)),
        "context_hurst_negative": _safe_float(row.get("context_hurst_negative"), row.get("hurst_negative", 0.5)),
        "context_hurst_asymmetry": _safe_float(row.get("context_hurst_asymmetry"), row.get("hurst_asymmetry", 0.0)),
        "quant_regime_strength": _clip01(row.get("quant_regime_strength"), 0.5),
        "quant_transition_risk": _clip01(row.get("quant_transition_risk"), 0.5),
        "quant_vol_realism": _clip01(row.get("quant_vol_realism"), row.get("cone_realism")),
        "quant_fair_value_z": fair_value_z,
        "quant_route_confidence": _clip01(row.get("quant_route_confidence"), row.get("generator_probability")),
        "quant_trend_score": float(np.clip(_safe_float(row.get("quant_trend_score"), 0.0), -1.0, 1.0)),
        "hurst_overall": _safe_float(row.get("hurst_overall"), 0.5),
        "hurst_positive": _safe_float(row.get("hurst_positive"), 0.5),
        "hurst_negative": _safe_float(row.get("hurst_negative"), 0.5),
        "hurst_asymmetry": _safe_float(row.get("hurst_asymmetry"), 0.0),
        "cpm_score": cpm_score,
        "cone_width_pips": abs(_safe_float(row.get("predicted_price_15m"), 0.0) - _safe_float(row.get("anchor_price"), 0.0)) / 0.1,
        "branch_label": str(row.get("branch_label", "archive_branch")),
        "decision_direction": "BUY" if _safe_float(row.get("predicted_price_15m"), 0.0) >= _safe_float(row.get("anchor_price"), 0.0) else "SELL",
        "dominant_regime": str(row.get("dominant_regime", row.get("v10_regime_label", "ranging"))),
        "mfg_consensus_drift": _safe_float(row.get("mfg_consensus_drift"), 0.0),
    }
    payload.update(contradiction_flags)
    payload.update(regime_flags)
    payload.update(wltc_flags)
    return payload


def build_archive_v19_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [_archive_candidate_row(row) for row in frame.to_dict(orient="records")]
    return pd.DataFrame(rows)


def score_v19_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    model, branch_cols, context_cols, _ = load_v19_cabr_runtime()
    working = frame.copy()
    for column in branch_cols:
        if column not in working.columns:
            working[column] = 0.0
    for column in context_cols:
        if column not in working.columns:
            working[column] = 0.5 if "hurst" in column else 0.0
    raw_scores = score_cabr_model(
        model,
        working,
        branch_feature_names=branch_cols,
        context_feature_names=context_cols,
        device="cpu",
    )
    working["v19_cabr_raw_score"] = raw_scores
    working["v19_cabr_score"] = [round(_sigmoid(value), 6) for value in raw_scores]
    working["bst_proxy"] = (
        0.35 * working["consensus_strength"].astype(float).to_numpy()
        + 0.25 * working["analog_confidence"].astype(float).to_numpy()
        + 0.20 * working["cone_realism"].astype(float).to_numpy()
        + 0.20 * (1.0 - working["mfg_disagreement"].astype(float).to_numpy())
    )
    working["confidence_tier"] = [
        classify_confidence(
            float(working["v19_cabr_score"].iloc[idx]),
            float(np.clip(working["bst_proxy"].iloc[idx], 0.0, 1.0)),
            float(working["cone_width_pips"].iloc[idx]),
            float(np.clip(working["cpm_score"].iloc[idx], 0.0, 1.0)),
        ).value
        for idx in range(len(working))
    ]
    working["sqt_label"] = [
        infer_sqt_label(float(working["cpm_score"].iloc[idx]), float(working["v19_cabr_score"].iloc[idx]))
        for idx in range(len(working))
    ]
    return working


def build_sjd_context_from_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    direction = str(row.get("decision_direction", "HOLD")).upper()
    current_price = _safe_float(row.get("anchor_price"), 0.0)
    predicted_terminal = _safe_float(row.get("predicted_price_15m"), current_price)
    fair_value_z = _safe_float(row.get("quant_fair_value_z"), 0.0)
    structure = "bullish" if direction == "BUY" else "bearish" if direction == "SELL" else "balanced"
    location = "premium" if fair_value_z > 0.15 else "discount" if fair_value_z < -0.15 else "equilibrium"
    return {
        "market": {
            "current_price": round(current_price, 5),
            "recent_close_change": round(predicted_terminal - current_price, 5),
        },
        "simulation": {
            "direction": direction,
            "scenario_bias": direction.lower(),
            "overall_confidence": round(_safe_float(row.get("v19_cabr_score"), 0.0), 6),
            "cabr_score": round(_safe_float(row.get("v19_cabr_score"), 0.0), 6),
            "cpm_score": round(_safe_float(row.get("cpm_score"), 0.0), 6),
            "cone_width_pips": round(_safe_float(row.get("cone_width_pips"), 0.0), 3),
            "detected_regime": str(row.get("dominant_regime", "ranging")),
            "hurst_overall": round(_safe_float(row.get("hurst_overall"), 0.5), 6),
            "hurst_positive": round(_safe_float(row.get("hurst_positive"), 0.5), 6),
            "hurst_negative": round(_safe_float(row.get("hurst_negative"), 0.5), 6),
            "hurst_asymmetry": round(_safe_float(row.get("hurst_asymmetry"), 0.0), 6),
            "entry_zone": [
                round(current_price - 0.2, 5),
                round(current_price + 0.2, 5),
            ],
        },
        "technical_analysis": {
            "structure": structure,
            "location": location,
            "rsi_14": round(_safe_float(row.get("context_rsi_14"), 50.0), 4),
            "equilibrium": round(current_price - (fair_value_z * 0.1), 5),
        },
        "bot_swarm": {
            "aggregate": {
                "signal": "bullish" if direction == "BUY" else "bearish" if direction == "SELL" else "neutral",
            }
        },
        "sqt": {
            "label": str(row.get("sqt_label", infer_sqt_label(_safe_float(row.get("cpm_score"), 0.0), _safe_float(row.get("v19_cabr_score"), 0.0)))).upper(),
        },
        "mfg": {
            "disagreement": round(_safe_float(row.get("mfg_disagreement"), 0.0), 8),
            "consensus_drift": round(_safe_float(row.get("mfg_consensus_drift"), 0.0), 8),
        },
    }


def build_lepl_features(
    *,
    local_judge_content: Mapping[str, Any] | None,
    row: Mapping[str, Any],
    has_open_position: bool,
    open_position_pnl: float,
) -> dict[str, Any]:
    content = dict(local_judge_content or {})
    return {
        "sjd_stance": str(content.get("final_call", content.get("stance", "HOLD"))).upper(),
        "sjd_confidence": str(content.get("confidence", "LOW")).upper(),
        "sqt_label": str(row.get("sqt_label", "NEUTRAL")).upper(),
        "cabr_score": _safe_float(row.get("v19_cabr_score"), 0.5),
        "hurst_asymmetry": _safe_float(row.get("hurst_asymmetry"), 0.0),
        "mfg_disagreement": _safe_float(row.get("mfg_disagreement"), 0.0),
        "cpm_score": _safe_float(row.get("cpm_score"), 0.5),
        "has_open_position": bool(has_open_position),
        "open_position_pnl": _safe_float(open_position_pnl, 0.0),
    }


def predict_lepl_action(
    *,
    local_judge_content: Mapping[str, Any] | None,
    row: Mapping[str, Any],
    has_open_position: bool = False,
    open_position_pnl: float = 0.0,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    features = build_lepl_features(
        local_judge_content=local_judge_content,
        row=row,
        has_open_position=has_open_position,
        open_position_pnl=open_position_pnl,
    )
    policy = load_v19_lepl_policy()
    if policy is None:
        actionable = str((local_judge_content or {}).get("final_call", "SKIP")).upper() in {"BUY", "SELL"}
        fallback_action = "ENTER" if actionable and not has_open_position else "NOTHING"
        fallback_probs = {"ENTER": 1.0 if fallback_action == "ENTER" else 0.0, "HOLD": 0.0, "CLOSE": 0.0, "NOTHING": 1.0 if fallback_action == "NOTHING" else 0.0}
        return fallback_action, fallback_probs, features
    action = policy.predict(features)
    probs = policy.predict_proba(features)
    return action, probs, features


def build_v19_runtime_state(
    payload: Mapping[str, Any],
    *,
    local_judge: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frame = build_live_v19_candidate_frame(payload)
    if frame.empty:
        return {"available": False, "reason": "no_live_branches"}
    scored = score_v19_candidates(frame).sort_values(["v19_cabr_raw_score", "branch_id"], ascending=[False, True]).reset_index(drop=True)
    best = scored.iloc[0].to_dict()
    local_content = dict(((local_judge or {}).get("content") or {}) if isinstance(local_judge, Mapping) else {})
    action, probabilities, features = predict_lepl_action(local_judge_content=local_content, row=best, has_open_position=bool((payload.get("paper_trading", {}) or {}).get("open_positions")), open_position_pnl=_safe_float((payload.get("paper_trading", {}) or {}).get("summary", {}).get("unrealized_pnl"), 0.0))
    tier = ConfidenceTier(str(best.get("confidence_tier", "low")))
    local_call = str(local_content.get("final_call", "SKIP")).upper()
    runtime_call = local_call if local_call in {"BUY", "SELL"} else str(best.get("decision_direction", "HOLD")).upper()
    should_execute = bool(runtime_call in {"BUY", "SELL"} and action == "ENTER" and mode_allows_trade(str(payload.get("mode", "frequency")), tier))
    execution_reason = (
        f"LEPL={action}, runtime_call={runtime_call}, local_call={local_call}, tier={tier.value}, sqt={best.get('sqt_label', 'NEUTRAL')}"
        if runtime_call in {"BUY", "SELL"}
        else f"Both the local V19 judge and the runtime branch selector are observational on this bar."
    )
    return {
        "available": True,
        "selected_branch_id": int(_safe_float(best.get("branch_id"), 0)),
        "selected_branch_label": str(best.get("branch_label", "unknown")),
        "decision_direction": str(best.get("decision_direction", "HOLD")),
        "runtime_call": runtime_call,
        "local_call": local_call,
        "local_agrees_with_runtime": bool(local_call in {"BUY", "SELL"} and local_call == runtime_call),
        "cabr_score": round(_safe_float(best.get("v19_cabr_score"), 0.0), 6),
        "cabr_raw_score": round(_safe_float(best.get("v19_cabr_raw_score"), 0.0), 6),
        "cpm_score": round(_safe_float(best.get("cpm_score"), 0.0), 6),
        "confidence_tier": str(best.get("confidence_tier", "low")),
        "sqt_label": str(best.get("sqt_label", "NEUTRAL")),
        "cone_width_pips": round(_safe_float(best.get("cone_width_pips"), 0.0), 3),
        "lepl_action": action,
        "lepl_probabilities": {str(key): round(_safe_float(value), 6) for key, value in probabilities.items()},
        "lepl_features": {key: (round(_safe_float(value), 6) if isinstance(value, (int, float)) else value) for key, value in features.items()},
        "should_execute": should_execute,
        "execution_reason": execution_reason,
        "branch_scores": [
            {
                "branch_id": int(_safe_float(row.get("branch_id"), 0)),
                "branch_label": str(row.get("branch_label", "unknown")),
                "decision_direction": str(row.get("decision_direction", "HOLD")),
                "cabr_score": round(_safe_float(row.get("v19_cabr_score"), 0.0), 6),
                "cabr_raw_score": round(_safe_float(row.get("v19_cabr_raw_score"), 0.0), 6),
            }
            for row in scored.to_dict(orient="records")
        ],
    }


def load_v19_branch_archive(month: str | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(V19_BRANCH_ARCHIVE_PATH)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.loc[frame["timestamp"].notna()].copy()
        if month:
            start = pd.Timestamp(f"{month}-01 00:00:00+00:00")
            end = start + pd.offsets.MonthBegin(1)
            frame = frame.loc[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].copy()
    return frame.reset_index(drop=True)


def suggested_lot_for_trade(
    *,
    equity: float,
    tier: ConfidenceTier,
    sqt_label: str,
    mode: str,
    stop_pips: float,
    pip_value_per_lot: float,
) -> float:
    return float(
        sel_lot_size(
            equity=equity,
            confidence_tier=tier,
            sqt_label=sqt_label,
            mode=mode,
            stop_pips=max(_safe_float(stop_pips, 20.0), 1.0),
            pip_value_per_lot=max(_safe_float(pip_value_per_lot, 10.0), 1e-6),
            min_lot=0.05,
            max_lot=2.0,
        )
    )
