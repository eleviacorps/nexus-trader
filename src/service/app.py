from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from config.project_config import (
    FEATURE_DIM_TOTAL,
    FINAL_DASHBOARD_HTML_PATH,
    FUTURE_BRANCHES_PATH,
    LEGACY_TFT_CHECKPOINT_PATH,
    LATEST_MARKET_SNAPSHOT_PATH,
    MODEL_MANIFEST_PATH,
    MODEL_SERVICE_HOST,
    MODEL_SERVICE_PORT,
    PERSONA_BREAKDOWN_HTML_PATH,
    PROBABILITY_CONE_HTML_PATH,
    SEQUENCE_LEN,
    TFT_CHECKPOINT_PATH,
)
from src.models.nexus_tft import NexusTFT, NexusTFTConfig, load_checkpoint_with_expansion
from src.service.llm_sidecar import (
    is_nvidia_nim_provider,
    read_packet_log,
    request_kimi_judge,
    request_local_sjd_judge,
    request_market_context,
    sidecar_health,
)
from src.service.live_data import (
    build_fast_dashboard_payload,
    build_live_monitor,
    build_live_simulation,
    build_realtime_chart_payload,
    fetch_live_quote,
    record_simulation_history,
)
from src.ui.web import render_web_app_html
from src.utils.device import get_torch_device
from src.v13.s3pta import PaperTradeAccumulator
from src.v16.csl import build_v16_simulation_result
from src.v16.paper import PaperTradingEngine
from src.v16.sqt import SimulationQualityTracker
from src.v18.websocket_feed import LiveFeedManager
from src.v19.runtime import build_v19_runtime_state

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    from fastapi import WebSocket, WebSocketDisconnect  # type: ignore
    from fastapi.responses import HTMLResponse  # type: ignore
    from fastapi.staticfiles import StaticFiles  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
except ImportError:  # pragma: no cover
    FastAPI = None
    HTTPException = RuntimeError  # type: ignore
    WebSocket = Any  # type: ignore
    WebSocketDisconnect = RuntimeError  # type: ignore
    BaseModel = object  # type: ignore
    HTMLResponse = str  # type: ignore
    StaticFiles = Any  # type: ignore

    def Field(default: Any, **_: Any):  # type: ignore
        return default


class PredictRequest(BaseModel):  # type: ignore[misc]
    sequence: list[list[float]] = Field(..., description='Sequence of shape [sequence_len, feature_dim]')


class PredictResponse(BaseModel):  # type: ignore[misc]
    bullish_probability: float
    bearish_probability: float
    signal: str
    threshold: float
    sequence_len: int
    feature_dim: int
    horizon_probabilities: dict[str, float] | None = None


class PaperOpenRequest(BaseModel):  # type: ignore[misc]
    symbol: str = "XAUUSD"
    direction: str
    entry_price: float
    confidence_tier: str
    sqt_label: str = "NEUTRAL"
    mode: str = "frequency"
    leverage: float = 200.0
    stop_pips: float = 20.0
    take_profit_pips: float = 30.0
    stop_loss: float | None = None
    take_profit: float | None = None
    manual_lot: float | None = None
    note: str = ""


class PaperCloseRequest(BaseModel):  # type: ignore[misc]
    trade_id: str
    exit_price: float


class PaperResetRequest(BaseModel):  # type: ignore[misc]
    starting_balance: float = 1000.0


class PaperModifyRequest(BaseModel):  # type: ignore[misc]
    trade_id: str
    stop_loss: float | None = None
    take_profit: float | None = None


FRONTEND_DIST_PATH = Path(__file__).resolve().parents[2] / "ui" / "frontend" / "dist"


def read_system_telemetry() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "gpu_available": False,
        "gpu_name": "GPU unavailable",
        "gpu_utilization_pct": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_temperature_c": None,
        "broker_connection": "Local paper broker ready",
        "local_runtime": str(get_torch_device()),
    }
    query = shutil.which("nvidia-smi")
    if query:
        try:
            result = subprocess.run(
                [
                    query,
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=5,
                check=False,
            )
            line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
            if line:
                name, util, mem_used, mem_total, temperature = [part.strip() for part in line.split(",")[:5]]
                payload.update(
                    {
                        "gpu_available": True,
                        "gpu_name": name or payload["gpu_name"],
                        "gpu_utilization_pct": float(util),
                        "gpu_memory_used_mb": float(mem_used),
                        "gpu_memory_total_mb": float(mem_total),
                        "gpu_temperature_c": float(temperature),
                    }
                )
                return payload
        except Exception:
            pass
    if torch is not None:
        try:
            if bool(torch.cuda.is_available()):
                payload["gpu_available"] = True
                payload["gpu_name"] = str(torch.cuda.get_device_name(0))
        except Exception:
            pass
    return payload


def llm_numeric_prior(payload: dict[str, Any]) -> float:
    llm_context = payload.get('llm_context', {})
    if not isinstance(llm_context, dict):
        return 0.5
    content = llm_context.get('content', {})
    if not isinstance(content, dict):
        return 0.5
    institutional = float(content.get('institutional_bias', 0.0) or 0.0)
    whale = float(content.get('whale_bias', 0.0) or 0.0)
    retail = float(content.get('retail_bias', 0.0) or 0.0)
    weighted = (0.40 * institutional) + (0.35 * whale) + (0.25 * retail)
    return max(0.0, min(1.0, 0.5 + 0.5 * weighted))


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in weights.values()) or 1.0
    return {key: round(max(0.0, float(value)) / total, 6) for key, value in weights.items()}


def _infer_regime_weights(payload: dict[str, Any], model_prediction: dict[str, Any] | None) -> tuple[dict[str, float], dict[str, float]]:
    current_row = payload.get("current_row", {}) if isinstance(payload, dict) else {}
    aggregate = payload.get("bot_swarm", {}).get("aggregate", {}) if isinstance(payload, dict) else {}
    model_diag = (model_prediction or {}).get("model_diagnostics", {}) if isinstance(model_prediction, dict) else {}
    route_confidence = float(current_row.get("quant_route_confidence", 0.0) or 0.0)
    transition_risk = float(current_row.get("quant_transition_risk", 0.0) or 0.0)
    macro_shock = abs(float(current_row.get("macro_shock", 0.0) or 0.0))
    route_bias = float(current_row.get("quant_route_prob_up", 0.5) or 0.5) - float(current_row.get("quant_route_prob_down", 0.5) or 0.5)
    regime_affinity = aggregate.get("regime_affinity", {}) if isinstance(aggregate, dict) else {}
    model_regime = model_diag.get("regime_probabilities", {}) if isinstance(model_diag, dict) else {}

    trend_score = 0.35 * route_confidence + 0.30 * abs(route_bias) + 0.20 * float(regime_affinity.get("trend", 0.0) or 0.0) + 0.15 * float(model_regime.get("trend", 0.0) or 0.0)
    reversal_score = 0.45 * float(regime_affinity.get("reversal", 0.0) or 0.0) + 0.20 * transition_risk + 0.20 * float(model_regime.get("reversal", 0.0) or 0.0) + 0.15 * float(regime_affinity.get("balanced", 0.0) or 0.0)
    macro_score = 0.40 * macro_shock + 0.25 * float(regime_affinity.get("macro_shock", 0.0) or 0.0) + 0.20 * float(model_regime.get("macro_shock", 0.0) or 0.0) + 0.15 * abs(float(current_row.get("macro_bias", 0.0) or 0.0))
    balanced_score = 0.40 * float(regime_affinity.get("balanced", 0.0) or 0.0) + 0.30 * max(0.0, 1.0 - abs(route_bias)) + 0.30 * float(model_regime.get("balanced", 0.0) or 0.0)

    regime_scores = _normalize_weights(
        {
            "trend": trend_score,
            "reversal": reversal_score,
            "macro_shock": macro_score,
            "balanced": balanced_score,
        }
    )

    component_templates = {
        "trend": {"branch": 0.26, "analog": 0.11, "model": 0.30, "bot_swarm": 0.25, "llm": 0.08},
        "reversal": {"branch": 0.23, "analog": 0.18, "model": 0.20, "bot_swarm": 0.31, "llm": 0.08},
        "macro_shock": {"branch": 0.22, "analog": 0.10, "model": 0.19, "bot_swarm": 0.17, "llm": 0.32},
        "balanced": {"branch": 0.20, "analog": 0.20, "model": 0.22, "bot_swarm": 0.24, "llm": 0.14},
    }
    mixed_weights = {"branch": 0.0, "analog": 0.0, "model": 0.0, "bot_swarm": 0.0, "llm": 0.0}
    for regime_name, regime_weight in regime_scores.items():
        template = component_templates[regime_name]
        for component, value in template.items():
            mixed_weights[component] += regime_weight * value
    return regime_scores, _normalize_weights(mixed_weights)


def build_ensemble_prediction(payload: dict[str, Any], model_prediction: dict[str, Any] | None) -> dict[str, Any]:
    branch_probability = float(payload.get('simulation', {}).get('mean_probability', 0.5) or 0.5)
    branch_consensus = float(payload.get('simulation', {}).get('consensus_score', 0.0) or 0.0)
    branch_confidence = float(payload.get('simulation', {}).get('overall_confidence', 0.0) or 0.0)
    analog_probability = 0.5 + 0.5 * float(payload.get('simulation', {}).get('analog_bias', 0.0) or 0.0)
    analog_confidence = float(payload.get('simulation', {}).get('analog_confidence', 0.0) or 0.0)
    model_probability = float((model_prediction or {}).get('bullish_probability', 0.5) or 0.5)
    bot_probability = float(payload.get('bot_swarm', {}).get('aggregate', {}).get('bullish_probability', 0.5) or 0.5)
    bot_confidence = float(payload.get('bot_swarm', {}).get('aggregate', {}).get('confidence', 0.0) or 0.0)
    llm_probability = llm_numeric_prior(payload)
    regime_mix, regime_weights = _infer_regime_weights(payload, model_prediction)
    component_conf_boost = {
        'branch': 0.82 + 0.28 * max(branch_consensus, branch_confidence),
        'analog': 0.84 + 0.24 * analog_confidence,
        'model': 0.88 + 0.18 * (1.0 - abs(model_probability - 0.5)),
        'bot_swarm': 0.86 + 0.28 * bot_confidence,
        'llm': 0.82 + 0.22 * abs(llm_probability - 0.5) * 2.0,
    }
    weights = _normalize_weights({component: regime_weights[component] * component_conf_boost[component] for component in regime_weights})
    ensemble_probability = (
        (weights['branch'] * branch_probability)
        + (weights['analog'] * analog_probability)
        + (weights['model'] * model_probability)
        + (weights['bot_swarm'] * bot_probability)
        + (weights['llm'] * llm_probability)
    )
    disagreement = max(branch_probability, analog_probability, model_probability, bot_probability, llm_probability) - min(branch_probability, analog_probability, model_probability, bot_probability, llm_probability)
    ensemble_confidence = max(
        0.0,
        min(
            1.0,
            ((0.36 * branch_consensus) + (0.20 * branch_confidence) + (0.22 * bot_confidence) + (0.22 * analog_confidence)) * (1.0 - disagreement),
        ),
    )
    signal = 'bullish' if ensemble_probability >= 0.5 else 'bearish'
    horizon_predictions = payload.get('bot_swarm', {}).get('aggregate', {}).get('horizon_predictions', [])
    return {
        'bullish_probability': round(float(ensemble_probability), 6),
        'bearish_probability': round(float(1.0 - ensemble_probability), 6),
        'signal': signal,
        'confidence': round(float(ensemble_confidence), 6),
        'components': {
            'branch_probability': round(branch_probability, 6),
            'analog_probability': round(analog_probability, 6),
            'model_probability': round(model_probability, 6),
            'bot_probability': round(bot_probability, 6),
            'llm_probability': round(llm_probability, 6),
        },
        'weights': weights,
        'regime_mix': regime_mix,
        'horizon_predictions': horizon_predictions,
    }


def _model_horizon_probabilities(model_prediction: dict[str, Any] | None) -> dict[int, float]:
    horizon_probabilities = ((model_prediction or {}).get("horizon_probabilities") or {}) if isinstance(model_prediction, dict) else {}
    mapping: dict[int, float] = {}
    for key, value in horizon_probabilities.items():
        if str(key).startswith("hold_") or str(key).startswith("confidence_"):
            continue
        digits = "".join(ch for ch in str(key) if ch.isdigit())
        if not digits:
            continue
        mapping[int(digits)] = float(value)
    return mapping


def build_v8_direct_prediction(payload: dict[str, Any], model_prediction: dict[str, Any] | None) -> dict[str, Any]:
    market = payload.get("market", {}) if isinstance(payload, dict) else {}
    current_price = float(market.get("current_price", 0.0) or 0.0)
    horizon_probabilities = _model_horizon_probabilities(model_prediction)
    raw_horizons = ((model_prediction or {}).get("horizon_probabilities") or {}) if isinstance(model_prediction, dict) else {}
    primary_probability = float(horizon_probabilities.get(15, (model_prediction or {}).get("bullish_probability", 0.5) or 0.5))
    confidence_value = float(raw_horizons.get("confidence_15m", raw_horizons.get("confidence_30m", 0.0)) or 0.0)
    signal = "bullish" if primary_probability >= float((model_prediction or {}).get("threshold", 0.5) or 0.5) else "bearish"
    horizon_predictions = []
    atr = max(float(payload.get("current_row", {}).get("atr_14", current_price * 0.0015) or current_price * 0.0015), 0.25)
    scale_map = {5: 0.22, 10: 0.36, 15: 0.52, 30: 1.00}
    for minutes in [5, 10, 15, 30]:
        probability = float(horizon_probabilities.get(minutes, primary_probability))
        bias = (probability - 0.5) * 2.0
        target_price = current_price + (atr * scale_map.get(minutes, 0.36) * bias)
        horizon_predictions.append(
            {
                "minutes": minutes,
                "target_price": round(float(target_price), 5),
                "probability": round(probability, 6),
                "confidence": round(confidence_value, 6),
            }
        )
    return {
        "mode": "v8_direct",
        "bullish_probability": round(primary_probability, 6),
        "bearish_probability": round(1.0 - primary_probability, 6),
        "signal": signal,
        "confidence": round(confidence_value, 6),
        "components": {
            "model_probability": round(primary_probability, 6),
        },
        "weights": {"model": 1.0},
        "regime_mix": {},
        "horizon_predictions": horizon_predictions,
    }


def build_v8_direct_forecast(payload: dict[str, Any], model_prediction: dict[str, Any] | None) -> dict[str, Any]:
    market = payload.get("market", {}) if isinstance(payload, dict) else {}
    current_price = float(market.get("current_price", 0.0) or 0.0)
    atr = max(float(payload.get("current_row", {}).get("atr_14", current_price * 0.0015) or current_price * 0.0015), 0.25)
    cone = list(payload.get("cone", []) if isinstance(payload, dict) else [])
    cone_by_minutes = {
        int(float(point.get("horizon", 0.0) or 0.0) * 5): point
        for point in cone
        if point.get("horizon") is not None
    }
    horizon_probabilities = _model_horizon_probabilities(model_prediction)
    primary_probability = float(horizon_probabilities.get(15, (model_prediction or {}).get("bullish_probability", 0.5) or 0.5))
    scale_map = {5: 0.22, 10: 0.36, 15: 0.52, 30: 1.00}
    points = []
    for minutes in [5, 10, 15, 30]:
        probability = float(horizon_probabilities.get(minutes, primary_probability))
        bias = (probability - 0.5) * 2.0
        final_price = current_price + (atr * scale_map.get(minutes, 0.36) * bias)
        anchor = cone_by_minutes.get(minutes, {})
        points.append(
            {
                "minutes": minutes,
                "timestamp": anchor.get("timestamp"),
                "branch_center": round(float(anchor.get("center_price", current_price) or current_price), 5),
                "bot_target": round(current_price, 5),
                "llm_target": round(current_price, 5),
                "final_price": round(float(final_price), 5),
            }
        )
    return {
        "mode": "v8_direct",
        "points": points,
        "horizon_table": [
            {
                "minutes": item["minutes"],
                "final_price": item["final_price"],
                "branch_center": item["branch_center"],
                "bot_target": item["bot_target"],
                "llm_target": item["llm_target"],
            }
            for item in points
        ],
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _trade_like_pnl(item: Mapping[str, Any]) -> float:
    for key in ("pnl_usd", "profit", "net_pnl", "realized_pnl"):
        if key in item:
            return _safe_float(item.get(key), 0.0)
    return 0.0


def _paper_performance_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    paper = dict(payload.get("paper_trading", {}) if isinstance(payload.get("paper_trading"), Mapping) else {})
    summary = dict(paper.get("summary", {}) if isinstance(paper.get("summary"), Mapping) else {})
    closed_positions = list(
        paper.get("closed_positions", paper.get("closed_trades", []))
        if isinstance(paper.get("closed_positions", paper.get("closed_trades", [])), list)
        else []
    )
    recent_closed = [item for item in closed_positions[-10:] if isinstance(item, Mapping)]
    pnl = [_trade_like_pnl(item) for item in recent_closed]
    rolling_win_rate = float(sum(1 for value in pnl if value > 0.0) / len(pnl)) if pnl else 0.0
    consecutive_losses = 0
    for value in reversed(pnl):
        if value < 0.0:
            consecutive_losses += 1
        else:
            break
    today_iso = datetime.now(timezone.utc).date().isoformat()
    daily_pnl = 0.0
    for item in closed_positions:
        if not isinstance(item, Mapping):
            continue
        close_time = str(item.get("closed_at", item.get("exit_time", item.get("timestamp", ""))))
        if today_iso in close_time:
            daily_pnl += _trade_like_pnl(item)
    return {
        "balance": round(_safe_float(summary.get("balance"), 0.0), 6),
        "equity": round(_safe_float(summary.get("equity"), _safe_float(summary.get("balance"), 0.0)), 6),
        "open_positions": _safe_int(len(list(paper.get("open_positions", []))) if isinstance(paper.get("open_positions"), list) else 0, 0),
        "closed_trades_total": _safe_int(len(closed_positions), 0),
        "rolling_win_rate_10": round(rolling_win_rate, 6),
        "consecutive_losses": int(consecutive_losses),
        "daily_pnl": round(daily_pnl, 6),
    }


def _runtime_brief(runtime: Mapping[str, Any] | None) -> dict[str, Any]:
    runtime_mapping = dict(runtime or {})
    lepl = dict(runtime_mapping.get("lepl_features", {}) if isinstance(runtime_mapping.get("lepl_features"), Mapping) else {})
    return {
        "runtime_version": str(runtime_mapping.get("runtime_version", "")),
        "decision_direction": str(runtime_mapping.get("decision_direction", runtime_mapping.get("raw_stance", "HOLD"))),
        "raw_stance": str(runtime_mapping.get("raw_stance", "HOLD")),
        "should_execute": bool(runtime_mapping.get("should_execute", False)),
        "execution_reason": str(runtime_mapping.get("execution_reason", "")),
        "confidence_tier": str(runtime_mapping.get("confidence_tier", "")),
        "cabr_score": round(_safe_float(runtime_mapping.get("cabr_score"), 0.0), 6),
        "cpm_score": round(_safe_float(runtime_mapping.get("cpm_score"), 0.0), 6),
        "cone_width_pips": round(_safe_float(runtime_mapping.get("cone_width_pips"), 0.0), 3),
        "selected_branch_label": str(runtime_mapping.get("selected_branch_label", "")),
        "v21_dir_15m_prob": round(_safe_float(runtime_mapping.get("v21_dir_15m_prob"), 0.0), 6),
        "v21_bimamba_prob": round(_safe_float(runtime_mapping.get("v21_bimamba_prob"), 0.0), 6),
        "v21_ensemble_prob": round(_safe_float(runtime_mapping.get("v21_ensemble_prob"), 0.0), 6),
        "v21_disagree_prob": round(_safe_float(runtime_mapping.get("v21_disagree_prob"), 0.0), 6),
        "v21_meta_label_prob": round(_safe_float(runtime_mapping.get("v21_meta_label_prob"), 0.0), 6),
        "v21_dangerous_branch_count": _safe_int(runtime_mapping.get("v21_dangerous_branch_count"), 0),
        "v21_regime_label": str(runtime_mapping.get("v21_regime_label", "")),
        "v21_used_branch_fallback": bool(runtime_mapping.get("v21_used_branch_fallback", False)),
        "lepl_features": {
            "kelly_fraction": round(_safe_float(lepl.get("kelly_fraction"), 0.0), 6),
            "suggested_lot": round(_safe_float(lepl.get("suggested_lot"), 0.0), 4),
            "conformal_confidence": round(_safe_float(lepl.get("conformal_confidence"), 0.0), 6),
            "dangerous_branch_count": _safe_int(lepl.get("dangerous_branch_count"), 0),
            "paper_equity": round(_safe_float(lepl.get("paper_equity"), 0.0), 6),
        },
    }


def _v22_runtime_brief(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(payload.get("v22_runtime", {}) if isinstance(payload.get("v22_runtime"), Mapping) else {})
    online_hmm = dict(runtime.get("online_hmm", payload.get("online_hmm", {})) if isinstance(runtime.get("online_hmm", payload.get("online_hmm", {})), Mapping) else {})
    circuit_breaker = dict(runtime.get("circuit_breaker", payload.get("circuit_breaker", {})) if isinstance(runtime.get("circuit_breaker", payload.get("circuit_breaker", {})), Mapping) else {})
    ensemble = dict(runtime.get("ensemble", {}) if isinstance(runtime.get("ensemble"), Mapping) else {})
    risk_check = dict(runtime.get("risk_check", {}) if isinstance(runtime.get("risk_check"), Mapping) else {})
    return {
        "runtime_version": str(runtime.get("runtime_version", "")),
        "online_hmm": {
            "regime_label": str(online_hmm.get("regime_label", "")),
            "regime_confidence": round(_safe_float(online_hmm.get("regime_confidence"), 0.0), 6),
            "persistence_count": _safe_int(online_hmm.get("persistence_count"), 0),
            "lot_size_multiplier": round(_safe_float(online_hmm.get("lot_size_multiplier"), 1.0), 6),
            "low_confidence_flag": bool(online_hmm.get("low_confidence_flag", False)),
            "persistence_conflict": bool(online_hmm.get("persistence_conflict", False)),
            "reasons": list(online_hmm.get("reasons", []))[:6],
        },
        "circuit_breaker": {
            "trading_allowed": bool(circuit_breaker.get("trading_allowed", True)),
            "state": str(circuit_breaker.get("state", "CLEAR")),
            "pause_until": circuit_breaker.get("pause_until"),
            "size_multiplier": round(_safe_float(circuit_breaker.get("size_multiplier"), 1.0), 6),
            "consecutive_losses": _safe_int(circuit_breaker.get("consecutive_losses"), 0),
            "rolling_win_rate_10": round(_safe_float(circuit_breaker.get("rolling_win_rate_10"), 0.0), 6),
            "daily_drawdown_pct": round(_safe_float(circuit_breaker.get("daily_drawdown_pct"), 0.0), 6),
            "reasons": list(circuit_breaker.get("reasons", []))[:6],
        },
        "ensemble": {
            "action": str(ensemble.get("action", "")),
            "confidence": round(_safe_float(ensemble.get("confidence"), 0.0), 6),
            "agreement_count": _safe_int(ensemble.get("agreement_count"), 0),
            "agreement_rate": round(_safe_float(ensemble.get("agreement_rate"), 0.0), 6),
            "meta_label_prob": round(_safe_float(ensemble.get("meta_label_prob"), 0.0), 6),
            "risk_score": round(_safe_float(ensemble.get("risk_score"), 0.0), 6),
            "conformal_set_size": _safe_int(ensemble.get("conformal_set_size"), 0),
            "max_lot": round(_safe_float(ensemble.get("max_lot"), 0.0), 4),
        },
        "risk_check": {
            "rr_ratio": round(_safe_float(risk_check.get("rr_ratio"), runtime.get("rr_ratio", 0.0)), 6),
            "stop_loss": round(_safe_float(risk_check.get("stop_loss"), runtime.get("stop_loss", 0.0)), 5),
            "take_profit": round(_safe_float(risk_check.get("take_profit"), runtime.get("take_profit", 0.0)), 5),
            "atr_14": round(_safe_float(risk_check.get("atr_14"), 0.0), 5),
        },
    }


def build_kimi_context_payload(payload: dict[str, Any]) -> dict[str, Any]:
    market = payload.get("market", {}) if isinstance(payload, dict) else {}
    simulation = payload.get("simulation", {}) if isinstance(payload, dict) else {}
    technical_analysis = payload.get("technical_analysis", {}) if isinstance(payload, dict) else {}
    feeds = payload.get("feeds", {}) if isinstance(payload, dict) else {}
    bot_swarm = payload.get("bot_swarm", {}) if isinstance(payload, dict) else {}
    sqt = payload.get("sqt", {}) if isinstance(payload, dict) else {}
    model_prediction = payload.get("model_prediction", {}) if isinstance(payload, dict) else {}
    current_price = _safe_float(market.get("current_price"), 0.0)
    atr_14 = _safe_float(technical_analysis.get("atr_14"), _safe_float(payload.get("current_row", {}).get("atr_14"), 0.0))
    confidence = _safe_float(simulation.get("overall_confidence"), _safe_float(simulation.get("cabr_score"), 0.0))
    regime = str(
        simulation.get(
            "detected_regime",
            simulation.get("market_memory_regime", ((model_prediction.get("model_diagnostics") or {}).get("dominant_regime", "unknown"))),
        )
    )
    candles = list(market.get("candles", []) if isinstance(market, dict) else [])
    recent_candles = [
        {
            "timestamp": item.get("timestamp"),
            "open": round(_safe_float(item.get("open"), 0.0), 5),
            "high": round(_safe_float(item.get("high"), 0.0), 5),
            "low": round(_safe_float(item.get("low"), 0.0), 5),
            "close": round(_safe_float(item.get("close"), 0.0), 5),
        }
        for item in candles[-24:]
        if isinstance(item, dict)
    ]
    recent_closes = [_safe_float(item.get("close"), current_price) for item in recent_candles]
    market_context = {
        "current_price": round(current_price, 5),
        "atr_14": round(atr_14, 5),
        "recent_5m_candles": recent_candles,
        "session_high": round(max([_safe_float(item.get("high"), current_price) for item in recent_candles], default=current_price), 5),
        "session_low": round(min([_safe_float(item.get("low"), current_price) for item in recent_candles], default=current_price), 5),
        "recent_close_change": round((recent_closes[-1] - recent_closes[0]) if len(recent_closes) >= 2 else 0.0, 5),
    }
    reduced_technical_analysis = {
        "structure": technical_analysis.get("structure"),
        "location": technical_analysis.get("location"),
        "rsi_14": round(_safe_float(technical_analysis.get("rsi_14"), 50.0), 4),
        "atr_14": round(_safe_float(technical_analysis.get("atr_14"), 0.0), 5),
        "equilibrium": round(_safe_float(technical_analysis.get("equilibrium"), current_price), 5),
        "nearest_support": technical_analysis.get("nearest_support"),
        "nearest_resistance": technical_analysis.get("nearest_resistance"),
        "order_blocks": list(technical_analysis.get("order_blocks", []))[:3],
        "fair_value_gaps": list(technical_analysis.get("fair_value_gaps", []))[:3],
    }
    reduced_bot_swarm = {
        "aggregate": dict((bot_swarm.get("aggregate") or {}) if isinstance(bot_swarm, dict) else {}),
    }
    v18_paths = {
        "consensus_path": list(simulation.get("consensus_path", []))[:4],
        "minority_path": list(simulation.get("minority_path", []))[:4],
        "outer_upper": list(simulation.get("cone_outer_upper", []))[:4],
        "outer_lower": list(simulation.get("cone_outer_lower", []))[:4],
    }
    return {
        "symbol": payload.get("symbol", "XAUUSD"),
        "market": market_context,
        "simulation": {
            "scenario_bias": str(simulation.get("direction", simulation.get("scenario_bias", "neutral"))).lower(),
            "direction": simulation.get("direction"),
            "overall_confidence": round(confidence, 6),
            "cabr_score": round(_safe_float(simulation.get("cabr_score"), 0.0), 6),
            "cpm_score": round(_safe_float(simulation.get("cpm_score"), 0.0), 6),
            "cone_width_pips": round(_safe_float(simulation.get("cone_width_pips"), 0.0), 3),
            "contradiction_type": simulation.get("contradiction_type", payload.get("contradiction", {}).get("type", "unknown")),
            "detected_regime": regime,
            "cone_treatment": simulation.get("cone_treatment", payload.get("contradiction", {}).get("cone_treatment", "normal")),
            "hurst_overall": round(_safe_float(simulation.get("hurst_overall"), 0.5), 6),
            "hurst_positive": round(_safe_float(simulation.get("hurst_positive"), 0.5), 6),
            "hurst_negative": round(_safe_float(simulation.get("hurst_negative"), 0.5), 6),
            "hurst_asymmetry": round(_safe_float(simulation.get("hurst_asymmetry"), 0.0), 6),
            "testosterone_index": simulation.get("testosterone_index", {}),
            "suggested_lot": round(_safe_float(simulation.get("suggested_lot"), 0.0), 4),
            "cone_c_m": round(_safe_float(simulation.get("cone_c_m"), 0.0), 6),
            "entry_zone": simulation.get("entry_zone", []),
            "confidence_tier": str(simulation.get("confidence_tier", "")),
            "should_execute": bool(simulation.get("should_execute", False)),
            "execution_reason": str(simulation.get("execution_reason", "")),
        },
        "technical_analysis": reduced_technical_analysis,
        "bot_swarm": reduced_bot_swarm,
        "news_feed": list((feeds.get("news") or []))[:8],
        "public_discussions": list((feeds.get("public_discussions") or []))[:8],
        "sqt": sqt,
        "mfg": payload.get("mfg", {}),
        "v18_paths": v18_paths,
        "v21_runtime": _runtime_brief(payload.get("v21_runtime") if isinstance(payload.get("v21_runtime"), Mapping) else payload.get("v19_runtime")),
        "v22_runtime": _v22_runtime_brief(payload),
        "live_performance": _paper_performance_summary(payload),
        "risk_controls": {
            "broker": {
                "autotrade_enabled": bool(((payload.get("broker") or {}).get("autotrade_enabled")) if isinstance(payload.get("broker"), Mapping) else False),
                "last_order_volume": round(
                    _safe_float(
                        (((payload.get("broker") or {}).get("last_order") or {}).get("volume"))
                        if isinstance(((payload.get("broker") or {}).get("last_order")), Mapping)
                        else 0.0,
                        0.0,
                    ),
                    4,
                ),
                "suggested_volume": round(
                    _safe_float(
                        (((payload.get("broker") or {}).get("last_order") or {}).get("suggested_volume"))
                        if isinstance(((payload.get("broker") or {}).get("last_order")), Mapping)
                        else 0.0,
                        0.0,
                    ),
                    4,
                ),
            }
        },
    }


def fallback_kimi_judge(payload: dict[str, Any]) -> dict[str, Any]:
    market = payload.get("market", {}) if isinstance(payload, dict) else {}
    simulation = payload.get("simulation", {}) if isinstance(payload, dict) else {}
    technical_analysis = payload.get("technical_analysis", {}) if isinstance(payload, dict) else {}
    sqt = payload.get("sqt", {}) if isinstance(payload, dict) else {}
    current_price = _safe_float(market.get("current_price"), 0.0)
    atr = _safe_float(technical_analysis.get("atr_14"), _safe_float(payload.get("current_row", {}).get("atr_14"), current_price * 0.0015))
    direction = str(simulation.get("direction", "HOLD")).upper()
    contradiction = str(simulation.get("contradiction_type", "unknown")).lower()
    sqt_label = str(sqt.get("label", simulation.get("sqt_label", "NEUTRAL"))).upper()
    if sqt_label == "COLD" or contradiction == "full_disagreement":
        direction = "HOLD"
    support = _safe_float((technical_analysis.get("nearest_support") or {}).get("price"), current_price - atr)
    resistance = _safe_float((technical_analysis.get("nearest_resistance") or {}).get("price"), current_price + atr)
    if direction == "BUY":
        entry_zone = [round(current_price - (0.20 * atr), 2), round(current_price + (0.12 * atr), 2)]
        stop_loss = round(min(support, current_price - atr), 2)
        take_profit = round(max(resistance, current_price + (1.4 * atr)), 2)
    elif direction == "SELL":
        entry_zone = [round(current_price - (0.12 * atr), 2), round(current_price + (0.20 * atr), 2)]
        stop_loss = round(max(resistance, current_price + atr), 2)
        take_profit = round(min(support, current_price - (1.4 * atr)), 2)
    else:
        entry_zone = []
        stop_loss = None
        take_profit = None
    return {
        "stance": direction if direction in {"BUY", "SELL"} else "HOLD",
        "confidence": str(simulation.get("confidence_tier", "LOW")).replace("_", " ").upper(),
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "hold_time": "current_bar" if direction in {"BUY", "SELL"} else "skip",
        "final_call": "BUY" if direction == "BUY" else "SELL" if direction == "SELL" else "SKIP",
        "final_summary": f"{'BUY' if direction == 'BUY' else 'SELL' if direction == 'SELL' else 'SKIP'} - fallback judge is using the live V18 desk values because the remote Kimi packet is unavailable.",
        "market_only_summary": {
            "call": "BUY" if direction == "BUY" else "SELL" if direction == "SELL" else "SKIP",
            "summary": f"Live market-only read is {'bullish' if direction == 'BUY' else 'bearish' if direction == 'SELL' else 'mixed'}, using current price and structure.",
            "reasoning": f"Current price is {current_price:.2f}, structure is {technical_analysis.get('structure', 'unknown')}, and support/resistance are framing the bar.",
        },
        "v18_summary": {
            "call": "BUY" if direction == "BUY" else "SELL" if direction == "SELL" else "SKIP",
            "summary": f"V18-only read follows the simulator direction {direction}.",
            "reasoning": f"CABR is {_safe_float(simulation.get('cabr_score'), 0.0):.1%}, confidence tier is {simulation.get('confidence_tier', 'LOW')}, and SQT is {sqt_label}.",
        },
        "combined_summary": {
            "call": "BUY" if direction == "BUY" else "SELL" if direction == "SELL" else "SKIP",
            "summary": f"Combined read is {'actionable' if direction in {'BUY', 'SELL'} else 'a skip for now'} for the current 15-minute bar.",
            "reasoning": f"Fallback combines price structure, V18 direction {direction}, and risk gates such as SQT {sqt_label}.",
        },
        "reasoning": f"Fallback judge derived from the simulator direction {direction}, CABR {_safe_float(simulation.get('cabr_score'), 0.0):.1%}, and current structure.",
        "key_risk": "This is a local fallback because the remote Kimi response is unavailable.",
        "crowd_note": "Crowd state is being proxied locally from WLTC and bot swarm data.",
        "regime_note": f"Structure is {technical_analysis.get('structure', 'unknown')} in {technical_analysis.get('location', 'unknown')} location.",
        "invalidation": stop_loss,
    }


def fallback_local_v19_judge(payload: dict[str, Any]) -> dict[str, Any]:
    content = fallback_kimi_judge(payload)
    final_call = str(content.get("final_call", "SKIP")).upper()
    content["final_summary"] = (
        f"{final_call} - local V19 fallback is mirroring the live desk values because the distilled student is unavailable."
    )
    content["market_only_summary"] = {
        "call": final_call,
        "summary": f"Local V19 fallback market-only read is {final_call}.",
        "reasoning": "The distilled student is unavailable, so this mirror uses the live desk market values only.",
    }
    content["v18_summary"] = {
        "call": final_call,
        "summary": f"Local V19 fallback is borrowing the desk simulator stance {final_call}.",
        "reasoning": "This is a temporary mirror of the local desk rather than the trained distilled student.",
    }
    content["combined_summary"] = {
        "call": final_call,
        "summary": f"Local V19 fallback combined read is {final_call}.",
        "reasoning": "Use this only as a placeholder while the distilled student is unavailable.",
    }
    content["reasoning"] = "Local V19 fallback is active because the distilled SJD model could not be loaded."
    content["key_risk"] = "This is not the trained local student output; it is only a local desk fallback."
    content["crowd_note"] = "Crowd context is still present, but the trained V19 student did not score this bar."
    content["regime_note"] = "The local V19 student is unavailable, so regime handling is mirrored from the desk."
    return content


def _kimi_call_label(content: Mapping[str, Any]) -> str:
    raw = str(content.get("final_call", content.get("stance", "HOLD"))).strip().upper()
    if raw in {"BUY", "SELL", "SKIP"}:
        return raw
    if raw == "HOLD":
        return "SKIP"
    return "SKIP"


def _normalize_kimi_summary_block(block: Any, *, fallback_call: str, fallback_summary: str, fallback_reasoning: str) -> dict[str, str]:
    if isinstance(block, dict):
        call = str(block.get("call", fallback_call)).strip().upper() or fallback_call
        if call == "HOLD":
            call = "SKIP"
        summary = str(block.get("summary", fallback_summary)).strip() or fallback_summary
        reasoning = str(block.get("reasoning", fallback_reasoning)).strip() or fallback_reasoning
        return {"call": call, "summary": summary, "reasoning": reasoning}
    return {"call": fallback_call, "summary": fallback_summary, "reasoning": fallback_reasoning}


def _normalize_kimi_content(payload: dict[str, Any], raw_content: Mapping[str, Any]) -> dict[str, Any]:
    content = dict(raw_content)
    stance = str(content.get("stance", "HOLD")).strip().upper() or "HOLD"
    if stance not in {"BUY", "SELL", "HOLD"}:
        stance = "HOLD"
    call = _kimi_call_label({"final_call": content.get("final_call"), "stance": stance})
    reasoning = str(content.get("reasoning", "No Kimi reasoning available.")).strip() or "No Kimi reasoning available."
    key_risk = str(content.get("key_risk", "No Kimi risk note available.")).strip() or "No Kimi risk note available."
    market = payload.get("market", {}) if isinstance(payload, dict) else {}
    simulation = payload.get("simulation", {}) if isinstance(payload, dict) else {}
    technical_analysis = payload.get("technical_analysis", {}) if isinstance(payload, dict) else {}
    current_price = _safe_float(market.get("current_price"), 0.0)
    v18_summary = _normalize_kimi_summary_block(
        content.get("v18_summary"),
        fallback_call=call,
        fallback_summary=f"V18-only read is {call}.",
        fallback_reasoning=f"Direction {simulation.get('direction', 'HOLD')} with CABR {_safe_float(simulation.get('cabr_score'), 0.0):.1%} and Hurst {_safe_float(simulation.get('hurst_asymmetry'), 0.0):.3f}.",
    )
    market_summary = _normalize_kimi_summary_block(
        content.get("market_only_summary"),
        fallback_call=call,
        fallback_summary=f"Market-only read is {call}.",
        fallback_reasoning=f"Live price is {current_price:.2f}, structure is {technical_analysis.get('structure', 'unknown')}, and key risk is {key_risk}",
    )
    combined_summary = _normalize_kimi_summary_block(
        content.get("combined_summary"),
        fallback_call=call,
        fallback_summary=f"Combined read is {call}.",
        fallback_reasoning=reasoning,
    )
    content["stance"] = stance
    content["confidence"] = str(content.get("confidence", "VERY_LOW")).strip().upper() or "VERY_LOW"
    content["final_call"] = call
    content["final_summary"] = str(content.get("final_summary", f"{call} - {combined_summary['summary']}")).strip() or f"{call} - {combined_summary['summary']}"
    content["market_only_summary"] = market_summary
    content["v18_summary"] = v18_summary
    content["combined_summary"] = combined_summary
    content["reasoning"] = reasoning
    content["key_risk"] = key_risk
    content["crowd_note"] = str(content.get("crowd_note", "No crowd note available.")).strip() or "No crowd note available."
    content["regime_note"] = str(content.get("regime_note", "No regime note available.")).strip() or "No regime note available."
    return content


def _build_kimi_projection(payload: dict[str, Any], content: Mapping[str, Any]) -> dict[str, Any]:
    market = payload.get("market", {}) if isinstance(payload, dict) else {}
    current_price = _safe_float(market.get("current_price"), 0.0)
    stance = str(content.get("stance", "HOLD")).upper()
    entry_zone = content.get("entry_zone", [])
    stop_loss = content.get("stop_loss")
    take_profit = content.get("take_profit")
    if stance not in {"BUY", "SELL"}:
        return {"label": "Kimi path unavailable for HOLD/SKIP.", "points": []}
    if not isinstance(entry_zone, list) or len(entry_zone) != 2:
        return {"label": "Kimi path unavailable because entry zone is incomplete.", "points": []}
    entry_mid = (_safe_float(entry_zone[0], current_price) + _safe_float(entry_zone[1], current_price)) / 2.0
    target = _safe_float(take_profit, current_price)
    stop = _safe_float(stop_loss, current_price)
    midpoint = current_price + ((target - current_price) * 0.55)
    candles = list(market.get("candles", []) if isinstance(market, dict) else [])
    base_timestamp = str(candles[-1].get("timestamp")) if candles else ""
    if base_timestamp:
        try:
            base_dt = datetime.fromisoformat(base_timestamp.replace("Z", "+00:00"))
        except Exception:
            base_dt = datetime.now(timezone.utc)
    else:
        base_dt = datetime.now(timezone.utc)
    points = [
        {"minutes": 0, "timestamp": base_dt.isoformat(), "price": round(current_price, 5)},
        {"minutes": 5, "timestamp": (base_dt + timedelta(minutes=5)).isoformat(), "price": round(entry_mid, 5)},
        {"minutes": 10, "timestamp": (base_dt + timedelta(minutes=10)).isoformat(), "price": round(midpoint, 5)},
        {"minutes": 15, "timestamp": (base_dt + timedelta(minutes=15)).isoformat(), "price": round(target, 5)},
    ]
    return {
        "label": f"Kimi {stance} path",
        "entry_mid": round(entry_mid, 5),
        "target": round(target, 5),
        "stop_loss": round(stop, 5),
        "points": points,
    }


def _decorate_kimi_judge(payload: dict[str, Any], kimi_judge: Mapping[str, Any]) -> dict[str, Any]:
    decorated = dict(kimi_judge)
    raw_content = decorated.get("content", {}) if isinstance(decorated.get("content"), Mapping) else {}
    normalized_content = _normalize_kimi_content(payload, raw_content)
    decorated["content"] = normalized_content
    decorated["projection_path"] = _build_kimi_projection(payload, normalized_content)
    decorated["separate_from_v18"] = True
    return decorated


def _decorate_local_judge(payload: dict[str, Any], local_judge: Mapping[str, Any]) -> dict[str, Any]:
    decorated = _decorate_kimi_judge(payload, local_judge)
    decorated["judge_name"] = "local_v19"
    decorated["independent"] = True
    return decorated


def _judge_comparison(
    kimi_judge: Mapping[str, Any],
    local_judge: Mapping[str, Any],
    v19_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kimi_content = dict(kimi_judge.get("content", {}) if isinstance(kimi_judge.get("content"), Mapping) else {})
    local_content = dict(local_judge.get("content", {}) if isinstance(local_judge.get("content"), Mapping) else {})
    kimi_call = _kimi_call_label(kimi_content)
    local_call = _kimi_call_label(local_content)
    agree = kimi_call == local_call
    agreement_label = "aligned" if agree else "split"
    if agree:
        summary = f"Kimi and the local V19 student both read the current 15-minute bar as {kimi_call}."
        reasoning = "Both judges are aligned, so this bar is a clean comparison case for manual testing."
        preferred_source = "aligned"
    else:
        summary = f"Kimi is {kimi_call} while the local V19 student is {local_call}."
        reasoning = "The judges disagree, so this is the exact kind of bar worth tracking to compare real-world accuracy."
        preferred_source = "manual_compare"
    return {
        "agreement": bool(agree),
        "agreement_label": agreement_label,
        "kimi_call": kimi_call,
        "local_call": local_call,
        "summary": summary,
        "reasoning": reasoning,
        "preferred_source": preferred_source,
        "v19_should_execute": bool((v19_runtime or {}).get("should_execute", False)),
        "v19_execution_reason": str((v19_runtime or {}).get("execution_reason", "")),
    }


def validate_sequence_shape(sequence: list[list[float]], sequence_len: int, feature_dim: int) -> None:
    if len(sequence) != sequence_len:
        raise ValueError(f'Expected sequence length {sequence_len}, got {len(sequence)}')
    for row_index, row in enumerate(sequence):
        if len(row) != feature_dim:
            raise ValueError(f'Expected feature width {feature_dim} at row {row_index}, got {len(row)}')


def classify_probability(probability: float, threshold: float) -> str:
    return 'bullish' if probability >= threshold else 'bearish'


def load_model_manifest() -> dict[str, Any]:
    if not MODEL_MANIFEST_PATH.exists():
        return {
            'sequence_len': SEQUENCE_LEN,
            'feature_dim': FEATURE_DIM_TOTAL,
            'classification_threshold': 0.5,
        }
    return json.loads(MODEL_MANIFEST_PATH.read_text(encoding='utf-8'))


def load_json_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _resolve_checkpoint() -> Path:
    if TFT_CHECKPOINT_PATH.exists():
        return TFT_CHECKPOINT_PATH
    if LEGACY_TFT_CHECKPOINT_PATH.exists():
        return LEGACY_TFT_CHECKPOINT_PATH
    raise FileNotFoundError('No trained checkpoint available for inference.')


class ModelServer:
    def __init__(self) -> None:
        if torch is None:
            raise ImportError('PyTorch is required for inference.')
        self.manifest = load_model_manifest()
        self.sequence_len = int(self.manifest.get('sequence_len', SEQUENCE_LEN))
        self.feature_dim = int(self.manifest.get('feature_dim', FEATURE_DIM_TOTAL))
        self.threshold = float(self.manifest.get('classification_threshold', 0.5))
        self.device = get_torch_device()
        config_payload = self.manifest.get('model_config', {})
        self.horizon_labels = list(self.manifest.get('horizon_labels', ['5m']))
        self.output_labels = list(self.manifest.get('output_labels', [f'target_{label}' for label in self.horizon_labels]))
        config = NexusTFTConfig(
            input_dim=int(config_payload.get('input_dim', self.feature_dim)),
            hidden_dim=int(config_payload.get('hidden_dim', 128)),
            lstm_layers=int(config_payload.get('lstm_layers', 2)),
            dropout=float(config_payload.get('dropout', 0.1)),
            output_dim=int(config_payload.get('output_dim', len(self.horizon_labels))),
            regime_count=int(config_payload.get('regime_count', 4)),
            router_hidden_dim=int(config_payload.get('router_hidden_dim', 64)),
            router_temperature=float(config_payload.get('router_temperature', 1.0)),
        )
        self.model = NexusTFT(config).to(self.device)
        load_checkpoint_with_expansion(self.model, _resolve_checkpoint(), new_input_dim=config.input_dim)
        self.model.eval()

    def predict(self, sequence: list[list[float]]) -> PredictResponse:
        validate_sequence_shape(sequence, self.sequence_len, self.feature_dim)
        with torch.no_grad():
            tensor = torch.tensor([sequence], dtype=torch.float32, device=self.device)
            raw_output, diagnostics = self.model(tensor, return_diagnostics=True)
            raw_output = raw_output.detach().cpu().numpy()
        if raw_output.ndim == 1:
            horizon_values = raw_output.astype(float).tolist()
        else:
            horizon_values = raw_output[0].astype(float).tolist()
        output_map = {label: float(value) for label, value in zip(self.output_labels, horizon_values)}
        if any(label.startswith('target_') for label in self.output_labels):
            horizon_probabilities = {label.replace('target_', ''): value for label, value in output_map.items() if label.startswith('target_')}
            hold_probabilities = {label.replace('hold_', ''): value for label, value in output_map.items() if label.startswith('hold_')}
            confidence_outputs = {label.replace('confidence_', ''): value for label, value in output_map.items() if label.startswith('confidence_')}
        else:
            horizon_probabilities = {label: float(value) for label, value in zip(self.horizon_labels, horizon_values)}
            hold_probabilities = {}
            confidence_outputs = {}
        primary_horizon = str(self.manifest.get('primary_horizon', self.horizon_labels[0] if self.horizon_labels else '5m'))
        bullish_probability = float(horizon_probabilities.get(primary_horizon, next(iter(horizon_probabilities.values()), 0.5)))
        return PredictResponse(
            bullish_probability=bullish_probability,
            bearish_probability=float(1.0 - bullish_probability),
            signal=classify_probability(bullish_probability, self.threshold),
            threshold=self.threshold,
            sequence_len=self.sequence_len,
            feature_dim=self.feature_dim,
            horizon_probabilities={**horizon_probabilities, **{f'hold_{k}': v for k, v in hold_probabilities.items()}, **{f'confidence_{k}': v for k, v in confidence_outputs.items()}},
        )

    def predict_dict(self, sequence: list[list[float]]) -> dict[str, Any]:
        validate_sequence_shape(sequence, self.sequence_len, self.feature_dim)
        with torch.no_grad():
            tensor = torch.tensor([sequence], dtype=torch.float32, device=self.device)
            raw_output, diagnostics = self.model(tensor, return_diagnostics=True)
            raw_output = raw_output.detach().cpu().numpy()
        if raw_output.ndim == 1:
            horizon_values = raw_output.astype(float).tolist()
        else:
            horizon_values = raw_output[0].astype(float).tolist()
        output_map = {label: float(value) for label, value in zip(self.output_labels, horizon_values)}
        if any(label.startswith('target_') for label in self.output_labels):
            horizon_probabilities = {label.replace('target_', ''): value for label, value in output_map.items() if label.startswith('target_')}
            hold_probabilities = {label.replace('hold_', ''): value for label, value in output_map.items() if label.startswith('hold_')}
            confidence_outputs = {label.replace('confidence_', ''): value for label, value in output_map.items() if label.startswith('confidence_')}
        else:
            horizon_probabilities = {label: float(value) for label, value in zip(self.horizon_labels, horizon_values)}
            hold_probabilities = {}
            confidence_outputs = {}
        primary_horizon = str(self.manifest.get('primary_horizon', self.horizon_labels[0] if self.horizon_labels else '5m'))
        bullish_probability = float(horizon_probabilities.get(primary_horizon, next(iter(horizon_probabilities.values()), 0.5)))
        regime_names = list(self.manifest.get('regime_labels', ['trend', 'reversal', 'macro_shock', 'balanced']))
        regime_probs_tensor = diagnostics.get('regime_probabilities')
        regime_probs = regime_probs_tensor[0].detach().cpu().numpy().astype(float).tolist() if regime_probs_tensor is not None else []
        regime_probabilities = {name: round(value, 6) for name, value in zip(regime_names, regime_probs)}
        return {
            "bullish_probability": float(bullish_probability),
            "bearish_probability": float(1.0 - bullish_probability),
            "signal": str(classify_probability(bullish_probability, self.threshold)),
            "threshold": float(self.threshold),
            "sequence_len": int(self.sequence_len),
            "feature_dim": int(self.feature_dim),
            "horizon_probabilities": {**horizon_probabilities, **{f'hold_{k}': v for k, v in hold_probabilities.items()}, **{f'confidence_{k}': v for k, v in confidence_outputs.items()}},
            "model_diagnostics": {
                "regime_probabilities": regime_probabilities,
                "dominant_regime": max(regime_probabilities, key=regime_probabilities.get) if regime_probabilities else "unknown",
                "regime_confidence": round(max(regime_probabilities.values()), 6) if regime_probabilities else 0.0,
            },
        }


def create_app() -> Any:
    if FastAPI is None:
        raise ImportError('fastapi and pydantic are required to create the inference app.')

    server = ModelServer()
    app = FastAPI(title='Nexus Trader Inference API', version='0.2.0')
    frontend_served = False
    if FRONTEND_DIST_PATH.exists():
        app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIST_PATH), html=True), name="nexus_ui")
        frontend_served = True
    default_mode = os.getenv("NEXUS_V16_MODE", "frequency").strip().lower() or "frequency"
    s3pta_enabled = os.getenv("NEXUS_S3PTA_ENABLED", "0") == "1"
    paper_trade_accumulator = PaperTradeAccumulator() if s3pta_enabled else None
    sqt_tracker = SimulationQualityTracker()
    scored_predictions: set[str] = set()
    paper_trader = PaperTradingEngine(starting_balance=float(os.getenv("NEXUS_PAPER_START_BALANCE", "1000")))
    feed_manager = LiveFeedManager()
    feed_manager.set_paper_engine(paper_trader)
    kimi_cache: dict[str, dict[str, Any]] = {}

    def _kimi_cache_key(symbol: str, active_mode: str, llm_provider: str, llm_model: str | None) -> str:
        bucket = str(int(time.time() // 900))
        return "|".join(
            [
                str(symbol).upper(),
                str(active_mode).strip().lower() or "frequency",
                str(llm_provider).strip().lower() or "nvidia_nim",
                (llm_model or "").strip() or "",
                bucket,
            ]
        )

    def _current_prices(symbols: set[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for symbol_name in sorted({str(item).upper() for item in symbols if item}):
            try:
                price = float(fetch_live_quote(symbol_name) or 0.0)
                if price <= 0.0:
                    continue
                prices[symbol_name] = price
            except Exception:
                continue
        return prices

    def _paper_state(symbol: str | None = None) -> dict[str, Any]:
        preview = paper_trader.state()
        open_symbols = {str(position.get("symbol", "")).upper() for position in preview.get("open_positions", [])}
        if symbol:
            open_symbols.add(str(symbol).upper())
        return paper_trader.state(current_prices=_current_prices(open_symbols))

    def _live_price(symbol: str) -> float:
        return _current_prices({str(symbol).upper()}).get(str(symbol).upper(), 0.0)

    def _base_dashboard_payload(symbol: str, active_mode: str) -> dict[str, Any]:
        payload = build_fast_dashboard_payload(symbol)
        payload["paper_trading"] = _paper_state(symbol)
        payload["realtime_chart"] = build_realtime_chart_payload(symbol)
        payload["sqt"] = sqt_tracker.summary()
        v16_payload = build_v16_simulation_result(
            payload,
            None,
            mode=active_mode,
            sqt=sqt_tracker,
            eci_context=payload.get("eci"),
        )
        payload["simulation"] = dict(payload.get("simulation", {})) | dict(v16_payload.get("simulation", {}))
        payload["cone"] = v16_payload.get("cone", payload.get("cone", {}))
        payload["relativistic_cone"] = v16_payload.get("relativistic_cone", {})
        payload["final_forecast"] = v16_payload.get("final_forecast", {})
        payload["v16"] = v16_payload
        payload["ensemble_prediction"] = _v16_ensemble(payload)
        payload["stack_mode"] = "dashboard_fast_path"
        payload["mode"] = active_mode
        payload["manual_trading_mode"] = True
        return payload

    def _resolve_local_judge(payload: dict[str, Any]) -> dict[str, Any]:
        local_context = build_kimi_context_payload(payload)
        symbol_name = str(payload.get("symbol", "XAUUSD"))
        local_judge = request_local_sjd_judge(symbol_name, local_context)
        if not local_judge.get("available", False):
            local_judge = {
                "available": False,
                "provider": "local_sjd",
                "model": "v19_sjd_local",
                "error": local_judge.get("error", "Local V19 student request failed."),
                "content": fallback_local_v19_judge(payload),
            }
        return _decorate_local_judge(payload, local_judge)

    def _attach_v19_runtime(payload: dict[str, Any], local_judge: Mapping[str, Any]) -> dict[str, Any]:
        runtime = build_v19_runtime_state(payload, local_judge=local_judge)
        payload["local_judge"] = dict(local_judge)
        payload["v19_runtime"] = runtime
        simulation = dict(payload.get("simulation", {}))
        if runtime.get("available", False):
            simulation["v19_cabr_score"] = runtime.get("cabr_score")
            simulation["v19_confidence_tier"] = runtime.get("confidence_tier")
            simulation["v19_sqt_label"] = runtime.get("sqt_label")
            simulation["v19_should_execute"] = runtime.get("should_execute")
            simulation["v19_execution_reason"] = runtime.get("execution_reason")
            simulation["v19_selected_branch_label"] = runtime.get("selected_branch_label")
            simulation["v19_selected_branch_id"] = runtime.get("selected_branch_id")
        payload["simulation"] = simulation
        payload["stack_mode"] = "v19_live_dual_judge"
        return payload

    def _cached_or_fallback_kimi(payload: dict[str, Any], active_mode: str, llm_provider: str, llm_model: str | None) -> dict[str, Any]:
        cache_key = _kimi_cache_key(payload.get("symbol", "XAUUSD"), active_mode, llm_provider, llm_model)
        cached = kimi_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        fallback = {
            "available": False,
            "provider": llm_provider,
            "model": llm_model or "",
            "reason": "awaiting_15m_kimi_refresh",
            "content": fallback_kimi_judge(payload),
        }
        return _decorate_kimi_judge(payload, fallback)

    def _resolve_kimi_judge(payload: dict[str, Any], active_mode: str, llm_provider: str, llm_model: str | None, *, force: bool = False) -> dict[str, Any]:
        cache_key = _kimi_cache_key(payload.get("symbol", "XAUUSD"), active_mode, llm_provider, llm_model)
        if not force and cache_key in kimi_cache:
            return copy.deepcopy(kimi_cache[cache_key])
        kimi_context = build_kimi_context_payload(payload)
        payload_symbol = str(payload.get("symbol", "XAUUSD"))
        if is_nvidia_nim_provider(llm_provider):
            kimi_judge = request_kimi_judge(payload_symbol, kimi_context, provider=llm_provider, model=llm_model)
            if not kimi_judge.get("available", False):
                kimi_judge = {
                    "available": False,
                    "provider": llm_provider,
                    "model": llm_model or "",
                    "error": kimi_judge.get("error", "Kimi request failed."),
                    "content": fallback_kimi_judge(payload),
                }
        else:
            kimi_judge = {
                "available": False,
                "provider": llm_provider,
                "model": llm_model or "",
                "reason": "provider_not_nim",
                "content": fallback_kimi_judge(payload),
            }
        decorated = _decorate_kimi_judge(payload, kimi_judge)
        kimi_cache[cache_key] = copy.deepcopy(decorated)
        return decorated

    def _refresh_sqt(symbol: str) -> dict[str, Any]:
        monitor = build_live_monitor(symbol)
        for item in monitor.get("recent_simulations", []):
            if item.get("realized_direction") not in {"bullish", "bearish"}:
                continue
            key = f"{symbol}:{item.get('anchor_timestamp')}"
            if key in scored_predictions:
                continue
            predicted = "BUY" if str(item.get("scenario_bias", "bullish")) == "bullish" else "SELL"
            actual = "BUY" if str(item.get("realized_direction", "bullish")) == "bullish" else "SELL"
            sqt_tracker.record(predicted, actual, str(item.get("confidence_tier", "uncertain")))
            scored_predictions.add(key)
        monitor["sqt"] = sqt_tracker.summary()
        monitor["paper_trading"] = _paper_state(symbol)
        return monitor

    def _v16_ensemble(payload: dict[str, Any]) -> dict[str, Any]:
        simulation = payload.get("simulation", {}) if isinstance(payload, dict) else {}
        final_points = list((payload.get("final_forecast") or {}).get("points", []))
        model_prediction = payload.get("model_prediction", {}) if isinstance(payload, dict) else {}
        bullish_probability = float((model_prediction or {}).get("bullish_probability", 0.5) or 0.5)
        if str(simulation.get("direction", "NEUTRAL")).upper() == "SELL":
            bullish_probability = 1.0 - bullish_probability
        return {
            "signal": "bullish" if str(simulation.get("direction", "NEUTRAL")).upper() == "BUY" else "bearish" if str(simulation.get("direction", "NEUTRAL")).upper() == "SELL" else "neutral",
            "confidence": float(simulation.get("cabr_score", 0.5) or 0.5),
            "bullish_probability": round(float(bullish_probability), 6),
            "bearish_probability": round(float(1.0 - bullish_probability), 6),
            "horizon_predictions": [
                {
                    "minutes": int(item.get("minutes", 0) or 0),
                    "target_price": float(item.get("final_price", 0.0) or 0.0),
                    "confidence": float(simulation.get("cabr_score", 0.5) or 0.5),
                }
                for item in final_points
            ],
        }

    @app.get('/health')
    def health():
        return {'status': 'ok', 'sequence_len': server.sequence_len, 'feature_dim': server.feature_dim}

    @app.get('/api/system/telemetry')
    def system_telemetry():
        return read_system_telemetry()

    @app.get('/metadata')
    def metadata():
        payload = dict(server.manifest)
        payload['latest_snapshot'] = load_json_artifact(LATEST_MARKET_SNAPSHOT_PATH)
        return payload

    @app.get('/latest-cone')
    def latest_cone():
        return load_json_artifact(LATEST_MARKET_SNAPSHOT_PATH)

    @app.get('/latest-branches')
    def latest_branches():
        return load_json_artifact(FUTURE_BRANCHES_PATH) if FUTURE_BRANCHES_PATH.exists() else []

    @app.get('/api/simulate-live')
    def simulate_live(
        symbol: str = 'XAUUSD',
        llm_provider: str = 'lm_studio',
        llm_model: str | None = None,
        mode: str | None = None,
    ):
        _refresh_sqt(symbol)
        payload = build_live_simulation(
            symbol,
            sequence_len=server.sequence_len,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        sequence = payload.pop('sequence', None)
        if sequence:
            try:
                payload['model_prediction'] = server.predict_dict(sequence)
            except Exception as exc:  # pragma: no cover
                payload['model_prediction'] = {'error': str(exc)}
        else:
            payload['model_prediction'] = None
        active_mode = (mode or default_mode).strip().lower() or "frequency"
        payload['paper_trading'] = _paper_state(symbol)
        v16_payload = build_v16_simulation_result(
            payload,
            payload.get('model_prediction'),
            mode=active_mode,
            sqt=sqt_tracker,
            eci_context=payload.get('eci'),
        )
        payload['simulation'] = dict(payload.get('simulation', {})) | dict(v16_payload.get('simulation', {}))
        payload['cone'] = v16_payload.get('cone', payload.get('cone', {}))
        payload['relativistic_cone'] = v16_payload.get('relativistic_cone', {})
        payload['final_forecast'] = v16_payload.get('final_forecast', {})
        payload['v16'] = v16_payload
        payload['ensemble_prediction'] = _v16_ensemble(payload)
        payload['history_entry'] = record_simulation_history(payload, payload.get('model_prediction'))
        payload['stack_mode'] = 'v16_simulator'
        payload['mode'] = active_mode
        payload['manual_trading_mode'] = True
        payload['sqt'] = sqt_tracker.summary()
        payload['paper_trading'] = _paper_state(symbol)
        payload['realtime_chart'] = build_realtime_chart_payload(symbol)
        payload = _attach_v19_runtime(payload, _resolve_local_judge(payload))
        kimi_context = build_kimi_context_payload(payload)
        if is_nvidia_nim_provider(llm_provider):
            kimi_judge = request_kimi_judge(symbol, kimi_context, provider=llm_provider, model=llm_model)
            if not kimi_judge.get("available", False):
                kimi_judge = {
                    "available": False,
                    "provider": llm_provider,
                    "model": llm_model or "",
                    "error": kimi_judge.get("error", "Kimi request failed."),
                    "content": fallback_kimi_judge(payload),
                }
        else:
            kimi_judge = {
                "available": False,
                "provider": llm_provider,
                "model": llm_model or "",
                "reason": "provider_not_nim",
                "content": fallback_kimi_judge(payload),
            }
        payload['kimi_judge'] = _decorate_kimi_judge(payload, kimi_judge)
        payload["judge_comparison"] = _judge_comparison(payload["kimi_judge"], payload["local_judge"], payload.get("v19_runtime"))
        if paper_trade_accumulator is not None:
            try:
                market = payload.get('market', {}) if isinstance(payload, dict) else {}
                current_price = float(market.get('current_price', 0.0) or 0.0)
                simulation = payload.get('simulation', {}) if isinstance(payload, dict) else {}
                direction = str(simulation.get('direction', 'BUY')).upper()
                regime = str(((payload.get('model_prediction') or {}).get('model_diagnostics') or {}).get('dominant_regime', 'unknown'))
                candles = market.get('candles', []) if isinstance(market, dict) else []
                timestamp = str(candles[-1].get('timestamp', '')) if candles else ""
                if current_price > 0.0 and timestamp:
                    paper_trade_accumulator.log_trade(
                        symbol=symbol,
                        direction=direction,
                        uts_score=float(simulation.get('bst_score', 0.5) or 0.5),
                        cabr_score=float(simulation.get('cabr_score', 0.5) or 0.5),
                        regime=regime,
                        entry_price=current_price,
                        entry_time=timestamp,
                        exit_time=timestamp,
                    )
            except Exception:
                pass
        return payload

    @app.get('/api/live-monitor')
    def live_monitor(symbol: str = 'XAUUSD'):
        return _refresh_sqt(symbol)

    @app.get('/api/dashboard/live')
    def dashboard_live(
        symbol: str = 'XAUUSD',
        llm_provider: str = 'nvidia_nim',
        llm_model: str | None = None,
        mode: str | None = None,
    ):
        _refresh_sqt(symbol)
        active_mode = (mode or default_mode).strip().lower() or "frequency"
        payload = _base_dashboard_payload(symbol, active_mode)
        payload["history_entry"] = record_simulation_history(payload, None)
        payload = _attach_v19_runtime(payload, _resolve_local_judge(payload))
        payload["kimi_judge"] = _cached_or_fallback_kimi(payload, active_mode, llm_provider, llm_model)
        payload["judge_comparison"] = _judge_comparison(payload["kimi_judge"], payload["local_judge"], payload.get("v19_runtime"))
        return payload

    @app.get('/api/chart/realtime')
    def realtime_chart(symbol: str = 'XAUUSD', bars: int = 240):
        return build_realtime_chart_payload(symbol, bars=max(60, min(int(bars), 720)))

    @app.get('/api/llm/health')
    def llm_health(provider: str = 'lm_studio', model: str | None = None):
        return sidecar_health(provider=provider, model=model)

    @app.get('/api/llm/kimi-live')
    def kimi_live(
        symbol: str = 'XAUUSD',
        llm_provider: str = 'nvidia_nim',
        llm_model: str | None = None,
        mode: str | None = None,
        force: bool = False,
    ):
        active_mode = (mode or default_mode).strip().lower() or "frequency"
        payload = _base_dashboard_payload(symbol, active_mode)
        payload = _attach_v19_runtime(payload, _resolve_local_judge(payload))
        payload["kimi_judge"] = _resolve_kimi_judge(payload, active_mode, llm_provider, llm_model, force=bool(force))
        payload["judge_comparison"] = _judge_comparison(payload["kimi_judge"], payload["local_judge"], payload.get("v19_runtime"))
        return {
            "symbol": payload.get("symbol", symbol),
            "mode": active_mode,
            "kimi_judge": payload["kimi_judge"],
            "local_judge": payload["local_judge"],
            "judge_comparison": payload["judge_comparison"],
            "v19_runtime": payload.get("v19_runtime", {}),
            "fallback_judge": fallback_kimi_judge(payload),
        }

    @app.get('/api/llm/judges-live')
    def judges_live(
        symbol: str = 'XAUUSD',
        llm_provider: str = 'nvidia_nim',
        llm_model: str | None = None,
        mode: str | None = None,
        force: bool = False,
    ):
        active_mode = (mode or default_mode).strip().lower() or "frequency"
        payload = _base_dashboard_payload(symbol, active_mode)
        payload = _attach_v19_runtime(payload, _resolve_local_judge(payload))
        payload["kimi_judge"] = _resolve_kimi_judge(payload, active_mode, llm_provider, llm_model, force=bool(force))
        payload["judge_comparison"] = _judge_comparison(payload["kimi_judge"], payload["local_judge"], payload.get("v19_runtime"))
        return {
            "symbol": payload.get("symbol", symbol),
            "mode": active_mode,
            "kimi_judge": payload["kimi_judge"],
            "local_judge": payload["local_judge"],
            "judge_comparison": payload["judge_comparison"],
            "v19_runtime": payload.get("v19_runtime", {}),
        }

    @app.get('/api/llm/context')
    def llm_context(symbol: str = 'XAUUSD', llm_provider: str = 'lm_studio', llm_model: str | None = None):
        payload = build_live_simulation(
            symbol,
            sequence_len=server.sequence_len,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        context = {
            'symbol': symbol,
            'market': payload.get('market', {}),
            'simulation': payload.get('simulation', {}),
            'macro': payload.get('feeds', {}).get('macro', {}),
            'news_headlines': [item.get('title', '') for item in payload.get('feeds', {}).get('news', [])[:5]],
            'crowd_items': [item.get('title', '') for item in payload.get('feeds', {}).get('public_discussions', [])[:5]],
        }
        return request_market_context(symbol, context, provider=llm_provider, model=llm_model)

    @app.get('/api/llm/kimi-log')
    def kimi_log(limit: int = 12):
        return {'entries': read_packet_log(limit=limit)}

    @app.get('/api/paper/state')
    def paper_state(symbol: str = 'XAUUSD'):
        return _paper_state(symbol)

    @app.post('/api/paper/open')
    def paper_open(request: PaperOpenRequest):
        try:
            position = paper_trader.open_position(
                symbol=request.symbol,
                direction=request.direction,
                entry_price=request.entry_price,
                confidence_tier=request.confidence_tier,
                sqt_label=request.sqt_label,
                mode=request.mode,
                leverage=request.leverage,
                stop_pips=request.stop_pips,
                take_profit_pips=request.take_profit_pips,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                manual_lot=request.manual_lot,
                note=request.note,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"opened": position, "paper_trading": _paper_state(request.symbol)}

    @app.post('/api/paper/modify')
    def paper_modify(request: PaperModifyRequest):
        try:
            updated = paper_trader.modify_position(
                request.trade_id,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"modified": True, "position": updated, "paper_trading": _paper_state(str(updated.get("symbol", "XAUUSD")))}

    @app.post('/api/paper/close')
    def paper_close(request: PaperCloseRequest):
        try:
            closed = paper_trader.close_position(request.trade_id, exit_price=request.exit_price)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"closed": closed, "paper_trading": _paper_state(str(closed.get("symbol", "XAUUSD")))}

    @app.post('/api/paper/reset')
    def paper_reset(request: PaperResetRequest):
        paper_trader.reset(starting_balance=request.starting_balance)
        return {"paper_trading": _paper_state()}

    @app.websocket('/ws/live')
    async def websocket_live(ws: WebSocket, symbol: str = 'XAUUSD'):
        await feed_manager.connect(ws, symbol=symbol)
        try:
            while True:
                message = (await ws.receive_text()).strip()
                if message and message.lower() != "ping":
                    feed_manager.set_symbol(ws, message)
        except WebSocketDisconnect:
            feed_manager.disconnect(ws)
        except Exception:
            feed_manager.disconnect(ws)

    @app.on_event("startup")
    async def startup_tasks():
        if getattr(app.state, "v18_feed_task", None) is None:
            app.state.v18_feed_task = asyncio.create_task(
                feed_manager.heartbeat_loop(
                    price_fn=_live_price,
                    sqt_fn=sqt_tracker.summary,
                )
            )

    @app.on_event("shutdown")
    async def shutdown_tasks():
        task = getattr(app.state, "v18_feed_task", None)
        if task is not None:
            task.cancel()
            app.state.v18_feed_task = None

    if not frontend_served:
        @app.get('/ui', response_class=HTMLResponse)
        def ui():
            return render_web_app_html()

    @app.get('/ui-legacy', response_class=HTMLResponse)
    def ui_legacy():
        return render_web_app_html()

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest):
        return server.predict(request.sequence)

    return app


if __name__ == '__main__':  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host=MODEL_SERVICE_HOST, port=MODEL_SERVICE_PORT)
