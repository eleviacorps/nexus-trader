from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
from src.service.llm_sidecar import read_packet_log, request_market_context, sidecar_health
from src.service.live_data import build_live_monitor, build_live_simulation, fetch_recent_market_candles, record_simulation_history
from src.ui.web import render_web_app_html
from src.utils.device import get_torch_device
from src.v13.s3pta import PaperTradeAccumulator
from src.v16.csl import build_v16_simulation_result
from src.v16.paper import PaperTradingEngine
from src.v16.sqt import SimulationQualityTracker

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    from fastapi.responses import HTMLResponse  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
except ImportError:  # pragma: no cover
    FastAPI = None
    HTTPException = RuntimeError  # type: ignore
    BaseModel = object  # type: ignore
    HTMLResponse = str  # type: ignore

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
    note: str = ""


class PaperCloseRequest(BaseModel):  # type: ignore[misc]
    trade_id: str
    exit_price: float


class PaperResetRequest(BaseModel):  # type: ignore[misc]
    starting_balance: float = 1000.0


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
    default_mode = os.getenv("NEXUS_V16_MODE", "frequency").strip().lower() or "frequency"
    s3pta_enabled = os.getenv("NEXUS_S3PTA_ENABLED", "0") == "1"
    paper_trade_accumulator = PaperTradeAccumulator() if s3pta_enabled else None
    sqt_tracker = SimulationQualityTracker()
    scored_predictions: set[str] = set()
    paper_trader = PaperTradingEngine(starting_balance=float(os.getenv("NEXUS_PAPER_START_BALANCE", "1000")))

    def _current_prices(symbols: set[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for symbol_name in sorted({str(item).upper() for item in symbols if item}):
            try:
                candles = fetch_recent_market_candles(symbol_name)
                if candles.empty:
                    continue
                prices[symbol_name] = float(candles.iloc[-1]["close"])
            except Exception:
                continue
        return prices

    def _paper_state(symbol: str | None = None) -> dict[str, Any]:
        preview = paper_trader.state()
        open_symbols = {str(position.get("symbol", "")).upper() for position in preview.get("open_positions", [])}
        if symbol:
            open_symbols.add(str(symbol).upper())
        return paper_trader.state(current_prices=_current_prices(open_symbols))

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

    @app.get('/api/llm/health')
    def llm_health(provider: str = 'lm_studio', model: str | None = None):
        return sidecar_health(provider=provider, model=model)

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
                note=request.note,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"opened": position, "paper_trading": _paper_state(request.symbol)}

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

    @app.get('/ui', response_class=HTMLResponse)
    def ui():
        return render_web_app_html()

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest):
        return server.predict(request.sequence)

    return app


if __name__ == '__main__':  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host=MODEL_SERVICE_HOST, port=MODEL_SERVICE_PORT)
