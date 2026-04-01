from __future__ import annotations

import json
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
from src.service.llm_sidecar import request_market_context, sidecar_health
from src.service.live_data import build_live_monitor, build_live_simulation, record_simulation_history, _build_final_forecast
from src.ui.web import render_web_app_html
from src.utils.device import get_torch_device

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

try:
    from fastapi import FastAPI  # type: ignore
    from fastapi.responses import HTMLResponse  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
except ImportError:  # pragma: no cover
    FastAPI = None
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

    branch_weight = 0.31 + 0.14 * max(branch_consensus, branch_confidence)
    analog_weight = 0.10 + 0.08 * analog_confidence
    model_weight = 0.20 + 0.07 * (1.0 - abs(model_probability - 0.5))
    bot_weight = 0.22 + 0.10 * bot_confidence
    llm_weight = max(0.06, 1.0 - branch_weight - analog_weight - model_weight - bot_weight)
    weight_sum = branch_weight + analog_weight + model_weight + bot_weight + llm_weight
    weights = {
        'branch': round(branch_weight / weight_sum, 6),
        'analog': round(analog_weight / weight_sum, 6),
        'model': round(model_weight / weight_sum, 6),
        'bot_swarm': round(bot_weight / weight_sum, 6),
        'llm': round(llm_weight / weight_sum, 6),
    }
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
        'horizon_predictions': horizon_predictions,
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
        )
        self.model = NexusTFT(config).to(self.device)
        load_checkpoint_with_expansion(self.model, _resolve_checkpoint(), new_input_dim=config.input_dim)
        self.model.eval()

    def predict(self, sequence: list[list[float]]) -> PredictResponse:
        validate_sequence_shape(sequence, self.sequence_len, self.feature_dim)
        with torch.no_grad():
            tensor = torch.tensor([sequence], dtype=torch.float32, device=self.device)
            raw_output = self.model(tensor).detach().cpu().numpy()
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
        response = self.predict(sequence)
        if hasattr(response, "model_dump"):
            return response.model_dump()  # type: ignore[no-any-return]
        return {
            "bullish_probability": float(response.bullish_probability),
            "bearish_probability": float(response.bearish_probability),
            "signal": str(response.signal),
            "threshold": float(response.threshold),
            "sequence_len": int(response.sequence_len),
            "feature_dim": int(response.feature_dim),
            "horizon_probabilities": dict(response.horizon_probabilities or {}),
        }


def create_app() -> Any:
    if FastAPI is None:
        raise ImportError('fastapi and pydantic are required to create the inference app.')

    server = ModelServer()
    app = FastAPI(title='Nexus Trader Inference API', version='0.2.0')

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
    def simulate_live(symbol: str = 'XAUUSD', llm_provider: str = 'lm_studio'):
        payload = build_live_simulation(symbol, sequence_len=server.sequence_len, llm_provider=llm_provider)
        sequence = payload.pop('sequence', None)
        if sequence:
            try:
                payload['model_prediction'] = server.predict_dict(sequence)
            except Exception as exc:  # pragma: no cover
                payload['model_prediction'] = {'error': str(exc)}
        else:
            payload['model_prediction'] = None
        payload['final_forecast'] = _build_final_forecast(
            payload,
            payload.get('bot_swarm', {}),
            ((payload.get('llm_context') or {}).get('content') or {}),
            payload.get('model_prediction'),
        )
        payload['ensemble_prediction'] = build_ensemble_prediction(payload, payload.get('model_prediction'))
        payload['history_entry'] = record_simulation_history(payload, payload.get('model_prediction'))
        return payload

    @app.get('/api/live-monitor')
    def live_monitor(symbol: str = 'XAUUSD'):
        return build_live_monitor(symbol)

    @app.get('/api/llm/health')
    def llm_health(provider: str = 'lm_studio'):
        return sidecar_health(provider=provider)

    @app.get('/api/llm/context')
    def llm_context(symbol: str = 'XAUUSD', llm_provider: str = 'lm_studio'):
        payload = build_live_simulation(symbol, sequence_len=server.sequence_len, llm_provider=llm_provider)
        context = {
            'symbol': symbol,
            'market': payload.get('market', {}),
            'simulation': payload.get('simulation', {}),
            'macro': payload.get('feeds', {}).get('macro', {}),
            'news_headlines': [item.get('title', '') for item in payload.get('feeds', {}).get('news', [])[:5]],
            'crowd_items': [item.get('title', '') for item in payload.get('feeds', {}).get('public_discussions', [])[:5]],
        }
        return request_market_context(symbol, context, provider=llm_provider)

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
