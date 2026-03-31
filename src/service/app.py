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
from src.service.live_data import build_live_simulation
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
        config = NexusTFTConfig(
            input_dim=int(config_payload.get('input_dim', self.feature_dim)),
            hidden_dim=int(config_payload.get('hidden_dim', 128)),
            lstm_layers=int(config_payload.get('lstm_layers', 2)),
            dropout=float(config_payload.get('dropout', 0.1)),
        )
        self.model = NexusTFT(config).to(self.device)
        load_checkpoint_with_expansion(self.model, _resolve_checkpoint(), new_input_dim=config.input_dim)
        self.model.eval()

    def predict(self, sequence: list[list[float]]) -> PredictResponse:
        validate_sequence_shape(sequence, self.sequence_len, self.feature_dim)
        with torch.no_grad():
            tensor = torch.tensor([sequence], dtype=torch.float32, device=self.device)
            probability = self.model(tensor).detach().cpu().item()
        bullish_probability = float(probability)
        return PredictResponse(
            bullish_probability=bullish_probability,
            bearish_probability=float(1.0 - bullish_probability),
            signal=classify_probability(bullish_probability, self.threshold),
            threshold=self.threshold,
            sequence_len=self.sequence_len,
            feature_dim=self.feature_dim,
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
    def simulate_live(symbol: str = 'XAUUSD'):
        payload = build_live_simulation(symbol, sequence_len=server.sequence_len)
        sequence = payload.pop('sequence', None)
        if sequence:
            try:
                payload['model_prediction'] = server.predict_dict(sequence)
            except Exception as exc:  # pragma: no cover
                payload['model_prediction'] = {'error': str(exc)}
        else:
            payload['model_prediction'] = None
        return payload

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
