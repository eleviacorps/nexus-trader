from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from config.project_config import V20_HMM_MODEL_PATH, V21_BIMAMBA_MODEL_PATH, V21_HMM_MODEL_PATH, V21_XLSTM_MODEL_PATH
from src.v20.feature_builder import build_v20_feature_frame
from src.v20.regime_detector import RegimeDetector
from src.v21.xlstm_backbone import NexusXLSTM


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


def _normalize_features(frame: pd.DataFrame, feature_columns: list[str], feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    numeric = frame.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    aligned = numeric.reindex(columns=feature_columns, fill_value=0.0).to_numpy(dtype=np.float32)
    return np.clip((aligned - feature_mean) / feature_std, -8.0, 8.0)


@lru_cache(maxsize=1)
def _load_xlstm_checkpoint() -> dict[str, Any] | None:
    if not V21_XLSTM_MODEL_PATH.exists():
        return None
    try:
        return torch.load(V21_XLSTM_MODEL_PATH, map_location="cpu")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_bimamba_checkpoint() -> dict[str, Any] | None:
    if not V21_BIMAMBA_MODEL_PATH.exists():
        return None
    try:
        return torch.load(V21_BIMAMBA_MODEL_PATH, map_location="cpu")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_hmm_detector() -> RegimeDetector | None:
    try:
        for candidate in (V21_HMM_MODEL_PATH, V20_HMM_MODEL_PATH):
            if candidate.exists():
                return RegimeDetector.load(candidate)
        return None
    except Exception:
        return None


def run_v21_xlstm_inference(payload: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _load_xlstm_checkpoint()
    if checkpoint is None:
        return {"available": False, "error": "missing_v21_xlstm_checkpoint"}
    feature_columns = list(checkpoint.get("feature_columns") or [])
    sequence_len = int(checkpoint.get("sequence_len", 120))
    feature_mean = np.asarray(checkpoint.get("feature_mean") or [0.0] * len(feature_columns), dtype=np.float32)
    feature_std = np.asarray(checkpoint.get("feature_std") or [1.0] * len(feature_columns), dtype=np.float32)
    d_model = int(checkpoint.get("d_model", 192))
    n_layers = int(checkpoint.get("n_layers", 3))

    price_frame = _price_frame_from_payload(payload)
    if price_frame.empty or len(price_frame) < 60:
        return {"available": False, "error": "insufficient_live_bars"}
    feature_frame, _metadata = build_v20_feature_frame(price_frame, hmm_detector=_load_hmm_detector())
    feature_frame = feature_frame.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    if len(feature_frame) < sequence_len:
        return {"available": False, "error": "insufficient_v21_feature_rows"}

    normalized = _normalize_features(feature_frame.tail(sequence_len), feature_columns, feature_mean, feature_std)
    regime_ids = (
        pd.to_numeric(feature_frame.tail(sequence_len).get("hmm_state"), errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=5)
        .astype(np.int64)
        .to_numpy()
    )
    model = NexusXLSTM(n_features=len(feature_columns), d_model=d_model, n_layers=n_layers, n_regimes=6)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        outputs = model(
            torch.tensor(normalized[None, ...], dtype=torch.float32),
            torch.tensor(regime_ids[None, ...], dtype=torch.long),
        )
    dir15_prob = float(torch.sigmoid(outputs["dir_15m"]).item())
    dir30_prob = float(torch.sigmoid(outputs["dir_30m"]).item())
    regime_probs = torch.softmax(outputs["regime"], dim=-1)[0].cpu().numpy().astype(float).tolist()
    vol_probs = torch.softmax(outputs["vol_env"], dim=-1)[0].cpu().numpy().astype(float).tolist()
    range_probs = torch.softmax(outputs["range"], dim=-1)[0].cpu().numpy().astype(float).tolist()
    vsn_weights = outputs["vsn_weights"][0].cpu().numpy().astype(float)
    order = np.argsort(vsn_weights)[::-1][:10]
    return {
        "available": True,
        "dir_15m_prob": round(dir15_prob, 6),
        "dir_30m_prob": round(dir30_prob, 6),
        "regime_probs": [round(float(value), 6) for value in regime_probs],
        "vol_probs": [round(float(value), 6) for value in vol_probs],
        "range_probs": [round(float(value), 6) for value in range_probs],
        "hidden": outputs["hidden"][0].cpu().numpy().astype(float).tolist(),
        "top_vsn_features": [
            {"feature": feature_columns[int(index)], "weight": round(float(vsn_weights[int(index)]), 6)}
            for index in order.tolist()
        ],
    }


def run_v21_bimamba_inference(payload: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _load_bimamba_checkpoint()
    if checkpoint is None:
        return {"available": False, "error": "missing_v21_bimamba_checkpoint"}
    feature_columns = list(checkpoint.get("feature_columns") or [])
    sequence_len = int(checkpoint.get("sequence_len", 120))
    feature_mean = np.asarray(checkpoint.get("feature_mean") or [0.0] * len(feature_columns), dtype=np.float32)
    feature_std = np.asarray(checkpoint.get("feature_std") or [1.0] * len(feature_columns), dtype=np.float32)

    price_frame = _price_frame_from_payload(payload)
    if price_frame.empty or len(price_frame) < 60:
        return {"available": False, "error": "insufficient_live_bars"}
    feature_frame, _metadata = build_v20_feature_frame(price_frame, hmm_detector=_load_hmm_detector())
    feature_frame = feature_frame.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    if len(feature_frame) < sequence_len:
        return {"available": False, "error": "insufficient_v21_feature_rows"}

    normalized = _normalize_features(feature_frame.tail(sequence_len), feature_columns, feature_mean, feature_std)
    from src.v20.mamba_backbone import NexusBiMamba

    model = NexusBiMamba(n_features=len(feature_columns), sequence_len=sequence_len)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(normalized[None, ...], dtype=torch.float32))
    dir15_prob = float(torch.sigmoid(outputs["dir_15m"]).item())
    return {"available": True, "dir_15m_prob": round(dir15_prob, 6)}


__all__ = ["run_v21_xlstm_inference", "run_v21_bimamba_inference"]
