from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from config.project_config import V19_SJD_MODEL_NPZ_PATH
from src.v19.context_sampler import context_to_feature_vector


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _confidence_summary(confidence: str) -> str:
    mapping = {
        "HIGH": "high-confidence",
        "MODERATE": "moderate-confidence",
        "LOW": "low-confidence",
        "VERY_LOW": "very-low-confidence",
    }
    return mapping.get(confidence, "low-confidence")


def _gelu(value: np.ndarray) -> np.ndarray:
    return 0.5 * value * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value + 0.044715 * np.power(value, 3))))


def _layer_norm(value: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = value.mean(axis=-1, keepdims=True)
    variance = np.mean(np.square(value - mean), axis=-1, keepdims=True)
    normalized = (value - mean) / np.sqrt(variance + eps)
    return (normalized * weight) + bias


@dataclass
class NumpySjdBundle:
    weights: dict[str, np.ndarray]
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    stance_labels: list[str]
    confidence_labels: list[str]


def load_sjd_npz_bundle(path=V19_SJD_MODEL_NPZ_PATH) -> NumpySjdBundle:
    payload = np.load(path, allow_pickle=True)
    special = {"feature_names", "feature_mean", "feature_std", "stance_labels", "confidence_labels"}
    weights = {key: np.asarray(payload[key]) for key in payload.files if key not in special}
    return NumpySjdBundle(
        weights=weights,
        feature_names=[str(item) for item in payload["feature_names"].tolist()],
        mean=np.asarray(payload["feature_mean"], dtype=np.float32),
        std=np.asarray(payload["feature_std"], dtype=np.float32),
        stance_labels=[str(item) for item in payload["stance_labels"].tolist()],
        confidence_labels=[str(item) for item in payload["confidence_labels"].tolist()],
    )


def _decode_prediction_outputs(
    *,
    stance_labels: Sequence[str],
    confidence_labels: Sequence[str],
    stance_logits: np.ndarray,
    confidence_logits: np.ndarray,
    offsets: np.ndarray,
    context: Mapping[str, Any],
    symbol: str,
    pip_size: float,
) -> dict[str, Any]:
    stance = str(stance_labels[int(np.argmax(stance_logits))])
    confidence = str(confidence_labels[int(np.argmax(confidence_logits))])
    current_price = _safe_float((context.get("market") or {}).get("current_price"), 0.0)
    sqt_label = str((context.get("sqt") or {}).get("label", "NEUTRAL")).strip().upper()
    cabr_score = _safe_float((context.get("simulation") or {}).get("cabr_score"), 0.0)
    cpm_score = _safe_float((context.get("simulation") or {}).get("cpm_score"), 0.0)
    hurst = _safe_float((context.get("simulation") or {}).get("hurst_asymmetry"), 0.0)
    direction = stance
    if sqt_label == "COLD":
        direction = "HOLD"
        confidence = "VERY_LOW"
    if direction == "HOLD" or current_price <= 0.0:
        final_call = "SKIP"
        entry_zone: list[float] = []
        stop_loss = None
        take_profit = None
    else:
        entry_center = current_price + (_safe_float(offsets[0]) * pip_size)
        entry_zone = [round(entry_center - (2.0 * pip_size), 5), round(entry_center + (2.0 * pip_size), 5)]
        stop_loss = round(current_price + (_safe_float(offsets[1]) * pip_size), 5)
        take_profit = round(current_price + (_safe_float(offsets[2]) * pip_size), 5)
        if direction == "BUY":
            stop_loss = min(stop_loss, round(current_price - (5.0 * pip_size), 5))
            take_profit = max(take_profit, round(current_price + (6.0 * pip_size), 5))
        else:
            stop_loss = max(stop_loss, round(current_price + (5.0 * pip_size), 5))
            take_profit = min(take_profit, round(current_price - (6.0 * pip_size), 5))
        final_call = direction
    summary = f"{final_call} - local SJD issues a {_confidence_summary(confidence)} {final_call.lower()} read." if final_call != "SKIP" else "SKIP - local SJD abstains for this bar."
    reasoning = (
        f"Local SJD derived {direction} from CABR {cabr_score:.1%}, CPM {cpm_score:.1%}, and Hurst asymmetry {hurst:.3f}."
        if final_call != "SKIP"
        else f"Local SJD abstained because stance is HOLD or SQT is {sqt_label}."
    )
    return {
        "stance": direction,
        "confidence": confidence,
        "final_call": final_call,
        "final_summary": summary,
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "hold_time": "current_bar" if final_call != "SKIP" else "skip",
        "market_only_summary": {
            "call": final_call,
            "summary": f"Local market-only proxy is {final_call}.",
            "reasoning": f"Current price is {current_price:.2f} and the local distilled proxy uses the live market context already embedded in the feature vector.",
        },
        "v18_summary": {
            "call": final_call,
            "summary": f"Local V19 student read is {final_call}.",
            "reasoning": reasoning,
        },
        "combined_summary": {
            "call": final_call,
            "summary": f"Combined local distilled decision is {final_call}.",
            "reasoning": reasoning,
        },
        "reasoning": reasoning,
        "key_risk": "Local SJD is a distilled approximation of historical NIM judgments and should be monitored for drift.",
        "crowd_note": "Crowd and persona signals are included through the V19 context feature vector.",
        "regime_note": f"SQT is {sqt_label} and the current symbol is {symbol}.",
        "invalidation": stop_loss,
    }


def predict_sjd_from_context_numpy(
    bundle: NumpySjdBundle,
    context: Mapping[str, Any],
    *,
    symbol: str = "XAUUSD",
    pip_size: float = 0.1,
) -> dict[str, Any]:
    vector, _ = context_to_feature_vector(context, feature_names=bundle.feature_names)
    x = ((vector - bundle.mean) / bundle.std).astype(np.float32)[None, :]
    weights = bundle.weights
    x = _gelu(_layer_norm((x @ weights["encoder.0.weight"].T) + weights["encoder.0.bias"], weights["encoder.1.weight"], weights["encoder.1.bias"]))
    x = _gelu(_layer_norm((x @ weights["encoder.4.weight"].T) + weights["encoder.4.bias"], weights["encoder.5.weight"], weights["encoder.5.bias"]))
    x = _gelu(_layer_norm((x @ weights["encoder.8.weight"].T) + weights["encoder.8.bias"], weights["encoder.9.weight"], weights["encoder.9.bias"]))
    stance_logits = (x @ weights["stance_head.weight"].T) + weights["stance_head.bias"]
    confidence_logits = (x @ weights["confidence_head.weight"].T) + weights["confidence_head.bias"]
    offsets = (x @ weights["level_head.weight"].T) + weights["level_head.bias"]
    return _decode_prediction_outputs(
        stance_labels=bundle.stance_labels,
        confidence_labels=bundle.confidence_labels,
        stance_logits=stance_logits[0],
        confidence_logits=confidence_logits[0],
        offsets=offsets[0],
        context=context,
        symbol=symbol,
        pip_size=pip_size,
    )
