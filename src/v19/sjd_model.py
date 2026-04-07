from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config.project_config import V19_SJD_DATASET_PATH, V19_SJD_FEATURE_NAMES_PATH, V19_SJD_MODEL_NPZ_PATH, V19_SJD_MODEL_PATH
from src.v19.context_sampler import context_to_feature_vector

STANCE_LABELS: tuple[str, ...] = ("BUY", "SELL", "HOLD")
CONFIDENCE_LABELS: tuple[str, ...] = ("VERY_LOW", "LOW", "MODERATE", "HIGH")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _parse_feature_vector(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            payload = json.loads(value)
            if isinstance(payload, list):
                return [float(item) for item in payload]
        except Exception:
            return []
    return []


def _confidence_summary(confidence: str) -> str:
    mapping = {
        "HIGH": "high-confidence",
        "MODERATE": "moderate-confidence",
        "LOW": "low-confidence",
        "VERY_LOW": "very-low-confidence",
    }
    return mapping.get(confidence, "low-confidence")


class SJDDataset(Dataset[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    def __init__(
        self,
        features: np.ndarray,
        stances: np.ndarray,
        confidences: np.ndarray,
        level_offsets: np.ndarray,
        execute_mask: np.ndarray,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.stances = np.asarray(stances, dtype=np.int64)
        self.confidences = np.asarray(confidences, dtype=np.int64)
        self.level_offsets = np.asarray(level_offsets, dtype=np.float32)
        self.execute_mask = np.asarray(execute_mask, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.features))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.features[index],
            np.asarray(self.stances[index], dtype=np.int64),
            np.asarray(self.confidences[index], dtype=np.int64),
            self.level_offsets[index],
            np.asarray(self.execute_mask[index], dtype=np.float32),
        )


class JudgmentDistillationModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 512) -> None:
        super().__init__()
        mid = hidden // 2
        quarter = max(hidden // 4, 64)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(mid, quarter),
            nn.LayerNorm(quarter),
            nn.GELU(),
        )
        self.stance_head = nn.Linear(quarter, len(STANCE_LABELS))
        self.confidence_head = nn.Linear(quarter, len(CONFIDENCE_LABELS))
        self.level_head = nn.Linear(quarter, 3)
        self.execute_head = nn.Linear(quarter, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(features)
        return {
            "stance_logits": self.stance_head(hidden),
            "confidence_logits": self.confidence_head(hidden),
            "level_offsets": self.level_head(hidden),
            "execute_logit": self.execute_head(hidden),
        }


def sjd_loss(
    predictions: dict[str, torch.Tensor],
    stance_targets: torch.Tensor,
    confidence_targets: torch.Tensor,
    level_targets: torch.Tensor,
    execute_targets: torch.Tensor,
    *,
    stance_class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    stance_loss = F.cross_entropy(predictions["stance_logits"], stance_targets, weight=stance_class_weights)
    confidence_loss = F.cross_entropy(predictions["confidence_logits"], confidence_targets)
    level_loss = F.smooth_l1_loss(predictions["level_offsets"], level_targets)
    execute_loss = F.binary_cross_entropy_with_logits(predictions["execute_logit"].squeeze(-1), execute_targets)
    return (2.0 * stance_loss) + (1.0 * confidence_loss) + (1.5 * level_loss) + execute_loss


def _load_dataset_frame(
    dataset_path: Path = V19_SJD_DATASET_PATH,
    feature_names_path: Path = V19_SJD_FEATURE_NAMES_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(dataset_path)
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Feature names file is missing or empty.")
    return frame, [str(item) for item in feature_names]


def prepare_training_arrays(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> dict[str, np.ndarray]:
    features = np.asarray([_parse_feature_vector(value) for value in frame["feature_vector"].tolist()], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("Feature vectors do not align with the feature-name manifest.")
    stance_targets = np.asarray([STANCE_LABELS.index(str(item).strip().upper()) for item in frame["stance"].tolist()], dtype=np.int64)
    confidence_targets = np.asarray(
        [CONFIDENCE_LABELS.index(str(item).strip().upper()) if str(item).strip().upper() in CONFIDENCE_LABELS else 1 for item in frame["confidence"].tolist()],
        dtype=np.int64,
    )
    level_targets = np.asarray(
        frame.loc[:, ["entry_offset", "sl_offset", "tp_offset"]].fillna(0.0).to_numpy(dtype=np.float32),
        dtype=np.float32,
    )
    execute_targets = np.asarray((frame["stance"].astype(str).str.upper() != "HOLD").astype(float).to_numpy(dtype=np.float32), dtype=np.float32)
    return {
        "features": features,
        "stance_targets": stance_targets,
        "confidence_targets": confidence_targets,
        "level_targets": level_targets,
        "execute_targets": execute_targets,
    }


def _split_indices(frame: pd.DataFrame, validation_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    indices = np.arange(len(frame))
    rng.shuffle(indices)
    cutoff = max(1, int(len(indices) * (1.0 - float(validation_fraction))))
    return indices[:cutoff], indices[cutoff:]


def _dataset_loader(dataset: SJDDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _evaluate_bundle(
    model: JudgmentDistillationModel,
    dataset: SJDDataset,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: str,
) -> dict[str, float]:
    loader = _dataset_loader(dataset, batch_size=1024, shuffle=False)
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    stance_correct = 0
    total = 0
    tp_errors: list[float] = []
    sl_errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            features, stances, confidences, levels, execute_mask = batch
            x = (torch.as_tensor(features, dtype=torch.float32, device=target_device) - torch.as_tensor(mean, device=target_device)) / torch.as_tensor(std, device=target_device)
            outputs = model(x)
            predicted_stance = outputs["stance_logits"].argmax(dim=-1).detach().cpu().numpy()
            stance_correct += int(np.sum(predicted_stance == np.asarray(stances)))
            total += int(len(predicted_stance))
            predicted_levels = outputs["level_offsets"].detach().cpu().numpy()
            target_levels = np.asarray(levels)
            tp_errors.extend(np.abs(predicted_levels[:, 2] - target_levels[:, 2]).tolist())
            sl_errors.extend(np.abs(predicted_levels[:, 1] - target_levels[:, 1]).tolist())
    return {
        "stance_accuracy": float(stance_correct / max(total, 1)),
        "tp_mae": float(np.mean(tp_errors)) if tp_errors else 0.0,
        "sl_mae": float(np.mean(sl_errors)) if sl_errors else 0.0,
    }


def train_sjd_model(
    *,
    dataset_path: Path = V19_SJD_DATASET_PATH,
    feature_names_path: Path = V19_SJD_FEATURE_NAMES_PATH,
    checkpoint_path: Path = V19_SJD_MODEL_PATH,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 3e-4,
    device: str | None = None,
) -> dict[str, Any]:
    frame, feature_names = _load_dataset_frame(dataset_path=dataset_path, feature_names_path=feature_names_path)
    arrays = prepare_training_arrays(frame, feature_names)
    train_idx, valid_idx = _split_indices(frame)
    x_train = arrays["features"][train_idx]
    x_valid = arrays["features"][valid_idx]
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    train_dataset = SJDDataset(
        features=((x_train - mean) / std).astype(np.float32),
        stances=arrays["stance_targets"][train_idx],
        confidences=arrays["confidence_targets"][train_idx],
        level_offsets=arrays["level_targets"][train_idx],
        execute_mask=arrays["execute_targets"][train_idx],
    )
    valid_dataset = SJDDataset(
        features=((x_valid - mean) / std).astype(np.float32),
        stances=arrays["stance_targets"][valid_idx],
        confidences=arrays["confidence_targets"][valid_idx],
        level_offsets=arrays["level_targets"][valid_idx],
        execute_mask=arrays["execute_targets"][valid_idx],
    )

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = JudgmentDistillationModel(input_dim=len(feature_names)).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = _dataset_loader(train_dataset, batch_size=batch_size, shuffle=True)

    class_counts = np.bincount(arrays["stance_targets"][train_idx], minlength=len(STANCE_LABELS)).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    class_weights = torch.as_tensor(class_counts.max() / class_counts, dtype=torch.float32, device=target_device)

    best_metrics: dict[str, float] | None = None
    best_payload: dict[str, Any] | None = None
    loss_history: list[float] = []
    valid_history: list[dict[str, float]] = []

    for _ in range(max(int(epochs), 1)):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch in train_loader:
            features, stances, confidences, levels, execute_mask = batch
            x = torch.as_tensor(features, dtype=torch.float32, device=target_device)
            stance_targets = torch.as_tensor(stances, dtype=torch.long, device=target_device)
            confidence_targets = torch.as_tensor(confidences, dtype=torch.long, device=target_device)
            level_targets = torch.as_tensor(levels, dtype=torch.float32, device=target_device)
            execute_targets = torch.as_tensor(execute_mask, dtype=torch.float32, device=target_device)
            outputs = model(x)
            loss = sjd_loss(
                outputs,
                stance_targets,
                confidence_targets,
                level_targets,
                execute_targets,
                stance_class_weights=class_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
        loss_history.append(epoch_loss / max(steps, 1))
        metrics = _evaluate_bundle(model, valid_dataset, mean, std, device=str(target_device))
        valid_history.append(metrics)
        composite = metrics["stance_accuracy"] - (0.002 * (metrics["tp_mae"] + metrics["sl_mae"]))
        if best_metrics is None or composite > (best_metrics["stance_accuracy"] - (0.002 * (best_metrics["tp_mae"] + best_metrics["sl_mae"]))):
            best_metrics = metrics
            best_payload = {
                "state_dict": model.state_dict(),
                "input_dim": len(feature_names),
                "feature_names": list(feature_names),
                "feature_mean": mean.tolist(),
                "feature_std": std.tolist(),
                "stance_labels": list(STANCE_LABELS),
                "confidence_labels": list(CONFIDENCE_LABELS),
                "validation_metrics": metrics,
                "loss_history": loss_history,
                "validation_history": valid_history,
            }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if best_payload is None:
        raise RuntimeError("SJD training did not produce a valid checkpoint.")
    torch.save(best_payload, checkpoint_path)
    export_sjd_npz(checkpoint_path=checkpoint_path, output_path=V19_SJD_MODEL_NPZ_PATH)
    return {
        "checkpoint_path": str(checkpoint_path),
        "validation_metrics": best_metrics or {},
        "train_size": int(len(train_idx)),
        "valid_size": int(len(valid_idx)),
        "feature_count": int(len(feature_names)),
    }


@dataclass
class SjdBundle:
    model: JudgmentDistillationModel
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    stance_labels: list[str]
    confidence_labels: list[str]
    device: str


@dataclass
class NumpySjdBundle:
    weights: dict[str, np.ndarray]
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    stance_labels: list[str]
    confidence_labels: list[str]


def _gelu(value: np.ndarray) -> np.ndarray:
    return 0.5 * value * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value + 0.044715 * np.power(value, 3))))


def _layer_norm(value: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = value.mean(axis=-1, keepdims=True)
    variance = np.mean(np.square(value - mean), axis=-1, keepdims=True)
    normalized = (value - mean) / np.sqrt(variance + eps)
    return (normalized * weight) + bias


def load_sjd_bundle(
    path: Path = V19_SJD_MODEL_PATH,
    *,
    device: str | None = None,
) -> SjdBundle:
    payload = torch.load(path, map_location=device or "cpu")
    model = JudgmentDistillationModel(int(payload["input_dim"]))
    model.load_state_dict(payload["state_dict"])
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(target_device)
    model.eval()
    return SjdBundle(
        model=model,
        feature_names=[str(item) for item in payload["feature_names"]],
        mean=np.asarray(payload["feature_mean"], dtype=np.float32),
        std=np.asarray(payload["feature_std"], dtype=np.float32),
        stance_labels=[str(item) for item in payload.get("stance_labels", STANCE_LABELS)],
        confidence_labels=[str(item) for item in payload.get("confidence_labels", CONFIDENCE_LABELS)],
        device=target_device,
    )


def export_sjd_npz(
    *,
    checkpoint_path: Path = V19_SJD_MODEL_PATH,
    output_path: Path = V19_SJD_MODEL_NPZ_PATH,
) -> Path:
    payload = torch.load(checkpoint_path, map_location="cpu")
    arrays = {key: value.detach().cpu().numpy() for key, value in payload["state_dict"].items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        feature_names=np.asarray(payload["feature_names"], dtype=object),
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(payload["feature_std"], dtype=np.float32),
        stance_labels=np.asarray(payload.get("stance_labels", STANCE_LABELS), dtype=object),
        confidence_labels=np.asarray(payload.get("confidence_labels", CONFIDENCE_LABELS), dtype=object),
    )
    return output_path


def load_sjd_npz_bundle(path: Path = V19_SJD_MODEL_NPZ_PATH) -> NumpySjdBundle:
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


def predict_sjd_from_context(
    bundle: SjdBundle,
    context: Mapping[str, Any],
    *,
    symbol: str = "XAUUSD",
    pip_size: float = 0.1,
) -> dict[str, Any]:
    vector, _ = context_to_feature_vector(context, feature_names=bundle.feature_names)
    normalized = ((vector - bundle.mean) / bundle.std).astype(np.float32)
    x = torch.as_tensor(normalized[None, :], dtype=torch.float32, device=bundle.device)
    with torch.no_grad():
        outputs = bundle.model(x)
    return _decode_prediction_outputs(
        stance_labels=bundle.stance_labels,
        confidence_labels=bundle.confidence_labels,
        stance_logits=outputs["stance_logits"].detach().cpu().numpy()[0],
        confidence_logits=outputs["confidence_logits"].detach().cpu().numpy()[0],
        offsets=outputs["level_offsets"].detach().cpu().numpy()[0],
        context=context,
        symbol=symbol,
        pip_size=pip_size,
    )


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
