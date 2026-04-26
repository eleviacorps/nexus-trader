from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REGIME_BUCKETS = ("trend_up", "trend_down", "range", "macro_shock", "unknown")


def _normalize_regime(raw: Any) -> str:
    text = str(raw or "unknown").strip().lower()
    mapping = {
        "breakout": "macro_shock",
        "panic": "macro_shock",
        "shock": "macro_shock",
        "trend_continuation": "trend_up",
        "mean_reversion": "range",
    }
    if text in REGIME_BUCKETS:
        return text
    return mapping.get(text, "unknown")


@dataclass(frozen=True)
class RegimeRankerPrediction:
    regime: str
    probability: float


class _MiniLogit:
    def __init__(self, feature_names: Sequence[str], learning_rate: float = 0.08, epochs: int = 260):
        self.feature_names = list(feature_names)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -20.0, 20.0)))

    def fit(self, frame: pd.DataFrame, target_col: str) -> None:
        x = frame.loc[:, self.feature_names].to_numpy(dtype=np.float64)
        y = pd.to_numeric(frame[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0) + 1e-6
        x_norm = (x - self.mean_) / self.std_
        self.weights_ = np.zeros(x_norm.shape[1], dtype=np.float64)
        self.bias_ = 0.0
        for _ in range(self.epochs):
            logits = x_norm @ self.weights_ + self.bias_
            probs = self._sigmoid(logits)
            error = probs - y
            grad_w = (x_norm.T @ error) / max(1, x_norm.shape[0])
            grad_b = float(np.mean(error))
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

    def predict_proba(self, row: Mapping[str, Any]) -> float:
        if self.weights_ is None or self.mean_ is None or self.std_ is None:
            return 0.5
        values = np.asarray([float(row.get(name, 0.0) or 0.0) for name in self.feature_names], dtype=np.float64)
        normalized = (values - self.mean_) / self.std_
        logit = float(normalized @ self.weights_ + self.bias_)
        return float(self._sigmoid(np.asarray([logit], dtype=np.float64))[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "mean": [float(value) for value in (self.mean_ if self.mean_ is not None else [])],
            "std": [float(value) for value in (self.std_ if self.std_ is not None else [])],
            "weights": [float(value) for value in (self.weights_ if self.weights_ is not None else [])],
            "bias": float(self.bias_),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_MiniLogit":
        model = cls(
            feature_names=list(payload.get("feature_names", [])),
            learning_rate=float(payload.get("learning_rate", 0.08)),
            epochs=int(payload.get("epochs", 260)),
        )
        mean = np.asarray(payload.get("mean", []), dtype=np.float64)
        if mean.size == 0:
            return model
        std = np.asarray(payload.get("std", []), dtype=np.float64)
        weights = np.asarray(payload.get("weights", []), dtype=np.float64)
        model.mean_ = mean
        model.std_ = std if std.size == mean.size else np.ones_like(mean)
        model.weights_ = weights if weights.size == mean.size else np.zeros_like(mean)
        model.bias_ = float(payload.get("bias", 0.0))
        return model


class RegimeSpecificBranchRanker:
    FEATURE_NAMES = [
        "cabr_score",
        "analog_similarity",
        "volatility_realism",
        "hmm_regime_match",
        "branch_disagreement",
        "news_consistency",
        "crowd_consistency",
        "branch_volatility",
        "branch_move_zscore_abs",
    ]

    def __init__(self):
        self.models: dict[str, _MiniLogit] = {}

    @staticmethod
    def _prep_frame(frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        if "branch_move_zscore_abs" not in working.columns:
            working["branch_move_zscore_abs"] = pd.to_numeric(working.get("branch_move_zscore"), errors="coerce").fillna(0.0).abs()
        for feature in RegimeSpecificBranchRanker.FEATURE_NAMES:
            working[feature] = pd.to_numeric(working.get(feature, 0.0), errors="coerce").fillna(0.0)
        working["regime_bucket"] = working.get("regime_bucket", working.get("dominant_regime", "unknown")).map(_normalize_regime)
        working["label_hit"] = pd.to_numeric(working.get("label_hit"), errors="coerce").fillna(0.0)
        return working

    def fit(self, frame: pd.DataFrame) -> dict[str, Any]:
        working = self._prep_frame(frame)
        summary: dict[str, Any] = {}
        for regime in REGIME_BUCKETS:
            subset = working.loc[working["regime_bucket"] == regime].copy()
            if subset.empty:
                continue
            model = _MiniLogit(self.FEATURE_NAMES)
            model.fit(subset, target_col="label_hit")
            self.models[regime] = model
            preds = np.asarray([model.predict_proba(row) for row in subset.to_dict(orient="records")], dtype=np.float64)
            summary[regime] = {
                "rows": int(len(subset)),
                "mean_prediction": float(np.mean(preds)),
                "mean_label": float(np.mean(subset["label_hit"])),
            }
        return summary

    def predict(self, row: Mapping[str, Any]) -> RegimeRankerPrediction:
        regime = _normalize_regime(row.get("regime_bucket", row.get("dominant_regime", "unknown")))
        model = self.models.get(regime) or self.models.get("unknown")
        if model is None:
            return RegimeRankerPrediction(regime=regime, probability=0.5)
        prepared = {name: float(row.get(name, 0.0) or 0.0) for name in self.FEATURE_NAMES}
        probability = float(np.clip(model.predict_proba(prepared), 0.0, 1.0))
        return RegimeRankerPrediction(regime=regime, probability=probability)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": list(self.FEATURE_NAMES),
            "models": {regime: model.to_dict() for regime, model in self.models.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "RegimeSpecificBranchRanker":
        payload = json.loads(path.read_text(encoding="utf-8"))
        ranker = cls()
        for regime, model_payload in dict(payload.get("models", {})).items():
            ranker.models[str(regime)] = _MiniLogit.from_dict(model_payload)
        return ranker
