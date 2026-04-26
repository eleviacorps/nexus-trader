from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class _TradeLogit:
    def __init__(self, feature_names: Sequence[str], learning_rate: float = 0.07, epochs: int = 240):
        self.feature_names = list(feature_names)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
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

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("TradeLogit model is not fit.")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_norm = (x - self.mean_) / self.std_
        return self._sigmoid(x_norm @ self.weights_ + self.bias_)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "mean": [float(v) for v in (self.mean_ if self.mean_ is not None else [])],
            "std": [float(v) for v in (self.std_ if self.std_ is not None else [])],
            "weights": [float(v) for v in (self.weights_ if self.weights_ is not None else [])],
            "bias": float(self.bias_),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_TradeLogit":
        model = cls(payload.get("feature_names", []), payload.get("learning_rate", 0.07), payload.get("epochs", 240))
        mean = np.asarray(payload.get("mean", []), dtype=np.float64)
        std = np.asarray(payload.get("std", []), dtype=np.float64)
        weights = np.asarray(payload.get("weights", []), dtype=np.float64)
        if mean.size > 0:
            model.mean_ = mean
            model.std_ = std if std.size == mean.size else np.ones_like(mean)
            model.weights_ = weights if weights.size == mean.size else np.zeros_like(mean)
            model.bias_ = float(payload.get("bias", 0.0))
        return model


@dataclass(frozen=True)
class TradeabilityDecision:
    take_trade_probability: float
    execute: bool
    reason: str


class TradeabilityModel:
    FEATURE_NAMES = [
        "admission_score",
        "regime_code",
        "direction_code",
        "spread",
        "slippage",
        "cabr_score",
        "branch_quality",
        "claude_confidence",
        "recent_streak",
        "cluster_count",
    ]

    def __init__(self, threshold: float = 0.62):
        self.threshold = float(threshold)
        self.model = _TradeLogit(self.FEATURE_NAMES)
        self.fitted = False

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _regime_code(regime: str) -> float:
        mapping = {
            "trend_up": 1.0,
            "trend_down": -1.0,
            "breakout": 0.5,
            "range": 0.2,
            "unknown": 0.0,
            "chop": -0.2,
        }
        return float(mapping.get(str(regime).lower(), 0.0))

    @staticmethod
    def _direction_code(direction: str) -> float:
        direction = str(direction).upper()
        if direction == "BUY":
            return 1.0
        if direction == "SELL":
            return -1.0
        return 0.0

    def _feature_row(self, item: Mapping[str, Any]) -> list[float]:
        return [
            float(np.clip(self._safe_float(item.get("admission_score"), 0.0), 0.0, 1.0)),
            self._regime_code(str(item.get("regime", "unknown"))),
            self._direction_code(str(item.get("direction", "HOLD"))),
            float(np.clip(self._safe_float(item.get("spread"), 0.0), 0.0, 2.0)),
            float(np.clip(self._safe_float(item.get("slippage"), 0.0), 0.0, 2.0)),
            float(np.clip(self._safe_float(item.get("cabr_score"), 0.5), 0.0, 1.0)),
            float(np.clip(self._safe_float(item.get("branch_quality"), 0.5), 0.0, 1.0)),
            float(np.clip(self._safe_float(item.get("claude_confidence"), 0.0), 0.0, 1.0)),
            float(np.clip(self._safe_float(item.get("recent_streak"), 0.0), -10.0, 10.0)),
            float(np.clip(self._safe_float(item.get("cluster_count"), 0.0), 0.0, 25.0)),
        ]

    def _feature_matrix(self, items: Iterable[Mapping[str, Any]]) -> np.ndarray:
        rows = [self._feature_row(item) for item in items]
        if not rows:
            return np.zeros((0, len(self.FEATURE_NAMES)), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def fit(self, items: Sequence[Mapping[str, Any]], labels: Sequence[float]) -> dict[str, float]:
        x = self._feature_matrix(items)
        y = np.asarray(labels, dtype=np.float64)
        if x.shape[0] == 0 or y.size == 0:
            raise ValueError("TradeabilityModel.fit requires non-empty training data.")
        self.model.fit(x, y)
        self.fitted = True
        probs = self.model.predict_proba(x)
        return {"mean_probability": float(np.mean(probs)), "mean_label": float(np.mean(y))}

    def predict_probability(self, item: Mapping[str, Any]) -> float:
        if not self.fitted:
            raise RuntimeError("TradeabilityModel is not fit.")
        x = np.asarray([self._feature_row(item)], dtype=np.float64)
        return float(self.model.predict_proba(x)[0])

    def final_execution_gate(
        self,
        *,
        item: Mapping[str, Any],
        claude_approve: bool,
        admission_score: float,
        threshold: float,
    ) -> TradeabilityDecision:
        probability = self.predict_probability(item)
        if probability <= self.threshold:
            return TradeabilityDecision(probability, False, "tradeability_probability_below_threshold")
        if not bool(claude_approve):
            return TradeabilityDecision(probability, False, "claude_rejected")
        if float(admission_score) <= float(threshold):
            return TradeabilityDecision(probability, False, "admission_below_regime_threshold")
        return TradeabilityDecision(probability, True, "tradeability_gate_pass")

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "threshold": self.threshold,
                    "fitted": self.fitted,
                    "model": self.model.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "TradeabilityModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(threshold=float(payload.get("threshold", 0.62)))
        model.model = _TradeLogit.from_dict(payload.get("model", {}))
        model.fitted = bool(payload.get("fitted", False))
        return model

