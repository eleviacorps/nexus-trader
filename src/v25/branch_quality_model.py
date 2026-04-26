from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.v25.branch_sequence_encoder import BranchSequenceEncoder


@dataclass(frozen=True)
class BranchQualityPrediction:
    branch_realism_score: float
    probability_branch_hits_target: float
    probability_branch_is_invalidated: float
    quality_score: float
    blended_rank_score: float


class _BinaryLogit:
    def __init__(self, feature_names: Sequence[str], learning_rate: float = 0.08, epochs: int = 220) -> None:
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

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("BinaryLogit.fit expects a non-empty 2D feature matrix.")
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0) + 1e-6
        x_norm = (x - self.mean_) / self.std_
        self.weights_ = np.zeros(x_norm.shape[1], dtype=np.float64)
        self.bias_ = 0.0

        for _ in range(self.epochs):
            logits = x_norm @ self.weights_ + self.bias_
            probs = self._sigmoid(logits)
            error = probs - y
            grad_w = (x_norm.T @ error) / x_norm.shape[0]
            grad_b = float(np.mean(error))
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("BinaryLogit model is not fit.")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_norm = (x - self.mean_) / self.std_
        logits = x_norm @ self.weights_ + self.bias_
        return self._sigmoid(logits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "mean": [float(v) for v in (self.mean_ if self.mean_ is not None else [])],
            "std": [float(v) for v in (self.std_ if self.std_ is not None else [])],
            "weights": [float(v) for v in (self.weights_ if self.weights_ is not None else [])],
            "bias": float(self.bias_),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_BinaryLogit":
        model = cls(
            feature_names=list(payload.get("feature_names", [])),
            learning_rate=float(payload.get("learning_rate", 0.08)),
            epochs=int(payload.get("epochs", 220)),
        )
        mean = np.asarray(payload.get("mean", []), dtype=np.float64)
        std = np.asarray(payload.get("std", []), dtype=np.float64)
        weights = np.asarray(payload.get("weights", []), dtype=np.float64)
        if mean.size > 0:
            model.mean_ = mean
            model.std_ = std if std.size == mean.size else np.ones_like(mean)
            model.weights_ = weights if weights.size == mean.size else np.zeros_like(mean)
            model.bias_ = float(payload.get("bias", 0.0))
        return model


class BranchQualityModel:
    """
    V25 branch-quality model:
      - branch_realism_score
      - probability_branch_hits_target
      - probability_branch_is_invalidated

    Ranking blend:
      0.50 * CABR + 0.30 * branch_quality + 0.20 * historical_analog_fit
    """

    FEATURE_NAMES = [
        "branch_volatility",
        "branch_acceleration",
        "regime_consistency",
        "analog_similarity",
        "specialist_bot_agreement",
        "cabr_score",
        "minority_disagreement",
        "historical_outcome_fit",
        "path_trend",
    ]

    def __init__(self, encoder: BranchSequenceEncoder | None = None) -> None:
        self.encoder = encoder or BranchSequenceEncoder()
        self.realism_model = _BinaryLogit(self.FEATURE_NAMES)
        self.hit_model = _BinaryLogit(self.FEATURE_NAMES)
        self.invalidation_model = _BinaryLogit(self.FEATURE_NAMES)
        self.fitted = False

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _extract_feature_row(self, sample: Mapping[str, Any]) -> dict[str, float]:
        path = sample.get("path", sample.get("branch_path", []))
        encoded = self.encoder.encode_path(path if isinstance(path, Sequence) else [])
        row = {
            "branch_volatility": float(np.clip(self._safe_float(sample.get("branch_volatility"), encoded.volatility), 0.0, 5.0)),
            "branch_acceleration": float(np.clip(self._safe_float(sample.get("branch_acceleration"), encoded.acceleration), 0.0, 5.0)),
            "regime_consistency": float(np.clip(self._safe_float(sample.get("regime_consistency"), 0.5), 0.0, 1.0)),
            "analog_similarity": float(np.clip(self._safe_float(sample.get("analog_similarity"), 0.5), 0.0, 1.0)),
            "specialist_bot_agreement": float(np.clip(self._safe_float(sample.get("specialist_bot_agreement"), 0.5), 0.0, 1.0)),
            "cabr_score": float(np.clip(self._safe_float(sample.get("cabr_score"), 0.5), 0.0, 1.0)),
            "minority_disagreement": float(np.clip(self._safe_float(sample.get("minority_disagreement"), 0.5), 0.0, 1.0)),
            "historical_outcome_fit": float(np.clip(self._safe_float(sample.get("historical_outcome_fit"), 0.5), 0.0, 1.0)),
            "path_trend": float(np.clip(encoded.trend, -1.0, 1.0)),
        }
        return row

    def _feature_matrix(self, samples: Iterable[Mapping[str, Any]]) -> np.ndarray:
        rows = []
        for sample in samples:
            values = self._extract_feature_row(sample)
            rows.append([values[name] for name in self.FEATURE_NAMES])
        if not rows:
            return np.zeros((0, len(self.FEATURE_NAMES)), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    def fit(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        x = self._feature_matrix(samples)
        if x.shape[0] == 0:
            raise ValueError("No samples provided to BranchQualityModel.fit.")
        realism_target = np.asarray([float(self._safe_float(item.get("label_realism"), 0.0)) for item in samples], dtype=np.float64)
        hit_target = np.asarray([float(self._safe_float(item.get("label_hits_target"), 0.0)) for item in samples], dtype=np.float64)
        invalid_target = np.asarray([float(self._safe_float(item.get("label_invalidated"), 0.0)) for item in samples], dtype=np.float64)

        self.realism_model.fit(x, realism_target)
        self.hit_model.fit(x, hit_target)
        self.invalidation_model.fit(x, invalid_target)
        self.fitted = True

        realism_probs = self.realism_model.predict_proba(x)
        hit_probs = self.hit_model.predict_proba(x)
        invalid_probs = self.invalidation_model.predict_proba(x)
        return {
            "realism_mean_prob": float(np.mean(realism_probs)),
            "hit_mean_prob": float(np.mean(hit_probs)),
            "invalid_mean_prob": float(np.mean(invalid_probs)),
        }

    @staticmethod
    def blended_rank_score(cabr_score: float, branch_quality_score: float, historical_analog_fit: float) -> float:
        return float(
            (0.50 * float(np.clip(cabr_score, 0.0, 1.0)))
            + (0.30 * float(np.clip(branch_quality_score, 0.0, 1.0)))
            + (0.20 * float(np.clip(historical_analog_fit, 0.0, 1.0)))
        )

    def predict(self, sample: Mapping[str, Any]) -> BranchQualityPrediction:
        if not self.fitted:
            raise RuntimeError("BranchQualityModel is not fit.")
        row = self._extract_feature_row(sample)
        x = np.asarray([[row[name] for name in self.FEATURE_NAMES]], dtype=np.float64)
        realism = float(self.realism_model.predict_proba(x)[0])
        hit = float(self.hit_model.predict_proba(x)[0])
        invalid = float(self.invalidation_model.predict_proba(x)[0])
        quality = float(np.clip((0.60 * realism) + (0.40 * hit) - (0.35 * invalid), 0.0, 1.0))
        blended = self.blended_rank_score(
            cabr_score=row["cabr_score"],
            branch_quality_score=quality,
            historical_analog_fit=row["historical_outcome_fit"],
        )
        return BranchQualityPrediction(
            branch_realism_score=realism,
            probability_branch_hits_target=hit,
            probability_branch_is_invalidated=invalid,
            quality_score=quality,
            blended_rank_score=blended,
        )

    def rank_branches(self, branches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for branch in branches:
            prediction = self.predict(branch)
            row = dict(branch)
            row["branch_realism_score"] = prediction.branch_realism_score
            row["probability_branch_hits_target"] = prediction.probability_branch_hits_target
            row["probability_branch_is_invalidated"] = prediction.probability_branch_is_invalidated
            row["branch_quality_score"] = prediction.quality_score
            row["blended_rank_score"] = prediction.blended_rank_score
            ranked.append(row)
        return sorted(ranked, key=lambda item: float(item.get("blended_rank_score", 0.0)), reverse=True)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": list(self.FEATURE_NAMES),
            "realism_model": self.realism_model.to_dict(),
            "hit_model": self.hit_model.to_dict(),
            "invalidation_model": self.invalidation_model.to_dict(),
            "fitted": bool(self.fitted),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "BranchQualityModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls()
        model.realism_model = _BinaryLogit.from_dict(payload.get("realism_model", {}))
        model.hit_model = _BinaryLogit.from_dict(payload.get("hit_model", {}))
        model.invalidation_model = _BinaryLogit.from_dict(payload.get("invalidation_model", {}))
        model.fitted = bool(payload.get("fitted", False))
        return model

