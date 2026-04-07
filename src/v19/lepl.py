from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from config.project_config import V19_LEPL_MODEL_PATH

LEPL_FEATURES = [
    "sjd_stance_buy",
    "sjd_stance_sell",
    "sjd_confidence",
    "sqt_label",
    "cabr_score",
    "hurst_asymmetry",
    "mfg_disagreement",
    "cpm_score",
    "has_open_position",
    "open_position_pnl",
]

LEPL_ACTIONS = ("ENTER", "HOLD", "CLOSE", "NOTHING")


def _confidence_to_scalar(raw: Any) -> float:
    mapping = {"VERY_LOW": 0.0, "LOW": 1.0, "MODERATE": 2.0, "HIGH": 3.0}
    return float(mapping.get(str(raw or "LOW").strip().upper(), 1.0))


def _sqt_to_scalar(raw: Any) -> float:
    mapping = {"COLD": 0.0, "NEUTRAL": 1.0, "GOOD": 2.0, "HOT": 3.0}
    return float(mapping.get(str(raw or "NEUTRAL").strip().upper(), 1.0))


class LiveExecutionPolicy:
    def __init__(self) -> None:
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LiveExecutionPolicy":
        self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        return self

    def predict(self, features: dict[str, Any]) -> str:
        vector = self._features_to_vector(features)
        output = str(self.model.predict([vector])[0])
        return output if output in LEPL_ACTIONS else "NOTHING"

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        vector = self._features_to_vector(features)
        probabilities = self.model.predict_proba([vector])[0]
        return {str(label): float(prob) for label, prob in zip(self.model.classes_, probabilities, strict=False)}

    def _features_to_vector(self, features: dict[str, Any]) -> np.ndarray:
        stance = str(features.get("sjd_stance", "HOLD")).strip().upper()
        return np.asarray(
            [
                1.0 if stance == "BUY" else 0.0,
                1.0 if stance == "SELL" else 0.0,
                _confidence_to_scalar(features.get("sjd_confidence", "LOW")),
                _sqt_to_scalar(features.get("sqt_label", "NEUTRAL")),
                float(features.get("cabr_score", 0.5) or 0.5),
                float(features.get("hurst_asymmetry", 0.0) or 0.0),
                float(features.get("mfg_disagreement", 0.0) or 0.0),
                float(features.get("cpm_score", 0.5) or 0.5),
                1.0 if bool(features.get("has_open_position")) else 0.0,
                float(features.get("open_position_pnl", 0.0) or 0.0),
            ],
            dtype=np.float32,
        )

    def save(self, path: Path = V19_LEPL_MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.model, handle)
        return path

    @classmethod
    def load(cls, path: Path = V19_LEPL_MODEL_PATH) -> "LiveExecutionPolicy":
        policy = cls()
        with path.open("rb") as handle:
            policy.model = pickle.load(handle)
        return policy
