from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
except ImportError:  # pragma: no cover
    HistGradientBoostingClassifier = None


class ContradictionType(str, Enum):
    FULL_AGREEMENT_BULL = "agreement_bull"
    FULL_AGREEMENT_BEAR = "agreement_bear"
    SHORT_TERM_CONTRARY = "short_term_contrary"
    LONG_TERM_CONTRARY = "long_term_contrary"
    FULL_DISAGREEMENT = "full_disagreement"


@dataclass(frozen=True)
class ContradictionAssessment:
    contradiction_type: ContradictionType
    confidence: float
    cone_treatment: str


def _direction_label(probability: float, threshold: float = 0.53) -> int:
    if probability >= threshold:
        return 1
    if probability <= (1.0 - threshold):
        return -1
    return 0


def classify_contradiction(
    *,
    prob_5m: float,
    prob_15m: float,
    prob_30m: float,
    conf_5m: float = 0.0,
    conf_15m: float = 0.0,
    conf_30m: float = 0.0,
) -> ContradictionAssessment:
    dir_5 = _direction_label(prob_5m)
    dir_15 = _direction_label(prob_15m)
    dir_30 = _direction_label(prob_30m)
    confidence = float(np.clip((conf_5m + conf_15m + conf_30m) / 3.0, 0.0, 1.0))
    if dir_5 > 0 and dir_15 > 0 and dir_30 > 0:
        return ContradictionAssessment(ContradictionType.FULL_AGREEMENT_BULL, confidence, "normal_cone")
    if dir_5 < 0 and dir_15 < 0 and dir_30 < 0:
        return ContradictionAssessment(ContradictionType.FULL_AGREEMENT_BEAR, confidence, "normal_cone")
    if dir_15 == dir_30 != 0 and dir_5 not in {0, dir_15}:
        return ContradictionAssessment(ContradictionType.SHORT_TERM_CONTRARY, confidence, "flag_liquidity_sweep")
    if dir_5 == dir_15 != 0 and dir_30 not in {0, dir_5}:
        return ContradictionAssessment(ContradictionType.LONG_TERM_CONTRARY, confidence, "flag_fake_breakout")
    return ContradictionAssessment(ContradictionType.FULL_DISAGREEMENT, confidence, "widen_cone")


def contradiction_feature_vector(
    probabilities: Sequence[float],
    confidences: Sequence[float],
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    confs = np.asarray(confidences, dtype=np.float32)
    dirs = np.asarray([_direction_label(float(prob)) for prob in probs], dtype=np.float32)
    return np.asarray(
        [
            *probs.tolist(),
            *confs.tolist(),
            float(np.mean(dirs)),
            float(np.std(dirs)),
            float(np.mean(confs)),
            float(np.std(probs)),
        ],
        dtype=np.float32,
    )


def train_contradiction_detector(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if HistGradientBoostingClassifier is None or len(np.unique(labels)) < 2:
        return {"available": False, "provider": "rule_based"}
    model = HistGradientBoostingClassifier(max_depth=4, max_iter=160, learning_rate=0.05, random_state=42)
    model.fit(features, labels)
    return {"available": True, "provider": "hist_gradient_boosting", "model": model}


def apply_contradiction_detector(model_payload: Mapping[str, Any] | None, features: np.ndarray) -> np.ndarray | None:
    if not model_payload or not model_payload.get("available", False):
        return None
    model = model_payload.get("model")
    if model is None:
        return None
    return np.asarray(model.predict_proba(features), dtype=np.float32)


def save_contradiction_detector(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dict(payload), handle)


def load_contradiction_detector(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if isinstance(payload, dict) else None
