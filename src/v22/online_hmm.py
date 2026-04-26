from __future__ import annotations

import math
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from config.project_config import V20_HMM_MODEL_PATH
from src.v20.regime_detector import DEFAULT_STATE_NAMES, RegimeDetector


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return float(default)
        return float(number)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class OnlineRegimeSnapshot:
    regime_index: int
    regime_label: str
    regime_confidence: float
    posterior: tuple[float, ...]
    uncertain: bool
    persistence_count: int
    lot_size_multiplier: float
    low_confidence_flag: bool
    persistence_conflict: bool
    reasons: tuple[str, ...]


class OnlineHMMRegimeDetector:
    """
    Streaming regime detector with the V22 confidence and persistence guards.
    """

    CONFIDENCE_THRESHOLD = 0.60
    PERSISTENCE_BARS_THRESHOLD = 12

    def __init__(
        self,
        base_hmm_path: str | Path = V20_HMM_MODEL_PATH,
        *,
        momentum: float = 0.95,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.base_hmm_path = Path(base_hmm_path)
        self.momentum = float(np.clip(momentum, 0.0, 1.0))
        self.confidence_threshold = float(confidence_threshold)
        detector = self._load_detector(self.base_hmm_path)
        self.model = detector.model
        self.state_names = dict(detector.state_names or DEFAULT_STATE_NAMES)
        self.feature_columns = tuple(detector.feature_columns or ())
        self.pi = np.asarray(getattr(self.model, "startprob_", np.full(6, 1.0 / 6.0)), dtype=np.float64)
        self.A = np.asarray(getattr(self.model, "transmat_", np.eye(len(self.pi), dtype=np.float64)), dtype=np.float64)
        self.alpha: np.ndarray | None = None
        self.posterior_history: deque[np.ndarray] = deque(maxlen=512)
        self.last_regime: int | None = None
        self.persistence_count = 0
        self.regime_anchor_price: float | None = None

    def _load_detector(self, path: Path) -> RegimeDetector:
        if not path.exists():
            raise FileNotFoundError(f"Missing base HMM checkpoint: {path}")
        try:
            return RegimeDetector.load(path)
        except Exception:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            model = payload["model"] if isinstance(payload, Mapping) and "model" in payload else payload
            state_names = dict(payload.get("state_names", DEFAULT_STATE_NAMES)) if isinstance(payload, Mapping) else DEFAULT_STATE_NAMES
            feature_columns = tuple(payload.get("feature_columns", ())) if isinstance(payload, Mapping) else ()
            return RegimeDetector(model=model, state_names=state_names, feature_columns=feature_columns)

    def _vectorize_observation(self, observation: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
        if isinstance(observation, Mapping):
            values = [_safe_float(observation.get(column), 0.0) for column in self.feature_columns]
            return np.asarray(values, dtype=np.float64)
        values = [_safe_float(item, 0.0) for item in observation]
        return np.asarray(values, dtype=np.float64)

    def _emission_prob(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float64).reshape(1, -1)
        log_likelihood = np.asarray(self.model._compute_log_likelihood(vector)[0], dtype=np.float64)  # type: ignore[attr-defined]
        centered = log_likelihood - np.max(log_likelihood)
        probs = np.exp(centered)
        total = probs.sum()
        if total <= 0.0:
            return np.full_like(probs, 1.0 / max(len(probs), 1), dtype=np.float64)
        return probs / total

    def update(self, observation: Mapping[str, Any] | Sequence[float], *, price: float | None = None) -> np.ndarray:
        emission = self._emission_prob(self._vectorize_observation(observation))
        if self.alpha is None:
            posterior = self.pi * emission
        else:
            blended = (self.alpha @ self.A) * emission
            posterior = (self.momentum * blended) + ((1.0 - self.momentum) * emission)
        total = float(posterior.sum())
        posterior = posterior / total if total > 0.0 else np.full_like(posterior, 1.0 / max(len(posterior), 1), dtype=np.float64)
        self.alpha = posterior
        regime = self.current_regime
        if self.last_regime is None or regime != self.last_regime:
            self.persistence_count = 1
            self.regime_anchor_price = float(price) if price is not None else self.regime_anchor_price
        else:
            self.persistence_count += 1
        self.last_regime = regime
        if price is not None and self.regime_anchor_price is None:
            self.regime_anchor_price = float(price)
        self.posterior_history.append(posterior.copy())
        return posterior

    @property
    def current_regime(self) -> int:
        if self.alpha is None:
            return 0
        return int(np.argmax(self.alpha))

    @property
    def regime_label(self) -> str:
        return str(self.state_names.get(self.current_regime, f"state_{self.current_regime}"))

    @property
    def regime_confidence(self) -> float:
        if self.alpha is None:
            return 0.0
        return float(np.max(self.alpha))

    @property
    def posterior(self) -> tuple[float, ...]:
        if self.alpha is None:
            return tuple()
        return tuple(round(float(item), 6) for item in self.alpha.tolist())

    @property
    def uncertain(self) -> bool:
        return self.regime_confidence < float(self.confidence_threshold)

    def runtime_flags(
        self,
        *,
        direction: str | None = None,
        current_price: float | None = None,
        atr_14: float | None = None,
    ) -> OnlineRegimeSnapshot:
        direction_value = str(direction or "HOLD").upper()
        atr_value = max(_safe_float(atr_14, 0.0), 1e-6)
        price_now = _safe_float(current_price, 0.0)
        anchor = _safe_float(self.regime_anchor_price, price_now)
        persistence_conflict = False
        if self.persistence_count > self.PERSISTENCE_BARS_THRESHOLD and direction_value in {"BUY", "SELL"}:
            confirmation_move = price_now - anchor
            if direction_value == "BUY":
                persistence_conflict = confirmation_move < atr_value
            else:
                persistence_conflict = (-confirmation_move) < atr_value
        reasons: list[str] = []
        lot_size_multiplier = 1.0
        low_confidence_flag = False
        if self.uncertain:
            reasons.append("uncertain_regime")
            lot_size_multiplier *= 0.50
            low_confidence_flag = True
        if persistence_conflict:
            reasons.append("regime_conflict")
            lot_size_multiplier = min(lot_size_multiplier, 0.25)
        return OnlineRegimeSnapshot(
            regime_index=self.current_regime,
            regime_label=self.regime_label,
            regime_confidence=round(self.regime_confidence, 6),
            posterior=self.posterior,
            uncertain=bool(self.uncertain),
            persistence_count=int(self.persistence_count),
            lot_size_multiplier=round(float(lot_size_multiplier), 6),
            low_confidence_flag=bool(low_confidence_flag),
            persistence_conflict=bool(persistence_conflict),
            reasons=tuple(reasons),
        )

    def summary(
        self,
        *,
        direction: str | None = None,
        current_price: float | None = None,
        atr_14: float | None = None,
    ) -> dict[str, Any]:
        snapshot = self.runtime_flags(direction=direction, current_price=current_price, atr_14=atr_14)
        return {
            "regime_index": snapshot.regime_index,
            "regime_label": snapshot.regime_label,
            "regime_confidence": snapshot.regime_confidence,
            "posterior": list(snapshot.posterior),
            "uncertain": snapshot.uncertain,
            "persistence_count": snapshot.persistence_count,
            "lot_size_multiplier": snapshot.lot_size_multiplier,
            "low_confidence_flag": snapshot.low_confidence_flag,
            "persistence_conflict": snapshot.persistence_conflict,
            "reasons": list(snapshot.reasons),
        }


def calibrate_confidence_threshold(
    confidences: Sequence[float] | np.ndarray,
    *,
    quantile: float = 0.20,
    minimum: float = 0.52,
    maximum: float = 0.62,
) -> float:
    values = np.asarray([_safe_float(item, 0.0) for item in confidences], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(np.clip(OnlineHMMRegimeDetector.CONFIDENCE_THRESHOLD, minimum, maximum))
    level = float(np.quantile(values, np.clip(float(quantile), 0.01, 0.99)))
    return float(np.clip(level, minimum, maximum))


__all__ = ["OnlineHMMRegimeDetector", "OnlineRegimeSnapshot", "calibrate_confidence_threshold"]
