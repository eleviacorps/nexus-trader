"""EV/dispersion-based decision engine for hybrid V27 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Decision = Literal["BUY", "SELL", "HOLD"]


@dataclass
class HybridDecision:
    """Decision payload returned by the hybrid logic."""

    decision: Decision
    confidence: float
    ev: float
    std: float
    skew: float
    regime: str
    positive_ratio: float
    negative_ratio: float


class HybridDecisionEngine:
    """Compute EV decision + dispersion-aware confidence."""

    def __init__(self, ev_threshold: float = 0.0002) -> None:
        self.ev_threshold = float(ev_threshold)

    @staticmethod
    def _normalized_confidence(ev: float, std: float) -> float:
        score = abs(float(ev)) / (float(std) + 1e-6)
        # score in [0, inf) -> confidence in [0, 1)
        return float(score / (1.0 + score))

    @staticmethod
    def _skew(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < 3:
            return 0.0
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-12:
            return 0.0
        centered = arr - mean
        return float(np.mean((centered / std) ** 3))

    def evaluate(self, paths: np.ndarray, entry_price: float, regime: str = "UNKNOWN") -> HybridDecision:
        """Evaluate BUY/SELL/HOLD using expected return and dispersion."""
        arr = np.asarray(paths, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2 or entry_price <= 0:
            return HybridDecision(
                decision="HOLD",
                confidence=0.0,
                ev=0.0,
                std=0.0,
                skew=0.0,
                regime=str(regime),
                positive_ratio=0.0,
                negative_ratio=0.0,
            )

        final_prices = arr[:, -1]
        returns = (final_prices - float(entry_price)) / float(entry_price)
        ev = float(np.mean(returns))
        std = float(np.std(returns))
        skew = self._skew(returns)
        positive_ratio = float(np.mean(returns > 0))
        negative_ratio = float(np.mean(returns < 0))

        if ev > self.ev_threshold:
            decision: Decision = "BUY"
        elif ev < -self.ev_threshold:
            decision = "SELL"
        else:
            decision = "HOLD"

        confidence = self._normalized_confidence(ev, std)
        if decision == "HOLD":
            # Preserve uncertainty semantics; HOLD confidence should not look overconfident.
            confidence *= 0.7

        return HybridDecision(
            decision=decision,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            ev=ev,
            std=std,
            skew=skew,
            regime=str(regime),
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
        )
