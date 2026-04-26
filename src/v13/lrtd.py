from __future__ import annotations

from collections import deque

import numpy as np


class LiveRegimeTransitionDetector:
    def __init__(self, transition_threshold: float = 0.30):
        self.threshold = float(transition_threshold)
        self.regime_history: deque[tuple[str, float]] = deque(maxlen=20)

    def update(self, regime: str, regime_confidence: float):
        self.regime_history.append((str(regime), float(regime_confidence)))

    def transition_risk(self) -> float:
        if len(self.regime_history) < 5:
            return 0.0
        recent_conf = [confidence for _, confidence in list(self.regime_history)[-5:]]
        conf_trend = float(np.polyfit(range(5), recent_conf, 1)[0])
        recent_regimes = [regime for regime, _ in list(self.regime_history)[-10:]]
        regime_flips = sum(1 for i in range(1, len(recent_regimes)) if recent_regimes[i] != recent_regimes[i - 1])
        current_conf = float(self.regime_history[-1][1])
        risk = 0.0
        if conf_trend < -0.02:
            risk += 0.30
        if current_conf < 0.65:
            risk += 0.25
        if regime_flips >= 2:
            risk += 0.25
        if regime_flips >= 4:
            risk += 0.20
        return float(min(1.0, risk))

    def should_suppress(self) -> bool:
        return self.transition_risk() >= self.threshold
