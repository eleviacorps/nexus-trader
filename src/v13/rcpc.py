from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from config.project_config import V13_RCPC_CALIBRATOR_PATH


REGIME_PRIOR_WIN_RATES = {
    'trending_up': 0.62,
    'trending_down': 0.59,
    'ranging': 0.51,
    'breakout': 0.55,
    'panic_shock': 0.48,
    'low_volatility': 0.52,
    'unknown': 0.50,
}

TRANSITION_THRESHOLD = 40


@dataclass
class RegimeConditionalPriorCalibrator:
    decay: float = 0.97
    _scores: list[float] = field(default_factory=list)
    _outcomes: list[float] = field(default_factory=list)
    _weights: list[float] = field(default_factory=list)
    _isotonic: IsotonicRegression | None = None
    _n_real: int = 0

    @property
    def uses_learned_calibration(self) -> bool:
        return self._n_real >= TRANSITION_THRESHOLD and self._isotonic is not None

    def calibrate(self, raw_score: float, regime: str) -> float:
        prior = float(REGIME_PRIOR_WIN_RATES.get(str(regime), REGIME_PRIOR_WIN_RATES['unknown']))
        clipped = float(np.clip(raw_score, 0.0, 1.0))
        if self.uses_learned_calibration:
            return float(np.clip(self._isotonic.predict([clipped])[0], 0.05, 0.95))
        deviation = (clipped - 0.5) * 2.0
        upper_limit = 0.85
        lower_limit = prior * 0.7
        calibrated = prior + deviation * (upper_limit - prior) * 0.5
        return float(np.clip(calibrated, lower_limit, upper_limit))

    def _fit_isotonic(self) -> None:
        if self._n_real < TRANSITION_THRESHOLD:
            return
        self._isotonic = IsotonicRegression(out_of_bounds='clip')
        self._isotonic.fit(
            np.asarray(self._scores, dtype=np.float64),
            np.asarray(self._outcomes, dtype=np.float64),
            sample_weight=np.asarray(self._weights, dtype=np.float64),
        )

    def record_outcome(self, raw_score: float, won: bool) -> None:
        self._scores.append(float(np.clip(raw_score, 0.0, 1.0)))
        self._outcomes.append(float(bool(won)))
        self._n_real += 1
        self._weights = [float(self.decay ** (self._n_real - i - 1)) for i in range(self._n_real)]
        if self._n_real >= TRANSITION_THRESHOLD:
            self._fit_isotonic()

    def calibration_error(self) -> float | None:
        if self._n_real == 0:
            return None
        calibrated = np.asarray([
            self.calibrate(score, 'unknown') for score in self._scores
        ], dtype=np.float64)
        outcomes = np.asarray(self._outcomes, dtype=np.float64)
        return float(np.mean(np.abs(calibrated - outcomes)))

    def save(self, path: Path = V13_RCPC_CALIBRATOR_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as handle:
            pickle.dump(self, handle)
        return path

    @classmethod
    def load(cls, path: Path = V13_RCPC_CALIBRATOR_PATH) -> 'RegimeConditionalPriorCalibrator':
        if not path.exists():
            return cls()
        with path.open('rb') as handle:
            return pickle.load(handle)

    def summary(self) -> dict[str, float | int | bool | None]:
        return {
            'real_trade_count': int(self._n_real),
            'uses_learned_calibration': bool(self.uses_learned_calibration),
            'calibration_error': self.calibration_error(),
            'transition_threshold': int(TRANSITION_THRESHOLD),
        }
