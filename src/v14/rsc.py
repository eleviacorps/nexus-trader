from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from config.project_config import V14_RSC_CALIBRATOR_PATH


REGIME_PRIOR_WIN_RATES = {
    "trending_up": 0.62,
    "trending_down": 0.59,
    "ranging": 0.51,
    "breakout": 0.55,
    "panic_shock": 0.48,
    "low_volatility": 0.52,
    "unknown": 0.50,
}

MIN_TRADES_PER_REGIME = 20


class RegimeStratifiedCalibrator:
    def __init__(self, decay: float = 0.97):
        self.decay = float(decay)
        self._records: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._calibrators: dict[str, IsotonicRegression] = {}
        self._counts: dict[str, int] = defaultdict(int)

    def record_outcome(self, raw_score: float, regime: str, won: bool) -> None:
        regime_name = str(regime or "unknown")
        self._records[regime_name].append((float(raw_score), float(won)))
        self._counts[regime_name] += 1
        n = self._counts[regime_name]
        if n >= MIN_TRADES_PER_REGIME:
            scores = [row[0] for row in self._records[regime_name]]
            outcomes = [row[1] for row in self._records[regime_name]]
            weights = [self.decay ** (n - idx - 1) for idx in range(n)]
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(scores, outcomes, sample_weight=weights)
            self._calibrators[regime_name] = calibrator

    def calibrate(self, raw_score: float, regime: str) -> float:
        regime_name = str(regime or "unknown")
        if regime_name in self._calibrators:
            return float(np.clip(self._calibrators[regime_name].predict([float(raw_score)])[0], 0.05, 0.95))
        prior = float(REGIME_PRIOR_WIN_RATES.get(regime_name, REGIME_PRIOR_WIN_RATES["unknown"]))
        deviation = (float(raw_score) - 0.5) * 2.0
        return float(np.clip(prior + deviation * 0.30, 0.10, 0.90))

    def calibration_error_per_regime(self) -> dict[str, float]:
        errors: dict[str, float] = {}
        for regime, records in self._records.items():
            if len(records) < 5:
                continue
            cal_scores = [self.calibrate(score, regime) for score, _ in records]
            outcomes = [outcome for _, outcome in records]
            errors[regime] = float(np.mean(np.abs(np.asarray(cal_scores) - np.asarray(outcomes))))
        return errors

    def summary(self) -> dict[str, object]:
        errors = self.calibration_error_per_regime()
        return {
            "counts_per_regime": {regime: int(count) for regime, count in self._counts.items()},
            "learned_regimes": sorted(self._calibrators.keys()),
            "calibration_error_per_regime": {regime: float(value) for regime, value in errors.items()},
            "max_calibration_error": float(max(errors.values())) if errors else None,
            "decay": float(self.decay),
        }

    def save(self, path: Path = V14_RSC_CALIBRATOR_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return path

    @classmethod
    def load(cls, path: Path = V14_RSC_CALIBRATOR_PATH) -> "RegimeStratifiedCalibrator":
        if not path.exists():
            return cls()
        with path.open("rb") as handle:
            return pickle.load(handle)
