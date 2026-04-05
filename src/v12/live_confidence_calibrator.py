from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from config.project_config import V12_CONFIDENCE_CALIBRATOR_PATH


@dataclass
class LiveConfidenceCalibrator:
    decay_factor: float = 0.98
    min_samples: int = 30
    calibrator: IsotonicRegression = field(default_factory=lambda: IsotonicRegression(out_of_bounds="clip"))
    _raw_scores: list[float] = field(default_factory=list)
    _outcomes: list[float] = field(default_factory=list)
    _weights: list[float] = field(default_factory=list)
    _fitted: bool = False

    def record_outcome(self, raw_score: float, won: bool) -> None:
        self._raw_scores.append(float(raw_score))
        self._outcomes.append(float(bool(won)))
        self._weights = [
            float(self.decay_factor ** (len(self._raw_scores) - index - 1))
            for index in range(len(self._raw_scores))
        ]
        if len(self._raw_scores) >= int(self.min_samples):
            self.calibrator.fit(
                np.asarray(self._raw_scores, dtype=np.float64),
                np.asarray(self._outcomes, dtype=np.float64),
                sample_weight=np.asarray(self._weights, dtype=np.float64),
            )
            self._fitted = True

    def calibrate(self, raw_score: float) -> float:
        if not self._fitted:
            return 0.5
        return float(self.calibrator.predict([float(raw_score)])[0])

    def calibration_error(self) -> float:
        if not self._fitted or not self._raw_scores:
            return 0.5
        calibrated = self.calibrator.predict(np.asarray(self._raw_scores, dtype=np.float64))
        outcomes = np.asarray(self._outcomes, dtype=np.float64)
        return float(np.mean(np.abs(calibrated - outcomes)))

    def save(self, path: Path = V12_CONFIDENCE_CALIBRATOR_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return path

    @classmethod
    def load(cls, path: Path = V12_CONFIDENCE_CALIBRATOR_PATH) -> "LiveConfidenceCalibrator":
        if not path.exists():
            return cls()
        with path.open("rb") as handle:
            return pickle.load(handle)
