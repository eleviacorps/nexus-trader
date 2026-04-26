from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class ConformalCone:
    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = float(alpha)
        self.regime_residuals: dict[int, list[float]] = {i: [] for i in range(6)}
        self.regime_q_hat: dict[int, float] = {i: float("inf") for i in range(6)}

    def calibrate(self, predicted_paths, realized_paths, regime_labels) -> None:
        self.regime_residuals = {i: [] for i in range(6)}
        for pred, actual, regime in zip(predicted_paths, realized_paths, regime_labels, strict=False):
            residual = float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(pred, dtype=np.float64))))
            self.regime_residuals[int(regime)].append(residual)
        for regime, residuals in self.regime_residuals.items():
            n = len(residuals)
            if n > 10:
                q_level = min(float(np.ceil((n + 1) * (1.0 - self.alpha)) / n), 1.0)
                self.regime_q_hat[regime] = float(np.quantile(np.asarray(residuals, dtype=np.float64), q_level))
            else:
                pooled = np.concatenate([np.asarray(values, dtype=np.float64) for values in self.regime_residuals.values() if values], axis=0) if any(self.regime_residuals.values()) else np.asarray([], dtype=np.float64)
                self.regime_q_hat[regime] = float(np.quantile(pooled, 1.0 - self.alpha)) if pooled.size else float("inf")

    def predict(self, consensus_path, current_regime: int) -> tuple[np.ndarray, np.ndarray, float]:
        path = np.asarray(consensus_path, dtype=np.float64)
        q_hat = float(self.regime_q_hat.get(int(current_regime), float("inf")))
        if not np.isfinite(q_hat):
            zeros = np.zeros_like(path)
            return path + zeros, path - zeros, 0.0
        return path + q_hat, path - q_hat, float(1.0 - self.alpha)

    def empirical_coverage(self, predicted_paths, realized_paths, regime_labels) -> float:
        inside = 0
        total = 0
        for pred, actual, regime in zip(predicted_paths, realized_paths, regime_labels, strict=False):
            upper, lower, _ = self.predict(pred, int(regime))
            actual_path = np.asarray(actual, dtype=np.float64)
            inside += int(np.all((actual_path <= upper) & (actual_path >= lower)))
            total += 1
        return float(inside / total) if total else 0.0

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump({"alpha": self.alpha, "regime_q_hat": self.regime_q_hat, "regime_residuals": self.regime_residuals}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "ConformalCone":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        cone = cls(alpha=float(payload.get("alpha", 0.15)))
        cone.regime_q_hat = {int(k): float(v) for k, v in dict(payload.get("regime_q_hat", {})).items()}
        cone.regime_residuals = {int(k): [float(item) for item in values] for k, values in dict(payload.get("regime_residuals", {})).items()}
        return cone
