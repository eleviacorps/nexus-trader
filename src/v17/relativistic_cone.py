from __future__ import annotations

import numpy as np


class RelativisticCone:
    def __init__(self, historical_log_returns: np.ndarray) -> None:
        returns = np.asarray(historical_log_returns, dtype=np.float64)
        if returns.size == 0:
            self.c_m = 0.003
        else:
            self.c_m = float(np.percentile(np.abs(returns), 99))
            if self.c_m < 1e-6:
                self.c_m = 0.003

    def envelope(
        self,
        current_price: float,
        n_bars: int,
        hurst_positive: float = 0.55,
        hurst_negative: float = 0.62,
    ) -> dict[str, object]:
        steps = np.arange(1, int(n_bars) + 1, dtype=np.float64)
        max_log_up = self.c_m * steps
        max_log_down = self.c_m * steps
        persistence_up = 2.0 * (float(hurst_positive) - 0.5)
        persistence_down = 2.0 * (float(hurst_negative) - 0.5)
        expected_log_up = max_log_up * persistence_up
        expected_log_down = max_log_down * persistence_down
        sigma = self.c_m * np.sqrt(steps) * (1.0 - min(abs(float(hurst_positive) - 0.5) * 2.0, 0.95))
        sigma = np.clip(sigma, self.c_m * 0.15, None)

        return {
            "cone_upper": (float(current_price) * np.exp(max_log_up)).tolist(),
            "cone_lower": (float(current_price) * np.exp(-max_log_down)).tolist(),
            "inner_upper": (float(current_price) * np.exp(expected_log_up + sigma)).tolist(),
            "inner_lower": (float(current_price) * np.exp(-expected_log_down - sigma)).tolist(),
            "c_m": round(float(self.c_m), 6),
            "compact_support": True,
            "asymmetric": abs(float(hurst_positive) - float(hurst_negative)) > 0.02,
            "h_plus": round(float(hurst_positive), 4),
            "h_minus": round(float(hurst_negative), 4),
        }

    def branch_plausibility(self, branch_log_returns: np.ndarray, n_bars: int) -> float:
        cumulative = np.cumsum(np.asarray(branch_log_returns[:n_bars], dtype=np.float64))
        if cumulative.size == 0:
            return 0.5
        steps = np.arange(1, min(len(cumulative), int(n_bars)) + 1, dtype=np.float64)
        max_allowed = self.c_m * steps
        inside = np.abs(cumulative[: len(max_allowed)]) <= max_allowed
        return float(np.mean(inside)) if inside.size else 0.5
