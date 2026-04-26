from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover
    from scipy.stats import linregress as _linregress
except Exception:  # pragma: no cover
    _linregress = None


def _linear_slope(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2:
        return 0.0
    if _linregress is not None:
        return float(_linregress(x_values, y_values).slope)
    slope, _ = np.polyfit(np.asarray(x_values, dtype=np.float64), np.asarray(y_values, dtype=np.float64), 1)
    return float(slope)


class MultifractalMarketMemory:
    def __init__(
        self,
        window: int = 252,
        overlap_ratio: float = 0.333,
        q: float = 2.0,
        scale_range: tuple[int, int] = (4, 64),
        n_scales: int = 10,
    ) -> None:
        self.window = int(window)
        self.overlap = float(overlap_ratio)
        self.q = float(q)
        self.scales = np.unique(
            np.logspace(np.log10(scale_range[0]), np.log10(scale_range[1]), n_scales).astype(int)
        )

    def _profile(self, series: np.ndarray) -> np.ndarray:
        centered = np.asarray(series, dtype=np.float64) - float(np.mean(series))
        return np.cumsum(centered)

    def _detrend_segment(self, segment: np.ndarray, order: int = 2) -> np.ndarray:
        x_axis = np.arange(len(segment), dtype=np.float64)
        degree = min(order, max(len(segment) - 1, 1))
        coeffs = np.polyfit(x_axis, segment, degree)
        return segment - np.polyval(coeffs, x_axis)

    def _fluctuation_q(
        self,
        profile_x: np.ndarray,
        profile_y: np.ndarray,
        scale: int,
        direction: str | None,
        trend_proxy: np.ndarray,
    ) -> float:
        stride = max(1, int(scale * (1.0 - self.overlap)))
        detrended_x: list[np.ndarray] = []
        detrended_y: list[np.ndarray] = []
        slopes: list[float] = []

        start = 0
        while start + scale <= len(profile_x):
            seg_x = self._detrend_segment(profile_x[start : start + scale])
            seg_y = self._detrend_segment(profile_y[start : start + scale])
            local_x = np.arange(scale, dtype=np.float64)
            local_slope, _ = np.polyfit(local_x, trend_proxy[start : start + scale], 1)
            detrended_x.append(seg_x)
            detrended_y.append(seg_y)
            slopes.append(float(local_slope))
            start += stride

        if not detrended_x:
            return float("nan")

        fluctuations = np.asarray(
            [np.mean(np.abs(seg_x * seg_y)) for seg_x, seg_y in zip(detrended_x, detrended_y, strict=False)],
            dtype=np.float64,
        )
        slope_array = np.asarray(slopes, dtype=np.float64)

        if direction == "positive":
            mask = slope_array > 0.0
        elif direction == "negative":
            mask = slope_array <= 0.0
        else:
            mask = np.ones(len(slope_array), dtype=bool)

        if int(np.sum(mask)) < 2:
            return float("nan")

        selected = np.clip(fluctuations[mask], 1e-12, None)
        fluctuation_q = np.mean(selected ** (self.q / 2.0)) ** (1.0 / self.q)
        return float(fluctuation_q)

    def compute_hurst(
        self,
        log_returns: np.ndarray,
        vol_increments: np.ndarray,
        direction: str | None = None,
    ) -> float:
        returns = np.asarray(log_returns, dtype=np.float64)
        volatility = np.asarray(vol_increments, dtype=np.float64)
        if len(returns) < int(max(self.scales) * 3):
            return 0.5

        profile_x = self._profile(returns)
        profile_y = self._profile(volatility)
        trend_proxy = returns
        log_scale: list[float] = []
        log_fluctuation: list[float] = []

        for scale in self.scales:
            fluctuation = self._fluctuation_q(profile_x, profile_y, int(scale), direction, trend_proxy)
            if np.isnan(fluctuation) or fluctuation <= 0.0:
                continue
            log_scale.append(float(np.log(scale)))
            log_fluctuation.append(float(np.log(fluctuation)))

        if len(log_scale) < 3:
            return 0.5
        slope = _linear_slope(log_scale, log_fluctuation)
        return float(np.clip(slope, 0.1, 1.5))

    def compute_all(self, log_returns: np.ndarray, vol_increments: np.ndarray) -> dict[str, float | str]:
        hurst_all = self.compute_hurst(log_returns, vol_increments, direction=None)
        hurst_positive = self.compute_hurst(log_returns, vol_increments, direction="positive")
        hurst_negative = self.compute_hurst(log_returns, vol_increments, direction="negative")
        return {
            "hurst_overall": round(hurst_all, 4),
            "hurst_positive": round(hurst_positive, 4),
            "hurst_negative": round(hurst_negative, 4),
            "hurst_asymmetry": round(hurst_positive - hurst_negative, 4),
            "market_memory_regime": (
                "persistent"
                if hurst_all > 0.55
                else "anti_persistent"
                if hurst_all < 0.45
                else "random_walk"
            ),
        }

    def rolling_features(
        self,
        features_df: pd.DataFrame,
        return_col: str = "return_1",
        vol_col: str = "atr_pct",
    ) -> pd.DataFrame:
        working = features_df.copy()
        results: list[dict[str, float | str]] = []
        for index in range(len(working)):
            start = max(0, index - self.window + 1)
            window_df = working.iloc[start : index + 1]
            if len(window_df) < 30:
                results.append(
                    {
                        "hurst_overall": 0.5,
                        "hurst_positive": 0.5,
                        "hurst_negative": 0.5,
                        "hurst_asymmetry": 0.0,
                        "market_memory_regime": "random_walk",
                    }
                )
                continue
            returns = window_df[return_col].fillna(0.0).to_numpy(dtype=np.float64)
            volatility = window_df[vol_col].fillna(0.0).to_numpy(dtype=np.float64)
            results.append(self.compute_all(returns, volatility))
        return pd.concat([working, pd.DataFrame(results, index=working.index)], axis=1)
