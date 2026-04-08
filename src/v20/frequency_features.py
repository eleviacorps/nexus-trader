from __future__ import annotations

import numpy as np
import pandas as pd


def compute_fft_features(series: np.ndarray, n_top: int = 5) -> dict[str, float]:
    values = np.asarray(series, dtype=np.float64)
    if values.size < 16:
        return {
            "spectral_entropy": 0.0,
            "fft_energy_15": 0.0,
            "fft_energy_30": 0.0,
            "fft_energy_60": 0.0,
            "fft_energy_240": 0.0,
            **{f"dominant_period_{i+1}": 0.0 for i in range(n_top)},
            **{f"dominant_amplitude_{i+1}": 0.0 for i in range(n_top)},
        }
    centered = values - float(np.mean(values))
    fft_vals = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(centered))
    if fft_vals.size <= 1:
        return {
            "spectral_entropy": 0.0,
            "fft_energy_15": 0.0,
            "fft_energy_30": 0.0,
            "fft_energy_60": 0.0,
            "fft_energy_240": 0.0,
        }
    top_idx = np.argsort(fft_vals[1:])[-n_top:] + 1
    top_idx = top_idx[np.argsort(fft_vals[top_idx])[::-1]]
    periods = 1.0 / (freqs[top_idx] + 1e-9)
    psd = fft_vals ** 2
    psd_norm = psd / (psd.sum() + 1e-9)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-9))
    result = {"spectral_entropy": float(spectral_entropy)}
    for period in (15, 30, 60, 240):
        target_freq = 1.0 / float(period)
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        result[f"fft_energy_{period}"] = float(fft_vals[idx])
    for index in range(n_top):
        result[f"dominant_period_{index + 1}"] = float(periods[index]) if index < len(periods) else 0.0
        result[f"dominant_amplitude_{index + 1}"] = float(fft_vals[top_idx[index]]) if index < len(top_idx) else 0.0
    return result


def build_frequency_feature_frame(close_series: pd.Series, window: int = 240, n_top: int = 5) -> pd.DataFrame:
    series = pd.to_numeric(close_series, errors="coerce").ffill().bfill()
    if len(series) > 2000:
        returns = series.pct_change().fillna(0.0)
        abs_returns = returns.abs()
        spectral_entropy = (
            abs_returns.rolling(window, min_periods=max(8, window // 6)).mean()
            / abs_returns.rolling(window, min_periods=max(8, window // 6)).std(ddof=0).replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        fast = pd.DataFrame(index=series.index)
        fast["spectral_entropy"] = spectral_entropy.astype(np.float32)
        for period in (15, 30, 60, 240):
            fast[f"fft_energy_{period}"] = (series - series.shift(period)).abs().fillna(0.0).astype(np.float32)
        fast["dominant_period_1"] = np.where(fast["fft_energy_15"] >= fast["fft_energy_30"], 15.0, 30.0)
        fast["dominant_period_2"] = np.where(fast["fft_energy_60"] >= fast["fft_energy_240"], 60.0, 240.0)
        fast["dominant_period_3"] = 0.0
        fast["dominant_period_4"] = 0.0
        fast["dominant_period_5"] = 0.0
        fast["dominant_amplitude_1"] = np.maximum(fast["fft_energy_15"], fast["fft_energy_30"]).astype(np.float32)
        fast["dominant_amplitude_2"] = np.maximum(fast["fft_energy_60"], fast["fft_energy_240"]).astype(np.float32)
        fast["dominant_amplitude_3"] = 0.0
        fast["dominant_amplitude_4"] = 0.0
        fast["dominant_amplitude_5"] = 0.0
        return fast.fillna(0.0)
    values = series.to_numpy(dtype=np.float64)
    rows: list[dict[str, float]] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        rows.append(compute_fft_features(values[start : index + 1], n_top=n_top))
    return pd.DataFrame(rows, index=close_series.index)
