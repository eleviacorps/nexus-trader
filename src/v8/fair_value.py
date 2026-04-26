from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


@dataclass(frozen=True)
class FairValueReport:
    rows: int
    avg_dislocation: float
    avg_mean_reversion_pressure: float


def _require_pandas():
    if pd is None:
        raise ImportError("pandas is required for v8 fair-value features.")
    return pd


def _series(frame, column: str, default: float = 0.0):
    pandas = _require_pandas()
    if column in frame.columns:
        return pandas.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)
    return pandas.Series(np.full(len(frame), default, dtype=np.float32), index=frame.index)


def _kalman_smooth(close: np.ndarray, anchor: np.ndarray, process_scale: np.ndarray, measurement_scale: np.ndarray):
    length = len(close)
    fair_value = np.zeros(length, dtype=np.float32)
    dislocation = np.zeros(length, dtype=np.float32)
    pressure = np.zeros(length, dtype=np.float32)
    if length == 0:
        return fair_value, dislocation, pressure
    state = float(anchor[0] if np.isfinite(anchor[0]) else close[0])
    covariance = 1.0
    for idx in range(length):
        measurement = float(anchor[idx])
        process_var = max(1e-6, float(process_scale[idx]))
        measurement_var = max(1e-6, float(measurement_scale[idx]))
        covariance = covariance + process_var
        kalman_gain = covariance / (covariance + measurement_var)
        state = state + kalman_gain * (measurement - state)
        covariance = (1.0 - kalman_gain) * covariance
        fair_value[idx] = state
        dislocation[idx] = float((close[idx] - state) / max(abs(state), 1e-6))
        pressure[idx] = float(np.tanh(-dislocation[idx] * 8.0))
    return fair_value, dislocation, pressure


def build_fair_value_frame(price_frame):
    pandas = _require_pandas()
    close = _series(price_frame, "close")
    atr_pct = _series(price_frame, "atr_pct").abs()
    dxy = _series(price_frame, "macro_dollar_proxy", _series(price_frame, "dollar_proxy", 0.0))
    rates = _series(price_frame, "macro_rates_10y", _series(price_frame, "rates_10y", 0.0))
    vix = _series(price_frame, "macro_volatility", _series(price_frame, "volatility", 0.0))
    bonds = _series(price_frame, "macro_bonds", _series(price_frame, "bonds", 0.0))
    trend = _series(price_frame, "quant_trend_score", 0.0)

    macro_anchor = (
        0.62 * close.ewm(span=64, adjust=False).mean()
        + 0.12 * (1.0 - np.tanh(dxy / 4.0)) * close
        + 0.10 * (1.0 - np.tanh(rates / 4.0)) * close
        + 0.08 * np.tanh(vix / 4.0) * close
        + 0.08 * np.tanh(bonds / 4.0) * close
    )
    process_scale = (atr_pct + close.pct_change().abs().fillna(0.0) + 1e-6).to_numpy(dtype=np.float32)
    measurement_scale = (atr_pct.rolling(24, min_periods=4).mean().fillna(atr_pct.mean() or 1e-4) + 1e-6).to_numpy(dtype=np.float32)
    fair_value, dislocation, pressure = _kalman_smooth(
        close.to_numpy(dtype=np.float32),
        macro_anchor.to_numpy(dtype=np.float32),
        process_scale,
        measurement_scale,
    )
    frame = pandas.DataFrame(
        {
            "v8_fair_value": fair_value.astype(np.float32),
            "v8_fair_value_dislocation": dislocation.astype(np.float32),
            "v8_mean_reversion_pressure": (pressure * (1.0 - np.abs(trend.to_numpy(dtype=np.float32)) * 0.35)).astype(np.float32),
        },
        index=price_frame.index,
    )
    return frame


def summarize_fair_value_frame(frame) -> FairValueReport:
    return FairValueReport(
        rows=int(len(frame)),
        avg_dislocation=float(np.abs(frame["v8_fair_value_dislocation"]).mean()) if len(frame) else 0.0,
        avg_mean_reversion_pressure=float(np.abs(frame["v8_mean_reversion_pressure"]).mean()) if len(frame) else 0.0,
    )
