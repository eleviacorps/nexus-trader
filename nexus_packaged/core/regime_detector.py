"""Signal and regime derivation from diffusion outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd


Signal = Literal["BUY", "SELL", "HOLD"]
Regime = Literal["TRENDING", "RANGING", "VOLATILE", "UNKNOWN"]


@dataclass
class SignalSnapshot:
    """Derived signal state used by UI/API panels."""

    signal: Signal
    confidence: float
    regime: Regime
    median_slope: float
    hurst_exponent: float
    updated_at: datetime
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    confidence_threshold: float = 0.55
    hold_reason: str = ""


@dataclass
class PathSignalDiagnostics:
    """Detailed path direction diagnostics for UI/debugging."""

    signal: Signal
    confidence: float
    median_slope: float
    positive_ratio: float
    negative_ratio: float
    threshold: float
    hold_reason: str = ""


def _hurst_exponent(series: np.ndarray) -> float:
    """Estimate Hurst exponent using a simple R/S log-log fit."""
    x = np.asarray(series, dtype=np.float64)
    if x.size < 32:
        return 0.5
    x = x[np.isfinite(x)]
    if x.size < 32:
        return 0.5
    lags = np.array([2, 4, 8, 16, 32], dtype=np.int64)
    tau = []
    for lag in lags:
        if lag >= x.size:
            continue
        diff = x[lag:] - x[:-lag]
        tau.append(np.sqrt(np.std(diff)))
    if len(tau) < 2:
        return 0.5
    poly = np.polyfit(np.log(lags[: len(tau)]), np.log(np.asarray(tau) + 1e-12), 1)
    hurst = float(np.clip(poly[0] * 2.0, 0.0, 1.0))
    return hurst


def classify_regime_from_prices(
    prices: pd.Series,
    *,
    hurst_window: int = 100,
    trending_threshold: float = 0.60,
    ranging_threshold: float = 0.40,
) -> Regime:
    """Classify market regime from recent prices."""
    if prices.empty:
        return "UNKNOWN"
    window = prices.tail(max(32, int(hurst_window)))
    if len(window) < 32:
        return "UNKNOWN"
    h = _hurst_exponent(window.to_numpy(dtype=np.float64))
    if h > trending_threshold:
        return "TRENDING"
    if h < ranging_threshold:
        return "RANGING"
    return "VOLATILE"


def derive_signal_from_paths(
    paths: np.ndarray,
    *,
    confidence_threshold: float = 0.55,
) -> tuple[Signal, float, float]:
    """Infer BUY/SELL/HOLD from diffusion paths.

    Returns:
        signal, confidence, median_slope
    """
    diag = derive_signal_diagnostics_from_paths(paths, confidence_threshold=confidence_threshold)
    return diag.signal, diag.confidence, diag.median_slope


def derive_signal_diagnostics_from_paths(
    paths: np.ndarray,
    *,
    confidence_threshold: float = 0.55,
) -> PathSignalDiagnostics:
    """Infer BUY/SELL/HOLD with direction diagnostics."""
    if paths.ndim != 2 or paths.shape[0] == 0 or paths.shape[1] < 2:
        return PathSignalDiagnostics(
            signal="HOLD",
            confidence=0.0,
            median_slope=0.0,
            positive_ratio=0.0,
            negative_ratio=0.0,
            threshold=float(confidence_threshold),
            hold_reason="insufficient_paths",
        )
    deltas = paths[:, -1] - paths[:, 0]
    positive = float(np.mean(deltas > 0.0))
    negative = float(np.mean(deltas < 0.0))
    median_slope = float(np.median(np.diff(np.median(paths, axis=0))))
    confidence = float(max(positive, negative))
    threshold = float(confidence_threshold)
    if confidence + 1e-9 < threshold:
        return PathSignalDiagnostics(
            signal="HOLD",
            confidence=confidence,
            median_slope=median_slope,
            positive_ratio=positive,
            negative_ratio=negative,
            threshold=threshold,
            hold_reason="below_confidence_threshold",
        )
    signal: Signal = "BUY" if positive >= negative else "SELL"
    return PathSignalDiagnostics(
        signal=signal,
        confidence=confidence,
        median_slope=median_slope,
        positive_ratio=positive,
        negative_ratio=negative,
        threshold=threshold,
        hold_reason="",
    )


def build_signal_snapshot(
    paths: np.ndarray,
    recent_prices: pd.Series,
    *,
    confidence_threshold: float,
    hurst_window: int,
    trending_threshold: float,
    ranging_threshold: float,
) -> SignalSnapshot:
    """Create a full signal snapshot from paths and recent prices."""
    diag = derive_signal_diagnostics_from_paths(
        paths,
        confidence_threshold=confidence_threshold,
    )
    regime = classify_regime_from_prices(
        recent_prices,
        hurst_window=hurst_window,
        trending_threshold=trending_threshold,
        ranging_threshold=ranging_threshold,
    )
    hurst = _hurst_exponent(
        recent_prices.tail(max(32, hurst_window)).to_numpy(dtype=np.float64)
    )
    return SignalSnapshot(
        signal=diag.signal,
        confidence=float(diag.confidence),
        regime=regime,
        median_slope=float(diag.median_slope),
        hurst_exponent=float(hurst),
        updated_at=datetime.now(timezone.utc),
        positive_ratio=float(diag.positive_ratio),
        negative_ratio=float(diag.negative_ratio),
        confidence_threshold=float(diag.threshold),
        hold_reason=str(diag.hold_reason),
    )
