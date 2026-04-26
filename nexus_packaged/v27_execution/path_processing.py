"""Path and OHLC processing utilities for execution/web visualization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _sanitize_path_matrix(paths: np.ndarray, current_price: float) -> np.ndarray:
    arr = np.asarray(paths, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    safe_price = float(current_price) if np.isfinite(current_price) and current_price > 0 else 1.0
    arr = np.where(np.isfinite(arr), arr, safe_price).astype(np.float32, copy=False)
    median_start = float(np.median(arr[:, 0]))
    median_abs = float(np.median(np.abs(arr)))
    mismatch = abs(median_start - safe_price) / max(safe_price, 1e-9)
    clearly_not_price = median_abs < safe_price * 0.2 or median_abs > safe_price * 5.0
    if mismatch > 0.5 or clearly_not_price:
        arr = arr - arr[:, [0]] + safe_price
    return arr.astype(np.float32, copy=False)


def _atr(ohlc: pd.DataFrame, period: int = 14) -> float:
    if ohlc is None or ohlc.empty:
        return 0.0
    frame = ohlc.tail(max(period + 2, 32)).copy()
    hl = frame["high"] - frame["low"]
    hc = (frame["high"] - frame["close"].shift(1)).abs()
    lc = (frame["low"] - frame["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(0.0 if pd.isna(atr) else atr)


def _scale_paths_to_price(
    *,
    raw_paths: np.ndarray,
    current_price: float,
    atr: float,
    output_normalized: bool,
    output_mean: float,
    output_std: float,
) -> np.ndarray:
    arr = np.asarray(raw_paths, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    safe_price = float(current_price)
    if output_normalized:
        std = float(output_std if abs(output_std) > 1e-9 else 1.0)
        arr = (arr * std) + float(output_mean)
    arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)
    safe_atr = float(atr if atr > 1e-9 else max(0.01, safe_price * 0.001))
    scale_factor = safe_atr * 0.5
    transformed = safe_price + (arr * scale_factor)
    # Force each path origin to live current price.
    transformed[:, 0] = safe_price
    max_move = safe_atr * 2.0
    transformed = np.clip(transformed, safe_price - max_move, safe_price + max_move)
    return transformed.astype(np.float32, copy=False)


def _series_from_matrix(paths: np.ndarray, start_ts: int, step_seconds: int) -> list[list[dict[str, Any]]]:
    if paths.ndim != 2 or paths.size == 0:
        return []
    step = max(1, int(step_seconds))
    out: list[list[dict[str, Any]]] = []
    for row in paths:
        points = [{"time": int(start_ts + i * step), "value": float(v)} for i, v in enumerate(row.tolist())]
        out.append(points)
    return out


def _ohlc_to_chart(ohlc: pd.DataFrame, limit: int = 500) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for idx, row in ohlc.tail(int(limit)).iterrows():
        payload.append(
            {
                "time": int(pd.Timestamp(idx).timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
    return payload


def prepare_chart_payload(
    *,
    paths: np.ndarray,
    current_price: float,
    bar_timestamp: pd.Timestamp,
    base_step_seconds: int,
    ohlc: pd.DataFrame,
    output_normalized: bool = False,
    output_mean: float = 0.0,
    output_std: float = 1.0,
) -> dict[str, Any]:
    """Convert runtime data into chart-ready payload."""
    atr = _atr(ohlc, period=14)
    scaled = _scale_paths_to_price(
        raw_paths=paths,
        current_price=current_price,
        atr=atr,
        output_normalized=output_normalized,
        output_mean=output_mean,
        output_std=output_std,
    )
    normalized = _sanitize_path_matrix(scaled, current_price=current_price)
    start_ts = int(pd.Timestamp(bar_timestamp).timestamp())
    series = _series_from_matrix(normalized, start_ts=start_ts, step_seconds=base_step_seconds)
    if normalized.ndim == 2 and normalized.size:
        mean = np.mean(normalized, axis=0).astype(np.float32)
        p10 = np.percentile(normalized, 10, axis=0).astype(np.float32)
        p90 = np.percentile(normalized, 90, axis=0).astype(np.float32)
        pmin = float(np.min(normalized))
        pmax = float(np.max(normalized))
    else:
        mean = np.zeros((0,), dtype=np.float32)
        p10 = np.zeros((0,), dtype=np.float32)
        p90 = np.zeros((0,), dtype=np.float32)
        pmin = float(current_price)
        pmax = float(current_price)
    return {
        "paths_matrix": normalized,
        "paths": series,
        "mean_path": _series_from_matrix(mean[None, :], start_ts=start_ts, step_seconds=base_step_seconds)[0]
        if mean.size
        else [],
        "confidence_band_10": _series_from_matrix(p10[None, :], start_ts=start_ts, step_seconds=base_step_seconds)[0]
        if p10.size
        else [],
        "confidence_band_90": _series_from_matrix(p90[None, :], start_ts=start_ts, step_seconds=base_step_seconds)[0]
        if p90.size
        else [],
        "ohlc": _ohlc_to_chart(ohlc, limit=500),
        "atr": float(atr),
        "path_min": pmin,
        "path_max": pmax,
        "scale_factor": float((atr if atr > 1e-9 else max(0.01, float(current_price) * 0.001)) * 0.5),
    }
