"""Path normalization and serialization helpers for hybrid charting."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_paths_relative_to_price(paths: np.ndarray, current_price: float) -> tuple[np.ndarray, str]:
    """Ensure path level is in current market price space.

    Returns:
        normalized_paths, mode
    """
    arr = np.asarray(paths, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0 or current_price <= 0:
        return arr, "passthrough"

    median_start = float(np.median(arr[:, 0]))
    median_abs = float(np.median(np.abs(arr)))
    mismatch = abs(median_start - current_price) / max(current_price, 1e-9)
    clearly_not_price = median_abs < current_price * 0.2 or median_abs > current_price * 5.0
    if mismatch > 0.5 or clearly_not_price:
        anchored = arr - arr[:, [0]] + float(current_price)
        return anchored.astype(np.float32, copy=False), "anchored_to_current_price"
    return arr.astype(np.float32, copy=False), "passthrough"


def summarize_path_distribution(paths: np.ndarray) -> dict[str, np.ndarray]:
    """Return mean/median/bands over path ensemble."""
    arr = np.asarray(paths, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        z = np.zeros((1,), dtype=np.float32)
        return {"mean": z, "median": z, "p10": z, "p90": z}
    return {
        "mean": np.mean(arr, axis=0).astype(np.float32),
        "median": np.median(arr, axis=0).astype(np.float32),
        "p10": np.percentile(arr, 10, axis=0).astype(np.float32),
        "p90": np.percentile(arr, 90, axis=0).astype(np.float32),
    }


def paths_to_time_value(paths: np.ndarray, start_ts: int, step_seconds: int) -> list[list[dict[str, Any]]]:
    """Convert (N,H) paths to [{time,value}] series list."""
    arr = np.asarray(paths, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return []
    step = max(1, int(step_seconds))
    series: list[list[dict[str, Any]]] = []
    for row in arr:
        points = [{"time": int(start_ts + i * step), "value": float(v)} for i, v in enumerate(row.tolist())]
        series.append(points)
    return series
