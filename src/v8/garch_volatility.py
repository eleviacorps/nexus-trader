from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    from arch import arch_model  # type: ignore
except ImportError:  # pragma: no cover
    arch_model = None


@dataclass(frozen=True)
class VolatilityModelReport:
    rows: int
    provider: str
    avg_expected_vol_15m: float
    avg_large_move_probability: float
    avg_vol_zscore: float


def _require_pandas():
    if pd is None:
        raise ImportError("pandas is required for v8 volatility features.")
    return pd


def _series(frame, column: str, default: float = 0.0):
    pandas = _require_pandas()
    if column in frame.columns:
        return pandas.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)
    return pandas.Series(np.full(len(frame), default, dtype=np.float32), index=frame.index)


def build_garch_like_frame(price_frame):
    pandas = _require_pandas()
    returns = _series(price_frame, "return_1")
    abs_ret = returns.abs()
    atr_pct = _series(price_frame, "atr_pct").abs()
    base_vol = abs_ret.ewm(span=24, adjust=False).mean().fillna(0.0)
    provider = "ewm_fallback"
    if arch_model is not None and 500 <= len(returns) <= 250_000:
        try:
            model = arch_model(returns.to_numpy(dtype=np.float64) * 100.0, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
            fitted = model.fit(disp="off", show_warning=False)
            cond_vol = np.asarray(fitted.conditional_volatility, dtype=np.float32) / 100.0
            base_vol = pandas.Series(cond_vol, index=price_frame.index).ffill().bfill().fillna(0.0)
            provider = "arch_garch"
        except Exception:
            provider = "ewm_fallback"
    elif arch_model is not None and len(returns) > 250_000:
        provider = "ewm_fallback_large_dataset"
    expected_vol_5m = np.maximum(base_vol.to_numpy(dtype=np.float32), atr_pct.to_numpy(dtype=np.float32) * 0.55)
    expected_vol_15m = expected_vol_5m * np.sqrt(3.0)
    expected_vol_30m = expected_vol_5m * np.sqrt(6.0)
    forward_15m = _series(price_frame, "forward_return_15m")
    vol_zscore_15m = forward_15m.to_numpy(dtype=np.float32) / np.maximum(expected_vol_15m, 1e-6)
    large_move_probability = np.clip(np.abs(vol_zscore_15m) / 3.0, 0.0, 1.0).astype(np.float32)
    plausible_upper = _series(price_frame, "close").to_numpy(dtype=np.float32) * (1.0 + expected_vol_15m)
    plausible_lower = _series(price_frame, "close").to_numpy(dtype=np.float32) * (1.0 - expected_vol_15m)
    frame = pandas.DataFrame(
        {
            "v8_expected_vol_5m": expected_vol_5m.astype(np.float32),
            "v8_expected_vol_15m": expected_vol_15m.astype(np.float32),
            "v8_expected_vol_30m": expected_vol_30m.astype(np.float32),
            "v8_volatility_zscore_15m": vol_zscore_15m.astype(np.float32),
            "v8_large_move_probability": large_move_probability,
            "v8_plausible_upper_15m": plausible_upper.astype(np.float32),
            "v8_plausible_lower_15m": plausible_lower.astype(np.float32),
        },
        index=price_frame.index,
    )
    frame.attrs["provider"] = provider
    return frame


def summarize_volatility_frame(frame) -> VolatilityModelReport:
    return VolatilityModelReport(
        rows=int(len(frame)),
        provider=str(frame.attrs.get("provider", "unknown")),
        avg_expected_vol_15m=float(frame["v8_expected_vol_15m"].mean()) if len(frame) else 0.0,
        avg_large_move_probability=float(frame["v8_large_move_probability"].mean()) if len(frame) else 0.0,
        avg_vol_zscore=float(np.abs(frame["v8_volatility_zscore_15m"]).mean()) if len(frame) else 0.0,
    )
