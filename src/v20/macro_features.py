from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MACRO_FEATURE_COLS = [
    "macro_dxy_zscore_20d",
    "macro_dxy_zscore_60d",
    "macro_vol_regime_class",
    "macro_trend_strength",
    "macro_jump_flag",
    "macro_realized_vol_20",
    "macro_realized_vol_60",
    "macro_bipower_var_20",
]


def infer_bars_per_day(index: pd.Index) -> int:
    timestamps = pd.to_datetime(index, utc=True, errors="coerce")
    if len(timestamps) < 3:
        return 96
    deltas = pd.Series(timestamps).diff().dropna()
    median_seconds = max(float(deltas.dt.total_seconds().median()), 60.0)
    return int(max(1, round(86400.0 / median_seconds)))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(8, window // 5)).mean()
    std = series.rolling(window, min_periods=max(8, window // 5)).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _bipower_variation(log_returns: pd.Series, window: int) -> pd.Series:
    abs_returns = log_returns.abs()
    bpv = (np.pi / 2.0) * abs_returns.rolling(window, min_periods=max(6, window // 3)).apply(
        lambda values: float(np.sum(values[1:] * values[:-1])) if len(values) > 1 else 0.0,
        raw=True,
    )
    return bpv.fillna(0.0)


def _vol_regime_from_zscore(zscore: pd.Series) -> pd.Series:
    bins = pd.Series(np.zeros(len(zscore), dtype=np.int64), index=zscore.index)
    bins.loc[zscore >= 0.5] = 1
    bins.loc[zscore >= 1.25] = 2
    bins.loc[zscore >= 2.0] = 3
    return bins.astype(np.int64)


def _merge_external_macro(base: pd.DataFrame, external_dir: Path | None) -> pd.DataFrame:
    if external_dir is None or not external_dir.exists():
        return base
    output = base.copy()
    for name in ("DXY", "TIPS", "VIX"):
        path = external_dir / f"{name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "date" not in frame.columns:
            continue
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        output = output.merge(frame, how="left", on="date", suffixes=("", f"_{name.lower()}"))
    return output


def compute_macro_features(
    ohlcv: pd.DataFrame,
    *,
    external_dir: str | Path | None = None,
    bars_per_day: int | None = None,
) -> pd.DataFrame:
    frame = ohlcv.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().bfill()
    log_return = np.log(close).diff().fillna(0.0)
    bars_per_day = int(bars_per_day or infer_bars_per_day(frame.index))
    win20 = max(20, bars_per_day * 20)
    win60 = max(60, bars_per_day * 60)
    realized_vol_20 = log_return.rolling(win20, min_periods=max(20, win20 // 8)).std(ddof=0).fillna(0.0)
    realized_vol_60 = log_return.rolling(win60, min_periods=max(20, win60 // 8)).std(ddof=0).fillna(0.0)
    dxy_proxy = -close.pct_change(max(1, bars_per_day)).fillna(0.0)
    dxy_z20 = _rolling_zscore(dxy_proxy, win20)
    dxy_z60 = _rolling_zscore(dxy_proxy, win60)
    baseline = close.pct_change(max(2, bars_per_day * 5)).rolling(max(10, bars_per_day * 20), min_periods=max(10, bars_per_day)).mean().fillna(0.0)
    trend_strength = close.pct_change(max(1, bars_per_day)).fillna(0.0) - baseline
    vol_z = _rolling_zscore(realized_vol_20, max(20, bars_per_day * 10))
    vol_regime = _vol_regime_from_zscore(vol_z)
    bpv20 = _bipower_variation(log_return, max(20, bars_per_day))
    rv20 = log_return.pow(2).rolling(max(20, bars_per_day), min_periods=max(10, bars_per_day // 2)).sum().fillna(0.0)
    jump_ratio = ((rv20 - bpv20).clip(lower=0.0) / (bpv20.abs() + 1e-9)).fillna(0.0)
    jump_flag = (jump_ratio > jump_ratio.rolling(max(20, bars_per_day * 5), min_periods=max(10, bars_per_day)).quantile(0.8)).astype(np.int64)
    macro = pd.DataFrame(
        {
            "date": pd.to_datetime(frame.index, utc=True).floor("D"),
            "macro_dxy_zscore_20d": dxy_z20.astype(np.float32),
            "macro_dxy_zscore_60d": dxy_z60.astype(np.float32),
            "macro_vol_regime_class": vol_regime.astype(np.int64),
            "macro_trend_strength": trend_strength.astype(np.float32),
            "macro_jump_flag": jump_flag.astype(np.int64),
            "macro_realized_vol_20": realized_vol_20.astype(np.float32),
            "macro_realized_vol_60": realized_vol_60.astype(np.float32),
            "macro_bipower_var_20": bpv20.astype(np.float32),
        },
        index=frame.index,
    )
    merged = _merge_external_macro(macro.reset_index(drop=True), Path(external_dir) if external_dir is not None else None)
    merged.index = frame.index
    return merged[MACRO_FEATURE_COLS + [col for col in merged.columns if col not in MACRO_FEATURE_COLS]]
