from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config.project_config import LEGACY_PROCESSED_DIR, PRICE_FEATURE_COLUMNS


DEFAULT_RAW_BAR_PATH = LEGACY_PROCESSED_DIR / "XAUUSD_1m_full.parquet"
DEFAULT_ARCHIVE_FEATURE_PATH = LEGACY_PROCESSED_DIR / "XAUUSD_1m_features.parquet"


def _to_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _coerce_bar_index(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "datetime" in working.columns:
        working["datetime"] = pd.to_datetime(working["datetime"], utc=True, errors="coerce")
        working = working.set_index("datetime")
    else:
        working.index = pd.to_datetime(working.index, utc=True, errors="coerce")
    working = working.sort_index()
    working = working.loc[~working.index.isna()].copy()
    return working


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _finalize_causal(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.replace([np.inf, -np.inf], np.nan).ffill()
    defaults = {
        "rsi_14": 50.0,
        "rsi_7": 50.0,
        "stoch_k": 50.0,
        "stoch_d": 50.0,
        "bb_pct": 0.5,
        "volume_ratio": 1.0,
        "is_bullish": 0.0,
    }
    for column in output.columns:
        fill_value = defaults.get(column, 0.0)
        output[column] = output[column].fillna(fill_value)
    return output


def compute_bar_consistent_features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = _coerce_bar_index(candles)
    frame = frame[[column for column in ("open", "high", "low", "close", "volume") if column in frame.columns]].copy()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            frame[column] = 0.0
    frame["volume"] = _safe_numeric(frame["volume"], default=0.0)
    if float(frame["volume"].max()) <= 0.0:
        frame["volume"] = 1.0

    close = _safe_numeric(frame["close"])
    high = _safe_numeric(frame["high"])
    low = _safe_numeric(frame["low"])
    open_ = _safe_numeric(frame["open"])
    volume = _safe_numeric(frame["volume"], default=1.0)

    returns = {period: close.pct_change(period).fillna(0.0) for period in (1, 3, 6, 12)}
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_sig
    rsi_14 = _compute_rsi(close, 14)
    rsi_7 = _compute_rsi(close, 7)
    lowest_14 = low.rolling(14, min_periods=2).min()
    highest_14 = high.rolling(14, min_periods=2).max()
    stoch_k = ((close - lowest_14) / (highest_14 - lowest_14).replace(0.0, np.nan) * 100.0).fillna(50.0)
    stoch_d = stoch_k.rolling(3, min_periods=1).mean().fillna(50.0)

    prev_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    base_range = (high - low).expanding(min_periods=1).mean()
    atr_14 = true_range.rolling(14, min_periods=2).mean().fillna(base_range).fillna(0.0)
    atr_pct = (atr_14 / close.replace(0.0, np.nan)).fillna(0.0)

    rolling_mean = close.rolling(20, min_periods=5).mean()
    rolling_std = close.rolling(20, min_periods=5).std(ddof=0).replace(0.0, np.nan)
    bb_upper = rolling_mean + (2.0 * rolling_std)
    bb_lower = rolling_mean - (2.0 * rolling_std)
    bb_width = ((bb_upper - bb_lower) / rolling_mean.replace(0.0, np.nan)).fillna(0.0)
    bb_pct = ((close - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.5)

    candle_range = (high - low).replace(0.0, np.nan)
    body = (close - open_).abs()
    body_pct = (body / candle_range).fillna(0.0)
    upper_wick = ((high - pd.concat([open_, close], axis=1).max(axis=1)) / candle_range).fillna(0.0)
    lower_wick = ((pd.concat([open_, close], axis=1).min(axis=1) - low) / candle_range).fillna(0.0)
    displacement = ((close - open_) / atr_14.replace(0.0, np.nan)).fillna(0.0)

    rolling_high = high.rolling(20, min_periods=2).max()
    rolling_low = low.rolling(20, min_periods=2).min()
    dist_to_high = ((rolling_high - close) / atr_14.replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)
    dist_to_low = ((close - rolling_low) / atr_14.replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)
    hh = (high >= rolling_high.shift(1).fillna(rolling_high)).astype(float)
    ll = (low <= rolling_low.shift(1).fillna(rolling_low)).astype(float)
    volume_ratio = (volume / volume.rolling(20, min_periods=2).mean().replace(0.0, np.nan)).fillna(1.0)

    timestamps = frame.index.tz_convert("UTC") if frame.index.tz is not None else frame.index.tz_localize("UTC")
    hour = timestamps.hour.astype(float)
    dow = timestamps.dayofweek.astype(float)
    session_asian = ((hour >= 0) & (hour < 7)).astype(float)
    session_london = ((hour >= 7) & (hour < 13)).astype(float)
    session_ny = ((hour >= 13) & (hour < 21)).astype(float)
    session_overlap = ((hour >= 12) & (hour < 16)).astype(float)
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
    dow_cos = np.cos(2.0 * np.pi * dow / 7.0)

    enriched = frame.copy()
    enriched["return_1"] = returns[1]
    enriched["return_3"] = returns[3]
    enriched["return_6"] = returns[6]
    enriched["return_12"] = returns[12]
    enriched["rsi_14"] = rsi_14
    enriched["rsi_7"] = rsi_7
    enriched["macd_hist"] = macd_hist
    enriched["macd"] = macd
    enriched["macd_sig"] = macd_sig
    enriched["stoch_k"] = stoch_k
    enriched["stoch_d"] = stoch_d
    enriched["ema_9_ratio"] = ((close / ema_9.replace(0.0, np.nan)) - 1.0).fillna(0.0)
    enriched["ema_21_ratio"] = ((close / ema_21.replace(0.0, np.nan)) - 1.0).fillna(0.0)
    enriched["ema_50_ratio"] = ((close / ema_50.replace(0.0, np.nan)) - 1.0).fillna(0.0)
    enriched["ema_cross"] = np.sign(ema_9 - ema_21).astype(float)
    enriched["atr_pct"] = atr_pct
    enriched["bb_width"] = bb_width
    enriched["bb_pct"] = bb_pct
    enriched["body_pct"] = body_pct
    enriched["upper_wick"] = upper_wick
    enriched["lower_wick"] = lower_wick
    enriched["is_bullish"] = (close >= open_).astype(float)
    enriched["displacement"] = displacement
    enriched["dist_to_high"] = dist_to_high
    enriched["dist_to_low"] = dist_to_low
    enriched["hh"] = hh
    enriched["ll"] = ll
    enriched["volume_ratio"] = volume_ratio
    enriched["session_asian"] = session_asian
    enriched["session_london"] = session_london
    enriched["session_ny"] = session_ny
    enriched["session_overlap"] = session_overlap
    enriched["hour_sin"] = hour_sin
    enriched["hour_cos"] = hour_cos
    enriched["dow_sin"] = dow_sin
    enriched["dow_cos"] = dow_cos
    enriched = _finalize_causal(enriched)
    return enriched


def load_default_raw_bars(
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    path: Path = DEFAULT_RAW_BAR_PATH,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = _coerce_bar_index(frame)
    if start is not None:
        frame = frame.loc[frame.index >= _to_utc_timestamp(start)]
    if end is not None:
        frame = frame.loc[frame.index < _to_utc_timestamp(end)]
    return frame[["open", "high", "low", "close", "volume"]].copy()


def load_default_archive_features(
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    path: Path = DEFAULT_ARCHIVE_FEATURE_PATH,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = _coerce_bar_index(frame)
    if start is not None:
        frame = frame.loc[frame.index >= _to_utc_timestamp(start)]
    if end is not None:
        frame = frame.loc[frame.index < _to_utc_timestamp(end)]
    available = [name for name in PRICE_FEATURE_COLUMNS if name in frame.columns]
    return frame[available].copy()


@dataclass
class OnlineFeatureEngine:
    warmup_bars: int = 200
    buffer_size: int = 600

    def __post_init__(self) -> None:
        self.buffer: deque[dict[str, Any]] = deque(maxlen=self.buffer_size)
        self._bar_count = 0
        self._last_timestamp: pd.Timestamp | None = None

    def update(self, bar: dict[str, Any] | pd.Series) -> bool:
        if isinstance(bar, pd.Series):
            payload = bar.to_dict()
        else:
            payload = dict(bar)
        timestamp = payload.pop("datetime", payload.pop("timestamp", None))
        if timestamp is None:
            raise ValueError("OnlineFeatureEngine.update requires a datetime/timestamp field.")
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        payload["datetime"] = ts
        self.buffer.append(payload)
        self._bar_count += 1
        self._last_timestamp = ts
        return self.ready

    @property
    def ready(self) -> bool:
        return self._bar_count >= int(self.warmup_bars)

    def get_feature_frame(self) -> pd.DataFrame | None:
        if not self.ready:
            return None
        frame = pd.DataFrame(list(self.buffer))
        frame = frame.set_index("datetime")
        return compute_bar_consistent_features(frame)

    def get_features(self) -> dict[str, float] | None:
        feature_frame = self.get_feature_frame()
        if feature_frame is None or feature_frame.empty:
            return None
        latest = feature_frame.iloc[-1]
        return {name: float(latest.get(name, 0.0)) for name in PRICE_FEATURE_COLUMNS}


def compute_online_feature_frame(
    candles: pd.DataFrame,
    *,
    warmup_bars: int = 200,
    buffer_size: int = 600,
) -> pd.DataFrame:
    frame = _coerce_bar_index(candles)
    if frame.empty:
        return pd.DataFrame(columns=list(PRICE_FEATURE_COLUMNS))
    # BCFE is fully causal, so the vectorized pass is equivalent to bar-by-bar
    # replay after warmup without the prohibitive O(n * buffer) cost.
    causal = compute_bar_consistent_features(frame)
    start = max(int(warmup_bars) - 1, 0)
    return causal.iloc[start:][list(PRICE_FEATURE_COLUMNS)].copy()


def align_feature_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    feature_names: Iterable[str] = PRICE_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = [name for name in feature_names if name in left.columns and name in right.columns]
    common_index = left.index.intersection(right.index)
    return (
        left.loc[common_index, usable].sort_index(),
        right.loc[common_index, usable].sort_index(),
    )
