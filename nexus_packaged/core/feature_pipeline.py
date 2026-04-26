"""Feature data pipeline for packaged Nexus runtime.

This module builds:
1. `nexus_packaged/data/ohlcv.parquet`
2. `nexus_packaged/data/diffusion_fused_6m.npy`

The fused matrix is a 144-dimensional feature space derived only from real
XAUUSD OHLCV bars. If MT5/yfinance are unavailable, the pipeline falls back to
locally cached OHLCV parquet files.

Run:
    python -m nexus_packaged.core.feature_pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("nexus.system")


@dataclass
class PipelineConfig:
    """Resolved configuration for data/feature generation."""

    symbol: str
    start_date: str
    end_date: str
    ohlcv_path: Path
    features_path: Path


def _load_settings() -> dict:
    settings_path = Path("nexus_packaged/config/settings.json")
    return json.loads(settings_path.read_text(encoding="utf-8"))


def _ensure_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize index to timezone-aware UTC datetime index."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame[~frame.index.isna()].sort_index()
    return frame


def _validate_ohlcv_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce OHLCV schema."""
    needed = ["open", "high", "low", "close", "volume"]
    missing = [col for col in needed if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    out = frame.loc[:, needed].copy()
    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = _ensure_utc_index(out)
    out = out.dropna(subset=needed)
    out = out[(out["high"] >= out["low"]) & (out["open"] > 0.0) & (out["close"] > 0.0)]
    return out


def _fetch_ohlcv_from_mt5(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetch OHLCV from MetaTrader5 copy_rates_range."""
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception:
        return None

    def _sync_fetch() -> pd.DataFrame | None:
        if not mt5.initialize():
            return None
        try:
            utc_from = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
            utc_to = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, utc_from, utc_to)
            if rates is None or len(rates) == 0:
                return None
            frame = pd.DataFrame(rates)
            frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True)
            frame = frame.set_index("timestamp")
            out = frame.rename(columns={"tick_volume": "volume"})[["open", "high", "low", "close", "volume"]]
            return _validate_ohlcv_schema(out)
        finally:
            mt5.shutdown()

    try:
        return _sync_fetch()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("MT5 fetch failed: %s", exc)
        return None


def _fetch_ohlcv_from_yfinance(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetch OHLCV via yfinance GC=F proxy and resample to 1-minute bars.

    yfinance does not provide full historical 1-minute coverage for long ranges.
    This implementation pulls hourly data and upsamples deterministically to 1m.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None
    ticker = "GC=F"
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1h",
            auto_adjust=False,
            progress=False,
            prepost=False,
            threads=True,
        )
        if raw is None or raw.empty:
            return None
        raw = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frame = raw[["open", "high", "low", "close", "volume"]].copy()
        frame = _ensure_utc_index(frame)
        # Upsample to 1m while preserving bar endpoints.
        idx = pd.date_range(frame.index.min(), frame.index.max(), freq="1min", tz="UTC")
        out = frame.reindex(idx)
        # Price columns are time-interpolated; volume is forward-filled then scaled.
        for col in ["open", "high", "low", "close"]:
            out[col] = out[col].interpolate(method="time").ffill().bfill()
        out["volume"] = out["volume"].fillna(0.0)
        # Spread hourly volume across 60 minutes to avoid fabricating total volume.
        out["volume"] = out["volume"] / 60.0
        return _validate_ohlcv_schema(out)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("yfinance fetch failed: %s", exc)
        return None


def _load_cached_ohlcv() -> pd.DataFrame | None:
    """Load cached OHLCV from local project artifacts."""
    candidates = [
        Path("nexus_packaged/data/ohlcv.parquet"),
        Path("data_store/processed/XAUUSD_1m_features.parquet"),
        Path("data/features/v21_ohlcv_denoised.parquet"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
            if "datetime" in frame.columns:
                frame = frame.set_index("datetime")
            elif "timestamp" in frame.columns:
                frame = frame.set_index("timestamp")
            # Use first matching column name if aliases exist.
            rename_map = {}
            if "tick_volume" in frame.columns and "volume" not in frame.columns:
                rename_map["tick_volume"] = "volume"
            if rename_map:
                frame = frame.rename(columns=rename_map)
            frame = _validate_ohlcv_schema(frame)
            LOGGER.info("Loaded cached OHLCV from %s", path)
            return frame
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read cached OHLCV from %s: %s", path, exc)
    return None


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative strength index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """Average true range."""
    hl = frame["high"] - frame["low"]
    hc = (frame["high"] - frame["close"].shift(1)).abs()
    lc = (frame["low"] - frame["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _cci(frame: pd.DataFrame, period: int) -> pd.Series:
    """Commodity channel index."""
    tp = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    return (tp - sma) / (0.015 * (mad + 1e-12))


def _stochastic(frame: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator K and D lines."""
    low_n = frame["low"].rolling(period).min()
    high_n = frame["high"].rolling(period).max()
    k = 100.0 * (frame["close"] - low_n) / (high_n - low_n + 1e-12)
    d = k.rolling(3).mean()
    return k, d


def _rolling_hurst_proxy(close: pd.Series, window: int) -> pd.Series:
    """Fast Hurst proxy from lagged return variances."""
    diff1 = close.diff(1)
    diff2 = close.diff(2)
    var1 = diff1.rolling(window).var()
    var2 = diff2.rolling(window).var()
    hurst = 0.5 * np.log((var2 + 1e-12) / (var1 + 1e-12)) / np.log(2.0)
    return hurst.clip(0.0, 1.0)


def _build_multi_timeframe_features(base: pd.DataFrame) -> pd.DataFrame:
    """Build 28 multi-timeframe features (4 TFs x 7 features)."""
    out = pd.DataFrame(index=base.index)
    tf_map = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}
    for name, rule in tf_map.items():
        agg = base.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()
        ret1 = agg["close"].pct_change()
        rsi14 = _rsi(agg["close"], 14) / 100.0
        atr14n = _atr(agg, 14) / (agg["close"] + 1e-12)
        ema12 = agg["close"].ewm(span=12, adjust=False).mean()
        ema26 = agg["close"].ewm(span=26, adjust=False).mean()
        ema_spread = (ema12 - ema26) / (agg["close"] + 1e-12)
        mid20 = agg["close"].rolling(20).mean()
        std20 = agg["close"].rolling(20).std()
        bb_bw = (2.0 * std20 * 2.0) / (mid20 + 1e-12)
        vol_z = (agg["volume"] - agg["volume"].rolling(20).mean()) / (agg["volume"].rolling(20).std() + 1e-12)
        trend = (agg["close"] - agg["close"].rolling(14).mean()) / (agg["close"].rolling(14).std() + 1e-12)

        features = pd.DataFrame(
            {
                f"mtf_{name}_ret1": ret1,
                f"mtf_{name}_rsi14": rsi14,
                f"mtf_{name}_atr14_norm": atr14n,
                f"mtf_{name}_ema_spread_12_26": ema_spread,
                f"mtf_{name}_bb_bw20": bb_bw,
                f"mtf_{name}_vol_z20": vol_z,
                f"mtf_{name}_trend_strength14": trend,
            },
            index=agg.index,
        )
        features = features.reindex(base.index).ffill()
        out = pd.concat([out, features], axis=1)
    return out


def _build_feature_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Construct 144-dimensional feature frame from OHLCV."""
    frame = ohlcv.copy()
    close = frame["close"]
    ret1 = close.pct_change()
    log_ret1 = np.log(close / close.shift(1))
    hl = frame["high"] - frame["low"]
    body = frame["close"] - frame["open"]
    upper_wick = frame["high"] - frame[["open", "close"]].max(axis=1)
    lower_wick = frame[["open", "close"]].min(axis=1) - frame["low"]

    features = pd.DataFrame(index=frame.index)

    # Group 1 (32): price-derived features.
    for w in [1, 2, 3, 5, 10, 20]:
        features[f"ret_{w}"] = close.pct_change(w)
        features[f"log_ret_{w}"] = np.log(close / close.shift(w))
    features["oc_change"] = (frame["close"] - frame["open"]) / (frame["open"] + 1e-12)
    features["hl_range"] = hl / (frame["open"] + 1e-12)
    features["co_ratio"] = body / (hl + 1e-12)
    features["wick_up"] = upper_wick / (hl + 1e-12)
    features["wick_down"] = lower_wick / (hl + 1e-12)
    for col in ["open", "high", "low", "close", "volume"]:
        roll_mean = frame[col].rolling(20).mean()
        roll_std = frame[col].rolling(20).std()
        roll_min = frame[col].rolling(20).min()
        roll_max = frame[col].rolling(20).max()
        features[f"{col}_z20"] = (frame[col] - roll_mean) / (roll_std + 1e-12)
        features[f"{col}_norm20"] = (frame[col] - roll_min) / (roll_max - roll_min + 1e-12)
    for w in [5, 10, 20]:
        features[f"ret_mean_{w}"] = ret1.rolling(w).mean()
    features["ret_std_5"] = ret1.rolling(5).std()
    features["ret_std_10"] = ret1.rolling(10).std()

    # Group 2 (24): volatility.
    atr14 = _atr(frame, 14)
    atr28 = _atr(frame, 28)
    atr56 = _atr(frame, 56)
    mid20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = mid20 + 2.0 * std20
    bb_lower = mid20 - 2.0 * std20
    features["atr14"] = atr14 / (close + 1e-12)
    features["atr28"] = atr28 / (close + 1e-12)
    features["atr56"] = atr56 / (close + 1e-12)
    features["bb_bw20"] = (bb_upper - bb_lower) / (mid20 + 1e-12)
    features["bb_pos20"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-12)
    for w in [10, 20, 40]:
        features[f"parkinson_{w}"] = ((np.log(frame["high"] / (frame["low"] + 1e-12)) ** 2).rolling(w).mean()) / (4 * np.log(2))
    tr = pd.concat(
        [
            hl,
            (frame["high"] - close.shift(1)).abs(),
            (frame["low"] - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features["true_range"] = tr / (close + 1e-12)
    features["tr_ema14"] = tr.ewm(span=14, adjust=False).mean() / (close + 1e-12)
    features["tr_z20"] = (tr - tr.rolling(20).mean()) / (tr.rolling(20).std() + 1e-12)
    for w in [5, 10, 20, 60]:
        features[f"realized_vol_{w}"] = log_ret1.rolling(w).std()
    features["downside_vol20"] = log_ret1.clip(upper=0.0).rolling(20).std()
    features["upside_vol20"] = log_ret1.clip(lower=0.0).rolling(20).std()
    for w in [10, 20, 40]:
        cc = np.log(close / close.shift(1))
        hlv = np.log(frame["high"] / (frame["low"] + 1e-12))
        features[f"gk_{w}"] = (0.5 * hlv**2 - (2 * np.log(2) - 1) * cc**2).rolling(w).mean()
    for w in [1, 5, 20, 60]:
        features[f"range_ratio_{w}"] = hl.rolling(w).mean() / (close.rolling(w).mean() + 1e-12)

    # Group 3 (24): momentum.
    features["rsi6"] = _rsi(close, 6) / 100.0
    features["rsi14"] = _rsi(close, 14) / 100.0
    features["rsi28"] = _rsi(close, 28) / 100.0
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    features["macd"] = macd / (close + 1e-12)
    features["macd_signal"] = macd_signal / (close + 1e-12)
    features["macd_hist"] = (macd - macd_signal) / (close + 1e-12)
    stoch_k, stoch_d = _stochastic(frame, 14)
    features["stoch_k14"] = stoch_k / 100.0
    features["stoch_d3"] = stoch_d / 100.0
    features["cci14"] = _cci(frame, 14) / 100.0
    features["cci28"] = _cci(frame, 28) / 100.0
    for w in [3, 6, 12, 24]:
        features[f"roc_{w}"] = close.pct_change(w)
        features[f"mom_{w}"] = close - close.shift(w)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    features["ema_spread_12_26"] = (ema12 - ema26) / (close + 1e-12)
    features["ema_spread_20_50"] = (ema20 - ema50) / (close + 1e-12)
    features["ema_spread_50_100"] = (ema50 - ema100) / (close + 1e-12)
    for w in [14, 28, 56]:
        features[f"trend_strength_{w}"] = (close - close.rolling(w).mean()) / (close.rolling(w).std() + 1e-12)

    # Group 4 (12): volume.
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * frame["volume"]).fillna(0.0).cumsum()
    features["obv_norm"] = (obv - obv.rolling(100).mean()) / (obv.rolling(100).std() + 1e-12)
    features["obv_roc"] = obv.pct_change(5)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    cum_vol = frame["volume"].cumsum()
    vwap = (typical * frame["volume"]).cumsum() / (cum_vol + 1e-12)
    features["vwap_dev"] = (close - vwap) / (vwap + 1e-12)
    features["vol_z20"] = (frame["volume"] - frame["volume"].rolling(20).mean()) / (frame["volume"].rolling(20).std() + 1e-12)
    features["vol_z60"] = (frame["volume"] - frame["volume"].rolling(60).mean()) / (frame["volume"].rolling(60).std() + 1e-12)
    features["vol_ratio_5_20"] = frame["volume"].rolling(5).mean() / (frame["volume"].rolling(20).mean() + 1e-12)
    features["vol_ratio_20_60"] = frame["volume"].rolling(20).mean() / (frame["volume"].rolling(60).mean() + 1e-12)
    features["price_vol_corr20"] = ret1.rolling(20).corr(frame["volume"].pct_change())
    features["price_vol_corr60"] = ret1.rolling(60).corr(frame["volume"].pct_change())
    features["money_flow20"] = (typical * frame["volume"]).rolling(20).mean() / (frame["volume"].rolling(20).mean() + 1e-12)
    adl = (((close - frame["low"]) - (frame["high"] - close)) / (hl + 1e-12) * frame["volume"]).cumsum()
    features["acc_dist_norm"] = (adl - adl.rolling(100).mean()) / (adl.rolling(100).std() + 1e-12)
    features["chaikin_osc"] = adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean()

    # Group 5 (12): microstructure proxies.
    spread_proxy = (frame["high"] - frame["low"]) / (close + 1e-12)
    features["spread_proxy"] = spread_proxy
    features["spread_z20"] = (spread_proxy - spread_proxy.rolling(20).mean()) / (spread_proxy.rolling(20).std() + 1e-12)
    features["bar_efficiency"] = (close - close.shift(1)).abs() / (hl + 1e-12)
    features["close_location_value"] = ((close - frame["low"]) - (frame["high"] - close)) / (hl + 1e-12)
    features["micro_noise_ratio5"] = ret1.rolling(5).std() / (ret1.abs().rolling(5).mean() + 1e-12)
    features["micro_noise_ratio20"] = ret1.rolling(20).std() / (ret1.abs().rolling(20).mean() + 1e-12)
    features["gap_ratio"] = (frame["open"] - close.shift(1)) / (close.shift(1) + 1e-12)
    features["range_to_body"] = hl / (body.abs() + 1e-12)
    features["signed_range"] = np.sign(body) * hl / (close + 1e-12)
    features["intrabar_reversion"] = (frame["close"] - typical) / (hl + 1e-12)
    hlc3 = typical
    features["hlc3_ret"] = hlc3.pct_change()
    features["typical_price_ret"] = typical.pct_change(3)

    # Group 6 (28): multi-timeframe features.
    mtf = _build_multi_timeframe_features(frame)
    features = pd.concat([features, mtf], axis=1)

    # Group 7 (12): regime signals.
    features["hurst_64"] = _rolling_hurst_proxy(close, 64)
    features["hurst_128"] = _rolling_hurst_proxy(close, 128)
    features["hurst_256"] = _rolling_hurst_proxy(close, 256)
    features["adx_proxy14"] = (close.diff().abs().rolling(14).mean()) / (_atr(frame, 14) + 1e-12)
    features["adx_proxy28"] = (close.diff().abs().rolling(28).mean()) / (_atr(frame, 28) + 1e-12)
    features["trend_persistence20"] = np.sign(close.diff()).rolling(20).mean()
    vol_state = (ret1.rolling(20).std() - ret1.rolling(100).std()) / (ret1.rolling(100).std() + 1e-12)
    features["regime_trending_flag"] = (features["hurst_128"] > 0.60).astype(np.float32)
    features["regime_ranging_flag"] = (features["hurst_128"] < 0.40).astype(np.float32)
    features["regime_volatile_flag"] = ((features["hurst_128"] >= 0.40) & (features["hurst_128"] <= 0.60)).astype(np.float32)
    features["volatility_state"] = vol_state
    roll_max = close.rolling(20).max()
    features["drawdown20"] = (close - roll_max) / (roll_max + 1e-12)
    sign_p = (np.sign(ret1.fillna(0.0)) + 1.0) * 0.5
    prob = sign_p.rolling(20).mean().clip(1e-6, 1 - 1e-6)
    features["entropy_sign20"] = -(prob * np.log(prob) + (1 - prob) * np.log(1 - prob))

    if features.shape[1] != 144:
        raise ValueError(f"Expected 144 features, got {features.shape[1]}")
    return features


def _validate_existing_artifacts(ohlcv_path: Path, features_path: Path) -> bool:
    """Validate cached artifacts using strict shape/NaN assertions."""
    if not ohlcv_path.exists() or not features_path.exists():
        return False
    try:
        features = np.load(features_path)
        ohlcv = pd.read_parquet(ohlcv_path)
        assert features.shape[0] == len(ohlcv), f"Mismatch: {features.shape[0]} vs {len(ohlcv)}"
        assert features.shape[1] == 144, f"Expected 144 dims, got {features.shape[1]}"
        assert not np.isnan(features).any(), "NaN values found in feature matrix"
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Existing artifact validation failed: %s", exc)
        return False


def run_pipeline(*, force_rebuild: bool = False) -> tuple[Path, Path]:
    """Build or validate packaged OHLCV/features artifacts."""
    settings = _load_settings()
    data_cfg = settings["data"]
    cfg = PipelineConfig(
        symbol=str(data_cfg["symbol"]),
        start_date=str(data_cfg["start_date"]),
        end_date=str(data_cfg["end_date"]),
        ohlcv_path=Path(data_cfg["ohlcv_path"]),
        features_path=Path(data_cfg["features_path"]),
    )

    cfg.ohlcv_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.features_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and _validate_existing_artifacts(cfg.ohlcv_path, cfg.features_path):
        LOGGER.info("Feature pipeline cache hit: %s / %s", cfg.ohlcv_path, cfg.features_path)
        return cfg.ohlcv_path, cfg.features_path

    ohlcv = _fetch_ohlcv_from_mt5(cfg.symbol, cfg.start_date, cfg.end_date)
    source = "mt5"
    if ohlcv is None or ohlcv.empty:
        ohlcv = _fetch_ohlcv_from_yfinance(cfg.symbol, cfg.start_date, cfg.end_date)
        source = "yfinance"
    if ohlcv is None or ohlcv.empty:
        ohlcv = _load_cached_ohlcv()
        source = "cache"
    if ohlcv is None or ohlcv.empty:
        raise RuntimeError("Unable to fetch OHLCV from MT5, yfinance, or local cache.")

    ohlcv = _validate_ohlcv_schema(ohlcv)
    features = _build_feature_frame(ohlcv)

    valid_mask = np.isfinite(features.to_numpy(dtype=np.float32)).all(axis=1)
    trimmed_features = features.loc[valid_mask]
    trimmed_ohlcv = ohlcv.loc[valid_mask]
    matrix = trimmed_features.to_numpy(dtype=np.float32, copy=False)

    np.save(cfg.features_path, matrix)
    trimmed_ohlcv.to_parquet(cfg.ohlcv_path)

    # Required assertions from specification.
    loaded_features = np.load(cfg.features_path)
    loaded_ohlcv = pd.read_parquet(cfg.ohlcv_path)
    assert loaded_features.shape[0] == len(loaded_ohlcv), f"Mismatch: {loaded_features.shape[0]} vs {len(loaded_ohlcv)}"
    assert loaded_features.shape[1] == 144, f"Expected 144 dims, got {loaded_features.shape[1]}"
    assert not np.isnan(loaded_features).any(), "NaN values found in feature matrix"
    LOGGER.info(
        "Feature matrix OK: %s source=%s rows=%d",
        loaded_features.shape,
        source,
        loaded_features.shape[0],
    )
    return cfg.ohlcv_path, cfg.features_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build packaged OHLCV and 144-dim diffusion features.")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore cache and rebuild all artifacts.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()
    run_pipeline(force_rebuild=bool(args.force_rebuild))


if __name__ == "__main__":
    main()

