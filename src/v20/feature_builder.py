from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.v12.bar_consistent_features import compute_bar_consistent_features
from src.v17.mmm import MultifractalMarketMemory
from src.v20.frequency_features import build_frequency_feature_frame
from src.v20.macro_features import MACRO_FEATURE_COLS, compute_macro_features, infer_bars_per_day
from src.v20.regime_detector import RegimeDetector, train_hmm
from src.v20.wavelet_denoiser import WaveletDenoiser


def _safe_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(default)


def _build_wltc_features(ohlcv: pd.DataFrame, window: int = 96) -> pd.DataFrame:
    close = _safe_series(ohlcv, "close")
    returns = close.pct_change().fillna(0.0)
    pos_momentum = returns.clip(lower=0.0).rolling(window, min_periods=max(8, window // 4)).mean().fillna(0.0)
    neg_momentum = (-returns.clip(upper=0.0)).rolling(window, min_periods=max(8, window // 4)).mean().fillna(0.0)
    realized_noise = returns.abs().rolling(window, min_periods=max(8, window // 4)).std(ddof=0).fillna(0.0)
    institutional = (returns.ewm(span=max(4, window // 2), adjust=False).mean().abs() * 6.0).clip(lower=0.0)
    retail = (pos_momentum * 12.0).clip(lower=0.0)
    noise = (realized_noise * 25.0).clip(lower=0.0)
    phase_names = np.select(
        [
            (noise > retail) & (noise > institutional),
            (institutional > retail) & (institutional >= noise),
            retail > 0.0,
        ],
        ["noise", "institutional", "retail"],
        default="balanced",
    )
    phase_age: list[int] = []
    current_age = 0
    prev_phase: str | None = None
    for phase in phase_names.tolist():
        if phase == prev_phase:
            current_age += 1
        else:
            current_age = 1
            prev_phase = str(phase)
        phase_age.append(current_age)
    strength = np.maximum.reduce([retail.to_numpy(dtype=np.float64), institutional.to_numpy(dtype=np.float64), noise.to_numpy(dtype=np.float64)])
    return pd.DataFrame(
        {
            "wltc_phase": phase_names,
            "wltc_phase_age": np.asarray(phase_age, dtype=np.int64),
            "wltc_strength": np.clip(strength, 0.0, 1.0).astype(np.float32),
            "wltc_negative_pressure": neg_momentum.astype(np.float32),
        },
        index=ohlcv.index,
    )


def _build_mfg_features(ohlcv: pd.DataFrame, window: int = 96) -> pd.DataFrame:
    close = _safe_series(ohlcv, "close")
    returns = close.pct_change().fillna(0.0)
    retail = returns.ewm(span=max(4, window // 4), adjust=False).mean()
    institutional = returns.ewm(span=max(8, window // 2), adjust=False).mean()
    algo = returns.ewm(span=max(2, window // 8), adjust=False).mean()
    whale = returns.rolling(max(6, window // 6), min_periods=3).mean().fillna(0.0)
    noise = returns.rolling(max(6, window // 6), min_periods=3).std(ddof=0).fillna(0.0) * np.sign(returns.rolling(max(6, window // 6), min_periods=3).mean().fillna(0.0))
    persona_matrix = np.vstack([retail.to_numpy(), institutional.to_numpy(), algo.to_numpy(), whale.to_numpy(), noise.to_numpy()])
    drift = np.mean(persona_matrix, axis=0)
    disagreement = np.std(persona_matrix, axis=0)
    momentum = np.concatenate([[0.0], np.diff(drift)])
    herding = np.clip(1.0 - disagreement * 400.0, 0.0, 1.0)
    return pd.DataFrame(
        {
            "mfg_mean_belief": drift.astype(np.float32),
            "mfg_disagreement": disagreement.astype(np.float32),
            "mfg_momentum": momentum.astype(np.float32),
            "mfg_herding": herding.astype(np.float32),
        },
        index=ohlcv.index,
    )


def _build_structure_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = _safe_series(frame, "close")
    high = _safe_series(frame, "high")
    low = _safe_series(frame, "low")
    atr = _safe_series(frame, "atr_pct", 0.001) * close.replace(0.0, np.nan)
    rolling_high = high.rolling(60, min_periods=10).max().fillna(high)
    rolling_low = low.rolling(60, min_periods=10).min().fillna(low)
    nearest_ob_bull = ((close - rolling_low) / atr.replace(0.0, np.nan)).fillna(0.0).clip(lower=0.0)
    nearest_ob_bear = ((rolling_high - close) / atr.replace(0.0, np.nan)).fillna(0.0).clip(lower=0.0)
    gap_high = low.shift(2)
    gap_low = high.shift(0)
    nearest_fvg = ((gap_low - gap_high).abs() / atr.replace(0.0, np.nan)).fillna(0.0)
    premium_discount = np.where(close >= (rolling_high + rolling_low) / 2.0, 1.0, 0.0)
    return pd.DataFrame(
        {
            "nearest_ob_bull_dist": nearest_ob_bull.astype(np.float32),
            "nearest_ob_bear_dist": nearest_ob_bear.astype(np.float32),
            "nearest_fvg_dist": nearest_fvg.astype(np.float32),
            "premium_discount": premium_discount.astype(np.float32),
        },
        index=frame.index,
    )


def _join_mtf(base: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=base.index)
    for rule, suffix in (("5min", "5m"), ("15min", "15m"), ("30min", "30m")):
        resampled = ohlcv.resample(rule).agg({"close": "last"}).dropna()
        resampled[f"close_{suffix}"] = resampled["close"]
        resampled[f"roc_{suffix}"] = resampled["close"].pct_change().fillna(0.0)
        aligned = resampled[[f"close_{suffix}", f"roc_{suffix}"]].reindex(base.index, method="ffill")
        output = output.join(aligned, how="left")
    return output.fillna(0.0)


def _expand_with_lags(frame: pd.DataFrame, lag_columns: list[str], lags: tuple[int, ...] = (1, 2, 3, 6, 12)) -> pd.DataFrame:
    output = frame.copy()
    for column in lag_columns:
        if column not in output.columns:
            continue
        for lag in lags:
            output[f"{column}_lag_{lag}"] = output[column].shift(lag)
    return output


def build_v20_feature_frame(
    ohlcv: pd.DataFrame,
    *,
    hmm_detector: RegimeDetector | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = ohlcv.copy()
    working.index = pd.to_datetime(working.index, utc=True, errors="coerce")
    working = working.loc[~working.index.isna()].sort_index()
    denoised = WaveletDenoiser().fit_transform(working)
    micro = compute_bar_consistent_features(denoised)
    bars_per_day = infer_bars_per_day(working.index)
    macro = compute_macro_features(working, bars_per_day=bars_per_day)
    freq = build_frequency_feature_frame(_safe_series(micro, "close"), window=min(240, len(micro)))
    mmm = MultifractalMarketMemory(window=min(252, len(micro))).rolling_features(micro[["return_1", "atr_pct"]], return_col="return_1", vol_col="atr_pct")
    wltc = _build_wltc_features(working)
    mfg = _build_mfg_features(working)
    structure = _build_structure_features(micro)
    mtf = _join_mtf(micro, working)
    regime_source = pd.DataFrame(
        {
            "log_return": np.log(_safe_series(micro, "close").replace(0.0, np.nan)).diff().fillna(0.0),
            "realized_vol_20": _safe_series(macro, "macro_realized_vol_20"),
            "volume_zscore": ((_safe_series(micro, "volume") - _safe_series(micro, "volume").rolling(96, min_periods=12).mean()) / _safe_series(micro, "volume").rolling(96, min_periods=12).std(ddof=0).replace(0.0, np.nan)).fillna(0.0),
            "macro_vol_regime_class": _safe_series(macro, "macro_vol_regime_class"),
            "macro_jump_flag": _safe_series(macro, "macro_jump_flag"),
        },
        index=micro.index,
    )
    detector = hmm_detector
    if detector is None:
        detector, _, _ = train_hmm(regime_source)
    regime = detector.transform(regime_source)
    features = pd.concat(
        [
            micro,
            denoised[[col for col in denoised.columns if str(col).endswith("_denoised_delta")]],
            macro[MACRO_FEATURE_COLS],
            freq,
            mmm[[col for col in mmm.columns if str(col).startswith("hurst_")] + ["market_memory_regime"]],
            wltc,
            mfg,
            regime,
            mtf,
            structure,
        ],
        axis=1,
    )
    features["future_return_15m"] = _safe_series(features, "close").shift(-15) / _safe_series(features, "close").replace(0.0, np.nan) - 1.0
    features["future_return_30m"] = _safe_series(features, "close").shift(-30) / _safe_series(features, "close").replace(0.0, np.nan) - 1.0
    features["target_up_15m"] = (features["future_return_15m"] > 0.0).astype(np.float32)
    features["target_up_30m"] = (features["future_return_30m"] > 0.0).astype(np.float32)
    features["range_forward_15m"] = (
        _safe_series(features, "high").rolling(15, min_periods=1).max().shift(-14) - _safe_series(features, "low").rolling(15, min_periods=1).min().shift(-14)
    ).fillna(0.0)
    lag_candidates = [
        "return_1",
        "return_3",
        "rsi_14",
        "atr_pct",
        "macro_dxy_zscore_20d",
        "macro_trend_strength",
        "spectral_entropy",
        "fft_energy_15",
        "hurst_overall",
        "hurst_asymmetry",
        "mfg_disagreement",
        "mfg_mean_belief",
        "hmm_state",
        "hmm_duration",
        "close_5m",
        "roc_5m",
        "close_15m",
        "roc_15m",
        "close_30m",
        "roc_30m",
        "nearest_ob_bull_dist",
        "nearest_ob_bear_dist",
    ]
    features = _expand_with_lags(features, lag_candidates)
    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    metadata = {
        "feature_count": int(len(features.columns)),
        "bars_per_day": int(bars_per_day),
        "regime_state_names": detector.state_names,
        "groups": {
            "micro": [col for col in micro.columns],
            "macro": MACRO_FEATURE_COLS,
            "frequency": [col for col in freq.columns],
            "mmm": [col for col in mmm.columns if str(col).startswith("hurst_")] + ["market_memory_regime"],
            "wltc": list(wltc.columns),
            "mfg": list(mfg.columns),
            "regime": list(regime.columns),
            "mtf": list(mtf.columns),
            "structure": list(structure.columns),
        },
    }
    return features, metadata


def save_feature_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
