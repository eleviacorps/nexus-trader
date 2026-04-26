from __future__ import annotations

import json
import shutil
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq

from config.project_config import (
    MT5_COMMON_FILES_DIR,
    V21_BIMAMBA_MODEL_PATH,
    V21_FEATURES_PATH,
    V21_MT5_TESTER_SIGNALS_PATH,
    V21_MT5_TESTER_SUMMARY_PATH,
    V21_XLSTM_MODEL_PATH,
)
from src.v12.bar_consistent_features import load_default_raw_bars
from src.v20.mamba_backbone import NexusBiMamba
from src.v20.runtime import _build_branch_candidates, _safe_float as _runtime_safe_float
from src.v20.runtime import _load_conformal
from src.v21.runtime import build_v21_local_judge, build_v21_runtime_state
from src.v21.runtime import _clip01, _confidence_tier, _regime_label, _stance_from_prob
from src.v21.runtime_v21 import V21Runtime
from src.v21.xlstm_backbone import NexusXLSTM


RuntimeBuilder = Callable[[dict[str, Any]], dict[str, Any]]
JudgeBuilder = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _ensure_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working.index = pd.to_datetime(working.index, utc=True, errors="coerce")
    working = working.loc[~working.index.isna()].sort_index()
    return working


def _candles_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timestamp, row in frame.iterrows():
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "open": float(row.get("open", 0.0) or 0.0),
                "high": float(row.get("high", 0.0) or 0.0),
                "low": float(row.get("low", 0.0) or 0.0),
                "close": float(row.get("close", 0.0) or 0.0),
                "volume": float(row.get("volume", 0.0) or 0.0),
            }
        )
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _required_runtime_columns() -> list[str]:
    return [
        "close",
        "atr_14",
        "atr_pct",
        "macro_trend_strength",
        "quant_trend_score",
        "rsi_14",
        "hurst_overall",
        "mfg_mean_belief",
        "roc_15m",
        "quant_route_confidence",
        "quant_regime_strength",
        "mfg_disagreement",
        "quant_vol_realism",
        "spectral_entropy",
        "macro_dxy_zscore_20d",
        "macro_dxy_zscore_60d",
        "macro_realized_vol_20",
        "macro_realized_vol_60",
        "hmm_state",
        "hmm_prob_0",
        "hmm_prob_1",
        "hmm_prob_2",
        "hmm_prob_3",
        "hmm_prob_4",
        "hmm_prob_5",
    ]


@lru_cache(maxsize=1)
def _available_feature_columns() -> tuple[str, ...]:
    if not V21_FEATURES_PATH.exists():
        return tuple()
    parquet_file = pq.ParquetFile(V21_FEATURES_PATH)
    return tuple(str(name) for name in parquet_file.schema.names)


def _prepare_runtime_feature_defaults(feature_frame: pd.DataFrame) -> pd.DataFrame:
    working = feature_frame.copy()
    close_series = pd.to_numeric(working.get("close"), errors="coerce").ffill().bfill().fillna(0.0)
    atr_pct_series = pd.to_numeric(working.get("atr_pct"), errors="coerce").ffill().bfill().fillna(0.0015)

    if "atr_14" not in working.columns:
        working["atr_14"] = (close_series.abs() * atr_pct_series.abs()).clip(lower=close_series.abs() * 0.0005, upper=close_series.abs() * 0.05)

    hmm_probability_columns = [column for column in working.columns if str(column).startswith("hmm_prob_")]
    if hmm_probability_columns:
        hmm_probability_frame = working[hmm_probability_columns].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
        hmm_strength = hmm_probability_frame.max(axis=1)
    else:
        hmm_strength = pd.Series(0.5, index=working.index, dtype=np.float32)

    if "quant_route_confidence" not in working.columns:
        working["quant_route_confidence"] = hmm_strength.clip(lower=0.2, upper=0.95)
    if "quant_regime_strength" not in working.columns:
        working["quant_regime_strength"] = hmm_strength.clip(lower=0.2, upper=0.95)
    if "quant_vol_realism" not in working.columns:
        vol_realism = 1.0 - (atr_pct_series.abs() * 20.0).clip(lower=0.0, upper=0.8)
        working["quant_vol_realism"] = vol_realism.clip(lower=0.2, upper=0.95)
    if "spectral_entropy" not in working.columns:
        working["spectral_entropy"] = 0.0
    if "macro_realized_vol_20" not in working.columns:
        working["macro_realized_vol_20"] = atr_pct_series.abs()
    if "macro_realized_vol_60" not in working.columns:
        working["macro_realized_vol_60"] = atr_pct_series.abs()

    return working


@lru_cache(maxsize=1)
def _load_xlstm_export_bundle() -> tuple[dict[str, Any], NexusXLSTM] | tuple[None, None]:
    if not V21_XLSTM_MODEL_PATH.exists():
        return None, None
    checkpoint = torch.load(V21_XLSTM_MODEL_PATH, map_location="cpu")
    model = NexusXLSTM(
        n_features=len(checkpoint.get("feature_columns") or []),
        d_model=int(checkpoint.get("d_model", 128)),
        n_layers=int(checkpoint.get("n_layers", 2)),
        n_regimes=6,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model


@lru_cache(maxsize=1)
def _load_bimamba_export_bundle() -> tuple[dict[str, Any], NexusBiMamba] | tuple[None, None]:
    if not V21_BIMAMBA_MODEL_PATH.exists():
        return None, None
    checkpoint = torch.load(V21_BIMAMBA_MODEL_PATH, map_location="cpu")
    model = NexusBiMamba(
        n_features=len(checkpoint.get("feature_columns") or []),
        sequence_len=int(checkpoint.get("sequence_len", 120)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model


def _normalize_frame(frame: pd.DataFrame, feature_columns: list[str], feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    numeric = frame.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    aligned = numeric.reindex(columns=feature_columns, fill_value=0.0).to_numpy(dtype=np.float32)
    std = np.where(np.asarray(feature_std, dtype=np.float32) == 0.0, 1.0, np.asarray(feature_std, dtype=np.float32))
    return np.clip((aligned - np.asarray(feature_mean, dtype=np.float32)) / std, -8.0, 8.0)


def _build_fast_runtime_from_row(
    latest: dict[str, Any],
    *,
    mode: str,
    x_prob: float,
    b_prob: float,
    regime_probs: list[float],
    equity: float,
) -> dict[str, Any]:
    current_price = _runtime_safe_float(latest.get("close"), 0.0)
    branches = _build_branch_candidates(latest)
    if branches.empty:
        return {"available": False, "runtime_version": "v21_fast_export", "execution_reason": "no_branches"}
    top_branch = branches.iloc[0].to_dict()
    top_cabr = _runtime_safe_float(top_branch.get("v20_cabr_score"), 0.5)
    consensus_price = float(branches.head(5)["predicted_price_15m"].mean())
    minority_price = float(branches.tail(5)["predicted_price_15m"].mean()) if len(branches) >= 5 else consensus_price
    cone = _load_conformal()
    path = np.asarray([current_price, (current_price + consensus_price) / 2.0, consensus_price], dtype=np.float64)
    upper, lower, conformal_confidence = cone.predict(path, int(_runtime_safe_float(latest.get("hmm_state"), 0.0)))
    cone_width = abs(float(upper[-1] - lower[-1])) / 0.1 if len(upper) and len(lower) else 0.0
    ensemble_prob = _clip01((0.65 * float(x_prob)) + (0.35 * float(b_prob)))
    disagree_prob = _clip01(abs(float(x_prob) - float(b_prob)))
    raw_stance = _stance_from_prob(ensemble_prob)
    branch_stance = str(top_branch.get("decision_direction", "HOLD")).upper()
    frequency_mode = str(mode).lower() == "frequency"
    used_branch_fallback = False
    if frequency_mode and raw_stance == "HOLD" and branch_stance in {"BUY", "SELL"}:
        raw_stance = branch_stance
        used_branch_fallback = True

    atr_value = max(_runtime_safe_float(latest.get("atr_14"), current_price * 0.0015), current_price * 0.0005, 0.1)
    top_branch_prices = branches.head(10)["predicted_price_15m"].astype(float).tolist()
    dangerous_branch_count = 0
    for price in top_branch_prices:
        displacement = price - current_price
        if raw_stance == "BUY" and displacement < -(0.80 * atr_value):
            dangerous_branch_count += 1
        elif raw_stance == "SELL" and displacement > (0.80 * atr_value):
            dangerous_branch_count += 1
        elif raw_stance == "HOLD" and abs(displacement) > (1.20 * atr_value):
            dangerous_branch_count += 1

    meta_label_prob = _clip01(
        (0.30 * (abs(ensemble_prob - 0.5) * 2.0))
        + (0.25 * float(conformal_confidence))
        + (0.20 * top_cabr)
        + (0.15 * (1.0 - disagree_prob))
        + (0.10 * (1.0 - min(dangerous_branch_count, 5) / 5.0))
    )
    runtime_gate = V21Runtime(mode="research" if frequency_mode else "production")
    should_execute, failed_gates = runtime_gate.should_trade(
        sjd_output={"stance": raw_stance, "disagree_prob": disagree_prob},
        conformal_confidence=float(conformal_confidence),
        dangerous_branch_count=dangerous_branch_count,
        meta_label_prob=meta_label_prob,
    )
    confidence_tier = _confidence_tier(ensemble_prob, meta_label_prob, disagree_prob, cone_width)
    kelly_fraction = _clip01((abs(ensemble_prob - 0.5) * 2.0) * float(conformal_confidence) * (1.0 - disagree_prob) * 0.35)
    suggested_lot = runtime_gate.get_size(kelly_fraction=kelly_fraction, account_balance=float(equity), price=max(current_price, 1.0))
    cpm_score = _clip01(
        (0.28 * _runtime_safe_float(latest.get("quant_route_confidence"), 0.5))
        + (0.18 * _runtime_safe_float(latest.get("quant_regime_strength"), 0.5))
        + (0.16 * float(conformal_confidence))
        + (0.14 * (1.0 - min(abs(_runtime_safe_float(latest.get("mfg_disagreement"), 0.0)), 1.0)))
        + (0.24 * (abs(ensemble_prob - 0.5) * 2.0))
    )
    fallback_note = f", branch_fallback={branch_stance}" if used_branch_fallback else ""
    execution_reason = (
        f"V21 fast export cleared: stance={raw_stance}, ensemble={ensemble_prob:.3f}, conformal={float(conformal_confidence):.3f}, meta={meta_label_prob:.3f}, dangerous_branches={dangerous_branch_count}{fallback_note}."
        if should_execute
        else f"V21 fast export hold: raw_stance={raw_stance}, failed_gates={','.join(failed_gates) if failed_gates else 'none'}, ensemble={ensemble_prob:.3f}, conformal={float(conformal_confidence):.3f}, meta={meta_label_prob:.3f}, disagree={disagree_prob:.3f}{fallback_note}."
    )
    return {
        "available": True,
        "runtime_version": "v21_fast_export",
        "selected_branch_id": int(top_branch.get("branch_id", 0)),
        "selected_branch_label": str(top_branch.get("branch_label", "v21_branch")),
        "decision_direction": raw_stance if should_execute else "HOLD",
        "raw_stance": raw_stance,
        "cabr_score": round(top_cabr, 6),
        "cabr_raw_score": round(_runtime_safe_float(top_branch.get("v20_cabr_raw_score"), top_cabr), 6),
        "cpm_score": round(cpm_score, 6),
        "confidence_tier": confidence_tier,
        "sqt_label": "GOOD" if meta_label_prob >= 0.58 and confidence_tier in {"high", "very_high"} else "NEUTRAL" if meta_label_prob >= 0.38 else "CAUTION",
        "cone_width_pips": round(float(cone_width), 3),
        "lepl_action": raw_stance,
        "lepl_probabilities": {"execute": round(meta_label_prob, 6), "hold": round(1.0 - meta_label_prob, 6)},
        "lepl_features": {
            "kelly_fraction": round(kelly_fraction, 6),
            "suggested_lot": round(float(suggested_lot), 4),
            "conformal_confidence": round(float(conformal_confidence), 6),
            "dangerous_branch_count": int(dangerous_branch_count),
            "paper_equity": round(float(equity), 2),
        },
        "should_execute": bool(should_execute),
        "execution_reason": execution_reason,
        "consensus_path": [round(float(item), 5) for item in path.tolist()],
        "minority_path": [round(float(item), 5) for item in [current_price, (current_price + minority_price) / 2.0, minority_price]],
        "cone_upper": [round(float(item), 5) for item in upper.tolist()],
        "cone_lower": [round(float(item), 5) for item in lower.tolist()],
        "regime_probs": [round(float(value), 6) for value in regime_probs],
        "v21_dir_15m_prob": round(float(x_prob), 6),
        "v21_bimamba_prob": round(float(b_prob), 6),
        "v21_ensemble_prob": round(float(ensemble_prob), 6),
        "v21_disagree_prob": round(float(disagree_prob), 6),
        "v21_meta_label_prob": round(float(meta_label_prob), 6),
        "v21_dangerous_branch_count": int(dangerous_branch_count),
        "v21_top_vsn_features": [],
        "v21_regime_label": _regime_label(regime_probs),
        "v21_used_branch_fallback": bool(used_branch_fallback),
    }


def _fast_signal_rows_from_precomputed_features(
    raw_15m: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbol: str,
    mode: str,
    lookback_bars: int,
    equity: float,
    pip_size: float,
    judge_builder: JudgeBuilder,
) -> pd.DataFrame:
    x_checkpoint, x_model = _load_xlstm_export_bundle()
    b_checkpoint, b_model = _load_bimamba_export_bundle()
    if x_checkpoint is None or x_model is None or b_checkpoint is None or b_model is None:
        return pd.DataFrame()

    available_columns = set(_available_feature_columns())
    required_columns = sorted(set(list(x_checkpoint.get("feature_columns") or []) + list(b_checkpoint.get("feature_columns") or []) + _required_runtime_columns()))
    selected_columns = [column for column in required_columns if column in available_columns]
    if not selected_columns:
        return pd.DataFrame()
    feature_frame = pd.read_parquet(V21_FEATURES_PATH, columns=selected_columns)
    feature_frame = _ensure_utc_index(feature_frame)
    lead_start = start - pd.Timedelta(minutes=max(int(lookback_bars), int(x_checkpoint.get("sequence_len", 120))) * 15)
    feature_frame = feature_frame.loc[(feature_frame.index >= lead_start) & (feature_frame.index < end)].copy()
    if feature_frame.empty:
        return pd.DataFrame()
    feature_frame = _prepare_runtime_feature_defaults(feature_frame)

    x_columns = list(x_checkpoint.get("feature_columns") or [])
    b_columns = list(b_checkpoint.get("feature_columns") or [])
    x_normalized = _normalize_frame(feature_frame, x_columns, np.asarray(x_checkpoint["feature_mean"], dtype=np.float32), np.asarray(x_checkpoint["feature_std"], dtype=np.float32))
    b_normalized = _normalize_frame(feature_frame, b_columns, np.asarray(b_checkpoint["feature_mean"], dtype=np.float32), np.asarray(b_checkpoint["feature_std"], dtype=np.float32))
    regime_ids = (
        pd.to_numeric(feature_frame.get("hmm_state"), errors="coerce").fillna(0).clip(lower=0, upper=5).astype(np.int64).to_numpy()
    )

    seq_len = int(x_checkpoint.get("sequence_len", 120))
    eligible_positions: list[int] = []
    eligible_timestamps: list[pd.Timestamp] = []
    for pos, timestamp in enumerate(feature_frame.index):
        if pos < seq_len - 1:
            continue
        if timestamp < start or timestamp >= end:
            continue
        eligible_positions.append(pos)
        eligible_timestamps.append(pd.Timestamp(timestamp))
    if not eligible_positions:
        return pd.DataFrame()

    x_probabilities: dict[int, float] = {}
    regime_probability_rows: dict[int, list[float]] = {}
    batch_size = 128
    with torch.no_grad():
        for batch_start in range(0, len(eligible_positions), batch_size):
            batch_positions = eligible_positions[batch_start : batch_start + batch_size]
            x_batch = np.stack([x_normalized[pos - seq_len + 1 : pos + 1] for pos in batch_positions], axis=0)
            regime_batch = np.stack([regime_ids[pos - seq_len + 1 : pos + 1] for pos in batch_positions], axis=0)
            outputs = x_model(torch.tensor(x_batch, dtype=torch.float32), torch.tensor(regime_batch, dtype=torch.long))
            dir_probs = torch.sigmoid(outputs["dir_15m"]).cpu().numpy().astype(float)
            regime_probs = torch.softmax(outputs["regime"], dim=-1).cpu().numpy().astype(float)
            for local_index, pos in enumerate(batch_positions):
                x_probabilities[pos] = float(dir_probs[local_index])
                regime_probability_rows[pos] = [float(item) for item in regime_probs[local_index].tolist()]

    b_probabilities: dict[int, float] = {}
    with torch.no_grad():
        for batch_start in range(0, len(eligible_positions), batch_size):
            batch_positions = eligible_positions[batch_start : batch_start + batch_size]
            b_batch = np.stack([b_normalized[pos - seq_len + 1 : pos + 1] for pos in batch_positions], axis=0)
            outputs = b_model(torch.tensor(b_batch, dtype=torch.float32))
            dir_probs = torch.sigmoid(outputs["dir_15m"]).cpu().numpy().astype(float)
            for local_index, pos in enumerate(batch_positions):
                b_probabilities[pos] = float(dir_probs[local_index])

    market_15m = _ensure_utc_index(raw_15m)
    records: list[dict[str, Any]] = []
    signal_id = 1
    for pos, signal_time in zip(eligible_positions, eligible_timestamps, strict=False):
        execution_time = signal_time + pd.Timedelta(minutes=15)
        if execution_time >= end or execution_time not in market_15m.index:
            continue
        latest = feature_frame.iloc[pos].to_dict()
        runtime = _build_fast_runtime_from_row(
            latest,
            mode=mode,
            x_prob=x_probabilities.get(pos, 0.5),
            b_prob=b_probabilities.get(pos, 0.5),
            regime_probs=regime_probability_rows.get(pos, [0.0] * 6),
            equity=equity,
        )
        if not runtime.get("available", False):
            continue
        action = str(runtime.get("decision_direction", runtime.get("raw_stance", "HOLD"))).upper()
        if action not in {"BUY", "SELL"}:
            continue
        payload = {
            "symbol": str(symbol).upper(),
            "market": {"current_price": float(latest.get("close", 0.0) or 0.0)},
            "technical_analysis": {"atr_14": float(latest.get("atr_14", 0.0) or 0.0)},
            "feeds": {},
        }
        judge = judge_builder(payload, runtime)
        content = dict(judge.get("content", {}) if isinstance(judge.get("content"), dict) else {})
        reference_price = float(latest.get("close", 0.0) or 0.0)
        next_open = float(market_15m.loc[execution_time, "open"])
        stop_loss = content.get("stop_loss")
        take_profit = content.get("take_profit")
        stop_loss_value = _safe_float(stop_loss, 0.0) if stop_loss is not None else 0.0
        take_profit_value = _safe_float(take_profit, 0.0) if take_profit is not None else 0.0
        lot = max(_safe_float((runtime.get("lepl_features") or {}).get("suggested_lot"), 0.01), 0.01)
        records.append(
            {
                "signal_id": signal_id,
                "symbol": str(symbol).upper(),
                "mode": str(mode).lower(),
                "signal_time_utc": signal_time.isoformat(),
                "execution_time_utc": execution_time.isoformat(),
                "action": action,
                "lot": round(float(lot), 4),
                "reference_close": round(reference_price, 5),
                "execution_open": round(next_open, 5),
                "stop_loss": round(stop_loss_value, 5) if stop_loss else 0.0,
                "take_profit": round(take_profit_value, 5) if take_profit else 0.0,
                "stop_pips": round(abs(reference_price - stop_loss_value) / max(float(pip_size), 1e-9), 3) if stop_loss else 0.0,
                "take_profit_pips": round(abs(take_profit_value - reference_price) / max(float(pip_size), 1e-9), 3) if take_profit else 0.0,
                "confidence_tier": str(runtime.get("confidence_tier", "very_low")),
                "sqt_label": str(runtime.get("sqt_label", "NEUTRAL")),
                "cabr_score": round(_safe_float(runtime.get("cabr_score"), 0.0), 6),
                "cpm_score": round(_safe_float(runtime.get("cpm_score"), 0.0), 6),
                "conformal_confidence": round(_safe_float((runtime.get("lepl_features") or {}).get("conformal_confidence"), 0.0), 6),
                "kelly_fraction": round(_safe_float((runtime.get("lepl_features") or {}).get("kelly_fraction"), 0.0), 6),
                "dangerous_branch_count": int(_safe_float(runtime.get("v21_dangerous_branch_count"), 0.0)),
                "branch_label": str(runtime.get("selected_branch_label", "")),
                "execution_reason": str(runtime.get("execution_reason", "")),
                "final_summary": str(content.get("final_summary", "")),
            }
        )
        signal_id += 1
    return pd.DataFrame.from_records(records)


def _build_payload(symbol: str, history: pd.DataFrame, equity: float) -> dict[str, Any]:
    candles = _candles_from_frame(history)
    current_price = float(history.iloc[-1]["close"]) if not history.empty else 0.0
    return {
        "symbol": str(symbol).upper(),
        "market": {"current_price": current_price, "candles": candles},
        "realtime_chart": {"candles": candles},
        "paper_trading": {"summary": {"equity": float(equity)}},
    }


def build_v21_mt5_signal_rows(
    raw_15m: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    symbol: str = "XAUUSD",
    mode: str = "precision",
    lookback_bars: int = 240,
    equity: float = 1000.0,
    pip_size: float = 0.1,
    runtime_builder: Callable[..., dict[str, Any]] = build_v21_runtime_state,
    judge_builder: JudgeBuilder = build_v21_local_judge,
) -> pd.DataFrame:
    frame = _ensure_utc_index(raw_15m)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    if V21_FEATURES_PATH.exists() and V21_XLSTM_MODEL_PATH.exists() and V21_BIMAMBA_MODEL_PATH.exists():
        try:
            fast_rows = _fast_signal_rows_from_precomputed_features(
                frame,
                start=start_ts,
                end=end_ts,
                symbol=symbol,
                mode=mode,
                lookback_bars=lookback_bars,
                equity=equity,
                pip_size=pip_size,
                judge_builder=judge_builder,
            )
            if not fast_rows.empty:
                return fast_rows
        except Exception:
            pass

    records: list[dict[str, Any]] = []
    signal_id = 1
    for index in range(len(frame) - 1):
        signal_time = frame.index[index]
        execution_time = frame.index[index + 1]
        if signal_time < start_ts or signal_time >= end_ts:
            continue
        if execution_time >= end_ts:
            continue
        history = frame.iloc[max(0, index - int(lookback_bars) + 1) : index + 1].copy()
        payload = _build_payload(symbol, history, equity=float(equity))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            runtime = runtime_builder(payload, mode=mode)
        if not runtime.get("available", False):
            continue
        action = str(runtime.get("decision_direction", runtime.get("raw_stance", "HOLD"))).upper()
        if action not in {"BUY", "SELL"}:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            judge = judge_builder(payload, runtime)
        content = dict(judge.get("content", {}) if isinstance(judge.get("content"), dict) else {})
        reference_price = float(history.iloc[-1]["close"])
        next_open = float(frame.iloc[index + 1]["open"])
        stop_loss = content.get("stop_loss")
        take_profit = content.get("take_profit")
        stop_loss_value = _safe_float(stop_loss, 0.0) if stop_loss is not None else 0.0
        take_profit_value = _safe_float(take_profit, 0.0) if take_profit is not None else 0.0
        lot = max(_safe_float((runtime.get("lepl_features") or {}).get("suggested_lot"), 0.01), 0.01)
        records.append(
            {
                "signal_id": signal_id,
                "symbol": str(symbol).upper(),
                "mode": str(mode).lower(),
                "signal_time_utc": signal_time.isoformat(),
                "execution_time_utc": execution_time.isoformat(),
                "action": action,
                "lot": round(float(lot), 4),
                "reference_close": round(reference_price, 5),
                "execution_open": round(next_open, 5),
                "stop_loss": round(stop_loss_value, 5) if stop_loss else 0.0,
                "take_profit": round(take_profit_value, 5) if take_profit else 0.0,
                "stop_pips": round(abs(reference_price - stop_loss_value) / max(float(pip_size), 1e-9), 3) if stop_loss else 0.0,
                "take_profit_pips": round(abs(take_profit_value - reference_price) / max(float(pip_size), 1e-9), 3) if take_profit else 0.0,
                "confidence_tier": str(runtime.get("confidence_tier", "very_low")),
                "sqt_label": str(runtime.get("sqt_label", "NEUTRAL")),
                "cabr_score": round(_safe_float(runtime.get("cabr_score"), 0.0), 6),
                "cpm_score": round(_safe_float(runtime.get("cpm_score"), 0.0), 6),
                "conformal_confidence": round(_safe_float((runtime.get("lepl_features") or {}).get("conformal_confidence"), 0.0), 6),
                "kelly_fraction": round(_safe_float((runtime.get("lepl_features") or {}).get("kelly_fraction"), 0.0), 6),
                "dangerous_branch_count": int(_safe_float(runtime.get("v21_dangerous_branch_count"), 0.0)),
                "branch_label": str(runtime.get("selected_branch_label", "")),
                "execution_reason": str(runtime.get("execution_reason", "")),
                "final_summary": str(content.get("final_summary", "")),
            }
        )
        signal_id += 1
    return pd.DataFrame.from_records(records)


def summarize_v21_mt5_signal_rows(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "avg_lot": 0.0,
            "avg_cabr": 0.0,
            "avg_cpm": 0.0,
        }
    action_counts = rows["action"].value_counts().to_dict()
    return {
        "signals": int(len(rows)),
        "buy_signals": int(action_counts.get("BUY", 0)),
        "sell_signals": int(action_counts.get("SELL", 0)),
        "avg_lot": round(float(pd.to_numeric(rows["lot"], errors="coerce").fillna(0.0).mean()), 6),
        "avg_cabr": round(float(pd.to_numeric(rows["cabr_score"], errors="coerce").fillna(0.0).mean()), 6),
        "avg_cpm": round(float(pd.to_numeric(rows["cpm_score"], errors="coerce").fillna(0.0).mean()), 6),
        "first_execution_time_utc": str(rows["execution_time_utc"].iloc[0]),
        "last_execution_time_utc": str(rows["execution_time_utc"].iloc[-1]),
    }


def export_v21_mt5_tester_bridge(
    *,
    month: str,
    symbol: str = "XAUUSD",
    mode: str = "precision",
    lookback_days: int = 90,
    lookback_bars: int = 240,
    equity: float = 1000.0,
    pip_size: float = 0.1,
    csv_path: Path = V21_MT5_TESTER_SIGNALS_PATH,
    summary_path: Path = V21_MT5_TESTER_SUMMARY_PATH,
    copy_to_common: bool = False,
) -> dict[str, Any]:
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    raw = load_default_raw_bars(start=month_start - pd.Timedelta(days=int(lookback_days)), end=month_end + pd.Timedelta(days=1))
    raw_15m = raw.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    rows = build_v21_mt5_signal_rows(
        raw_15m,
        start=month_start,
        end=month_end,
        symbol=symbol,
        mode=mode,
        lookback_bars=lookback_bars,
        equity=equity,
        pip_size=pip_size,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(csv_path, index=False)
    summary = {
        "bridge": "v21_mt5_tester",
        "month": month,
        "symbol": str(symbol).upper(),
        "mode": str(mode).lower(),
        "equity_reference": float(equity),
        "lookback_days": int(lookback_days),
        "lookback_bars": int(lookback_bars),
        "csv_path": str(csv_path),
        **summarize_v21_mt5_signal_rows(rows),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if copy_to_common:
        MT5_COMMON_FILES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, MT5_COMMON_FILES_DIR / csv_path.name)
        summary["mt5_common_copy"] = str(MT5_COMMON_FILES_DIR / csv_path.name)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "build_v21_mt5_signal_rows",
    "export_v21_mt5_tester_bridge",
    "summarize_v21_mt5_signal_rows",
]
