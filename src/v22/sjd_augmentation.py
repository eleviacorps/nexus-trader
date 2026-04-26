from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.project_config import OUTPUTS_V22_DIR, V19_SJD_DATASET_PATH, V19_SJD_FEATURE_NAMES_PATH, V21_FEATURES_PATH
from src.v19.context_sampler import context_to_feature_vector

_SIGNAL_CACHE_DIR = Path("outputs") / "v21" / "mt5_tester"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return float(number)
    except Exception:
        return float(default)


def _confidence_rank(raw: Any) -> int:
    mapping = {"very_low": 0, "low": 1, "moderate": 2, "high": 3, "very_high": 4}
    return mapping.get(str(raw or "very_low").strip().lower(), 0)


def _month_window(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def _signal_cache_path(month: str) -> Path:
    generic = _SIGNAL_CACHE_DIR / f"v21_mt5_tester_signals_{month.replace('-', '_')}.csv"
    if generic.exists():
        return generic
    default_2023 = _SIGNAL_CACHE_DIR / "v21_mt5_tester_signals.csv"
    if month == "2023-12" and default_2023.exists():
        return default_2023
    return generic


def _feature_frame(month: str, *, prelude_days: int = 90) -> pd.DataFrame:
    start, end = _month_window(month)
    columns = [
        "close",
        "atr_pct",
        "return_3",
        "return_12",
        "macro_vol_regime_class",
        "hmm_state_name",
        "future_return_15m",
        "target_up_15m",
    ]
    frame = pd.read_parquet(V21_FEATURES_PATH, columns=columns)
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    lead_start = start - pd.Timedelta(days=max(1, int(prelude_days)))
    return frame.loc[(frame.index >= lead_start) & (frame.index < end)].copy()


def _build_context(row: Mapping[str, Any], *, stance: str, confidence: str, current_price: float) -> dict[str, Any]:
    direction = str(stance).upper()
    bullish = direction == "BUY"
    stop_pips = max(_safe_float(row.get("stop_pips"), max(current_price * 0.01, 20.0)), 1.0)
    take_pips = max(_safe_float(row.get("take_profit_pips"), stop_pips * 1.5), 1.0)
    atr_14 = max(_safe_float(row.get("atr_pct"), 0.0) * current_price, stop_pips * 0.1, 0.1)
    regime = str(row.get("hmm_state_name", row.get("regime_state", "unknown")) or "unknown")
    sqt_label = str(row.get("sqt_label", "CAUTION")).upper()
    cpm_score = _safe_float(row.get("cpm_score"), 0.35)
    cabr_score = _safe_float(row.get("cabr_score"), 0.35)
    crowd_disagreement = min(1.0, max(0.0, 0.15 + _safe_float(row.get("mfg_disagreement"), 0.0)))
    return {
        "market": {
            "current_price": round(float(current_price), 5),
            "atr_14": round(float(atr_14), 5),
        },
        "simulation": {
            "direction": direction if direction in {"BUY", "SELL"} else "HOLD",
            "overall_confidence": 0.70 if str(confidence).upper() == "HIGH" else 0.28,
            "cabr_score": round(cabr_score, 6),
            "cpm_score": round(cpm_score, 6),
            "cone_width_pips": round(take_pips, 3),
            "detected_regime": regime,
            "sqt_label": sqt_label,
        },
        "technical_analysis": {
            "structure": "bullish" if bullish else "bearish" if direction == "SELL" else "neutral",
            "location": "discount" if bullish else "premium" if direction == "SELL" else "equilibrium",
        },
        "bot_swarm": {
            "aggregate": {
                "signal": "bullish" if bullish else "bearish" if direction == "SELL" else "neutral",
                "disagreement": round(crowd_disagreement, 6),
            }
        },
        "sqt": {"label": sqt_label},
        "mfg": {"disagreement": round(crowd_disagreement, 6)},
    }


def _dataset_row(
    *,
    feature_names: Sequence[str],
    timestamp: Any,
    source: str,
    teacher_model: str,
    context: Mapping[str, Any],
    stance: str,
    confidence: str,
    tp_offset: float,
    sl_offset: float,
    regime: str,
    sqt_label: str,
    cabr_score: float,
    mfg_disagreement: float,
    hurst_overall: float,
) -> dict[str, Any]:
    vector, _ = context_to_feature_vector(context, feature_names)
    return {
        "timestamp": str(pd.Timestamp(timestamp)),
        "source": source,
        "teacher_model": teacher_model,
        "feature_vector": json.dumps([round(float(item), 8) for item in vector.tolist()]),
        "stance": str(stance).upper(),
        "confidence": str(confidence).upper(),
        "entry_offset": 0.0,
        "tp_offset": round(float(tp_offset), 6),
        "sl_offset": round(float(sl_offset), 6),
        "regime": str(regime),
        "sqt_label": str(sqt_label).upper(),
        "cabr_score": round(float(cabr_score), 6),
        "mfg_disagreement": round(float(mfg_disagreement), 6),
        "hurst_overall": round(float(hurst_overall), 6),
    }


def build_augmented_sjd_dataset(
    months: Sequence[str],
    *,
    output_path: Path | None = None,
    report_path: Path | None = None,
    max_negative_per_month: int = 60,
    max_positive_per_month: int = 20,
) -> dict[str, Any]:
    feature_names = json.loads(V19_SJD_FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    base = pd.read_parquet(V19_SJD_DATASET_PATH).copy()
    augmented_rows: list[dict[str, Any]] = []
    month_reports: list[dict[str, Any]] = []

    for month in months:
        signal_path = _signal_cache_path(month)
        if not signal_path.exists():
            continue
        frame = _feature_frame(month)
        start, _ = _month_window(month)
        prelude = frame.loc[frame.index < start].copy()
        if prelude.empty:
            prelude = frame.copy()
        atr_cap = float(pd.to_numeric(prelude["atr_pct"], errors="coerce").fillna(0.0).quantile(0.80))
        return3_floor = float(pd.to_numeric(prelude["return_3"], errors="coerce").fillna(0.0).quantile(0.20))

        signals = pd.read_csv(signal_path)
        signals["signal_time_utc"] = pd.to_datetime(signals["signal_time_utc"], utc=True, errors="coerce")
        joined = signals.join(frame, on="signal_time_utc", how="left")
        joined["confidence_rank"] = joined["confidence_tier"].map(_confidence_rank).astype(int)
        joined["pnl_proxy"] = np.where(
            joined["action"].astype(str).str.upper().eq("BUY"),
            pd.to_numeric(joined["future_return_15m"], errors="coerce").fillna(0.0),
            -pd.to_numeric(joined["future_return_15m"], errors="coerce").fillna(0.0),
        )

        risky = joined.loc[
            (joined["confidence_rank"] >= 2)
            & (
                (pd.to_numeric(joined["macro_vol_regime_class"], errors="coerce").fillna(0).astype(int) == 3)
                | (pd.to_numeric(joined["atr_pct"], errors="coerce").fillna(0.0) > atr_cap)
                | (pd.to_numeric(joined["return_3"], errors="coerce").fillna(0.0) < return3_floor)
            )
        ].copy()
        risky = risky.sort_values(["confidence_rank", "cabr_score", "cpm_score"], ascending=[False, False, False]).head(int(max_negative_per_month))
        for row in risky.to_dict(orient="records"):
            current_price = _safe_float(row.get("reference_close"), _safe_float(row.get("close"), 0.0))
            context = _build_context(row, stance="HOLD", confidence="LOW", current_price=current_price)
            augmented_rows.append(
                _dataset_row(
                    feature_names=feature_names,
                    timestamp=row.get("signal_time_utc"),
                    source=f"v22_month_risk:{month}",
                    teacher_model="v22_hybrid_risk_labeler",
                    context=context,
                    stance="HOLD",
                    confidence="LOW",
                    tp_offset=_safe_float(row.get("take_profit_pips"), 0.0),
                    sl_offset=-abs(_safe_float(row.get("stop_pips"), 0.0)),
                    regime=str(row.get("hmm_state_name", "unknown")),
                    sqt_label=str(row.get("sqt_label", "CAUTION")),
                    cabr_score=_safe_float(row.get("cabr_score"), 0.0),
                    mfg_disagreement=0.20,
                    hurst_overall=0.50,
                )
            )

        positive = joined.loc[
            (joined["confidence_rank"] >= 3)
            & (pd.to_numeric(joined["macro_vol_regime_class"], errors="coerce").fillna(0).astype(int) != 3)
            & (pd.to_numeric(joined["atr_pct"], errors="coerce").fillna(0.0) <= atr_cap)
            & (pd.to_numeric(joined["return_3"], errors="coerce").fillna(0.0) >= return3_floor)
            & (pd.to_numeric(joined["pnl_proxy"], errors="coerce").fillna(0.0) > 0.0)
        ].copy()
        positive = positive.sort_values(["pnl_proxy", "confidence_rank"], ascending=[False, False]).head(int(max_positive_per_month))
        for row in positive.to_dict(orient="records"):
            stance = str(row.get("action", "HOLD")).upper()
            current_price = _safe_float(row.get("reference_close"), _safe_float(row.get("close"), 0.0))
            context = _build_context(row, stance=stance, confidence="HIGH", current_price=current_price)
            augmented_rows.append(
                _dataset_row(
                    feature_names=feature_names,
                    timestamp=row.get("signal_time_utc"),
                    source=f"v22_month_positive:{month}",
                    teacher_model="v22_hybrid_risk_labeler",
                    context=context,
                    stance=stance,
                    confidence="HIGH",
                    tp_offset=_safe_float(row.get("take_profit_pips"), 0.0),
                    sl_offset=-abs(_safe_float(row.get("stop_pips"), 0.0)),
                    regime=str(row.get("hmm_state_name", "unknown")),
                    sqt_label=str(row.get("sqt_label", "GOOD")),
                    cabr_score=_safe_float(row.get("cabr_score"), 0.0),
                    mfg_disagreement=0.10,
                    hurst_overall=0.56,
                )
            )

        month_reports.append(
            {
                "month": month,
                "risky_rows_added": int(len(risky)),
                "positive_rows_added": int(len(positive)),
                "atr_cap": round(atr_cap, 6),
                "return3_floor": round(return3_floor, 6),
            }
        )

    live_diagnostic_path = OUTPUTS_V22_DIR / "live_session_diagnostic_2026_04_10.json"
    live_added = 0
    if live_diagnostic_path.exists():
        payload = json.loads(live_diagnostic_path.read_text(encoding="utf-8"))
        for trade in payload.get("trades", []):
            rr_pretrade = _safe_float(trade.get("rr_pretrade"), 0.0)
            profit = _safe_float(trade.get("profit"), 0.0)
            direction = str(trade.get("direction", "HOLD")).upper()
            if rr_pretrade >= 1.5 and not (direction == "SELL" and profit < 0.0):
                continue
            current_price = _safe_float(trade.get("entry_price"), 0.0)
            pseudo_row = {
                "stop_pips": abs(_safe_float(trade.get("entry_price"), 0.0) - _safe_float(trade.get("stop_loss"), current_price)) / 0.1 if trade.get("stop_loss") is not None else 20.0,
                "take_profit_pips": abs(_safe_float(trade.get("take_profit"), current_price) - current_price) / 0.1 if trade.get("take_profit") is not None else 30.0,
                "atr_pct": abs(_safe_float(trade.get("entry_price"), 0.0) - _safe_float(trade.get("stop_loss"), current_price)) / max(current_price, 1e-6),
                "hmm_state_name": str(trade.get("regime_state", "unknown")),
                "cabr_score": 0.32,
                "cpm_score": 0.28,
                "sqt_label": "CAUTION",
                "reference_close": current_price,
                "close": current_price,
            }
            context = _build_context(pseudo_row, stance="HOLD", confidence="LOW", current_price=current_price)
            augmented_rows.append(
                _dataset_row(
                    feature_names=feature_names,
                    timestamp=trade.get("entry_time"),
                    source="v22_live_autopsy",
                    teacher_model="v22_live_loss_veto",
                    context=context,
                    stance="HOLD",
                    confidence="LOW",
                    tp_offset=_safe_float(pseudo_row["take_profit_pips"], 0.0),
                    sl_offset=-abs(_safe_float(pseudo_row["stop_pips"], 0.0)),
                    regime=str(trade.get("regime_state", "unknown")),
                    sqt_label="CAUTION",
                    cabr_score=0.32,
                    mfg_disagreement=0.25,
                    hurst_overall=0.48,
                )
            )
            live_added += 1

    augmented = pd.concat([base, pd.DataFrame.from_records(augmented_rows)], ignore_index=True)
    augmented = augmented.drop_duplicates(subset=["timestamp", "source", "stance", "feature_vector"], keep="last").reset_index(drop=True)
    output_path = output_path or (OUTPUTS_V22_DIR / "sjd_dataset_v22_augmented.parquet")
    report_path = report_path or (OUTPUTS_V22_DIR / "sjd_dataset_v22_augmented_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(output_path, index=False)
    report = {
        "base_rows": int(len(base)),
        "added_rows": int(len(augmented) - len(base)),
        "final_rows": int(len(augmented)),
        "months": month_reports,
        "live_autopsy_rows_added": int(live_added),
        "output_path": str(output_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["build_augmented_sjd_dataset"]
