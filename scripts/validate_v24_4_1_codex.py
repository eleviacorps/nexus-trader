from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_DIR, V21_FEATURES_PATH
from src.v21.mt5_tester_bridge import build_v21_mt5_signal_rows
from src.v24_3.tactical_router import TacticalRouter
from src.v24_4.tactical_router import V24_4TacticalRouter

PIP_SIZE = 0.1
OUTPUT_DIR = OUTPUTS_DIR / "v24_4_1"
V21_SIGNAL_CACHE_DIR = OUTPUTS_DIR / "v21" / "mt5_tester"


@dataclass(frozen=True)
class PeriodWindow:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return float(number)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _ensure_utc(timestamp: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(timestamp)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _available_feature_columns() -> set[str]:
    frame = pd.read_parquet(V21_FEATURES_PATH, columns=["close"]).head(1)
    _ = frame
    schema = pd.read_parquet(V21_FEATURES_PATH, engine="pyarrow").columns
    return {str(column) for column in schema}


def _load_feature_frame(start: pd.Timestamp, end: pd.Timestamp, *, prelude_days: int = 30) -> pd.DataFrame:
    requested_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "future_return_15m",
        "target_up_15m",
        "atr_pct",
        "macro_realized_vol_20",
        "macro_vol_regime_class",
        "return_3",
        "return_12",
        "hmm_state_name",
        "hmm_state",
    ]
    available = _available_feature_columns()
    selected = [column for column in requested_columns if column in available]
    frame = pd.read_parquet(V21_FEATURES_PATH, columns=selected).copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    lead_start = start - pd.Timedelta(days=int(prelude_days))
    return frame.loc[(frame.index >= lead_start) & (frame.index < end)].copy()


def _default_windows() -> list[PeriodWindow]:
    frame = pd.read_parquet(V21_FEATURES_PATH, columns=["close"]).copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    latest_end = pd.Timestamp(frame.index.max()).floor("15min")
    latest_start = latest_end - pd.Timedelta(days=30)
    return [
        PeriodWindow("2023-12", pd.Timestamp("2023-12-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
        PeriodWindow("2024-12", pd.Timestamp("2024-12-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
        PeriodWindow(
            f"latest_30d_{latest_start.date()}_{latest_end.date()}",
            latest_start,
            latest_end,
        ),
    ]


def _signal_cache_path(label: str) -> Path:
    if label == "2023-12":
        return V21_SIGNAL_CACHE_DIR / "v21_mt5_tester_signals_2023_12.csv"
    if label == "2024-12":
        return V21_SIGNAL_CACHE_DIR / "v21_mt5_tester_signals_2024_12.csv"
    slug = label.replace("-", "_")
    return V21_SIGNAL_CACHE_DIR / f"v21_mt5_tester_signals_{slug}.csv"


def _build_or_load_signals(window: PeriodWindow, feature_frame: pd.DataFrame) -> pd.DataFrame:
    cache_path = _signal_cache_path(window.label)
    if cache_path.exists():
        signals = pd.read_csv(cache_path)
    else:
        raw = feature_frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
        signals = build_v21_mt5_signal_rows(
            raw,
            start=window.start,
            end=window.end,
            symbol="XAUUSD",
            mode="frequency",
            lookback_bars=240,
            equity=1000.0,
            pip_size=PIP_SIZE,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(cache_path, index=False)
    if signals.empty:
        return signals
    signals["signal_time_utc"] = pd.to_datetime(signals["signal_time_utc"], utc=True, errors="coerce")
    signals["execution_time_utc"] = pd.to_datetime(signals["execution_time_utc"], utc=True, errors="coerce")
    signals = signals.loc[(signals["signal_time_utc"] >= window.start) & (signals["signal_time_utc"] < window.end)].copy()
    return signals.sort_values("signal_time_utc").reset_index(drop=True)


def _classify_regime_from_row(row: Mapping[str, Any]) -> str:
    macro_class = _safe_int(row.get("macro_vol_regime_class"), 0)
    state_name = str(row.get("hmm_state_name", "") or "").lower()
    return_3 = _safe_float(row.get("return_3"), 0.0)
    if macro_class >= 3 or "panic" in state_name or "breakout" in state_name:
        return "breakout"
    if "revert" in state_name:
        return "mean_reversion"
    if abs(return_3) >= 0.0012:
        return "liquidity_sweep_reversal"
    return "trend_continuation"


def _enrich_signals(signals: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    columns = [
        "future_return_15m",
        "target_up_15m",
        "atr_pct",
        "macro_realized_vol_20",
        "macro_vol_regime_class",
        "return_3",
        "return_12",
        "hmm_state_name",
    ]
    available = [column for column in columns if column in feature_frame.columns]
    enriched = signals.join(feature_frame[available], on="signal_time_utc", how="left")
    enriched["action"] = enriched["action"].astype(str).str.upper()
    enriched["direction_sign"] = np.where(enriched["action"] == "BUY", 1.0, -1.0)
    enriched["strategic_confidence"] = (
        pd.to_numeric(enriched["cabr_score"], errors="coerce").fillna(0.5)
        + pd.to_numeric(enriched["cpm_score"], errors="coerce").fillna(0.5)
    ) / 2.0
    enriched["rr_ratio"] = np.divide(
        pd.to_numeric(enriched["take_profit_pips"], errors="coerce").fillna(0.0),
        np.maximum(pd.to_numeric(enriched["stop_pips"], errors="coerce").fillna(0.0), 1e-6),
    )
    realized_return = pd.to_numeric(enriched["future_return_15m"], errors="coerce").fillna(0.0) * enriched["direction_sign"]
    stop_distance = pd.to_numeric(enriched["stop_pips"], errors="coerce").fillna(0.0) * PIP_SIZE
    fallback_stop = np.maximum(
        pd.to_numeric(enriched["reference_close"], errors="coerce").fillna(0.0) * pd.to_numeric(enriched.get("atr_pct", 0.001), errors="coerce").fillna(0.001),
        1e-4,
    )
    stop_distance = np.where(stop_distance <= 0.0, fallback_stop, stop_distance)
    enriched["realized_return"] = realized_return
    enriched["realized_r"] = realized_return / np.maximum(stop_distance, 1e-6)
    enriched["duration_minutes"] = (
        pd.to_datetime(enriched["execution_time_utc"], utc=True, errors="coerce")
        - pd.to_datetime(enriched["signal_time_utc"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    enriched["regime_label"] = enriched.apply(_classify_regime_from_row, axis=1)
    return enriched


def _aggregate_metrics(trades: pd.DataFrame, *, candidate_count: int) -> dict[str, Any]:
    trade_count = int(len(trades))
    participation_rate = float(trade_count / max(int(candidate_count), 1))
    if trade_count == 0:
        return {
            "number_of_trades": 0,
            "participation_rate": round(participation_rate, 6),
            "win_rate": 0.0,
            "expectancy_R": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "average_trade_duration": 0.0,
            "total_R": 0.0,
        }
    realized_r = pd.to_numeric(trades["realized_r_scaled"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    wins = float(np.mean(realized_r > 0.0))
    positives = float(realized_r[realized_r > 0.0].sum()) if np.any(realized_r > 0.0) else 0.0
    negatives = float(abs(realized_r[realized_r < 0.0].sum())) if np.any(realized_r < 0.0) else 0.0
    profit_factor = positives / negatives if negatives > 0 else 0.0
    equity_curve = np.cumsum(realized_r)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = running_peak - equity_curve
    max_drawdown = float(drawdown.max()) if drawdown.size else 0.0
    avg_duration = float(pd.to_numeric(trades["duration_minutes"], errors="coerce").fillna(0.0).mean())
    return {
        "number_of_trades": trade_count,
        "participation_rate": round(participation_rate, 6),
        "win_rate": round(wins, 6),
        "expectancy_R": round(float(realized_r.mean()), 6),
        "profit_factor": round(float(profit_factor), 6),
        "max_drawdown": round(max_drawdown, 6),
        "average_trade_duration": round(avg_duration, 3),
        "total_R": round(float(realized_r.sum()), 6),
    }


def _evaluate_v24_1(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    confidence_tiers = {"moderate", "high", "very_high"}
    trades = candidates.loc[
        candidates["confidence_tier"].astype(str).str.lower().isin(confidence_tiers)
        & (pd.to_numeric(candidates["rr_ratio"], errors="coerce").fillna(0.0) >= 1.5)
        & (pd.to_numeric(candidates["strategic_confidence"], errors="coerce").fillna(0.0) >= 0.68)
    ].copy()
    trades["size_multiplier"] = 1.0
    trades["variant_signal"] = trades["action"]
    trades["variant"] = "v24_1"
    trades["realized_r_scaled"] = pd.to_numeric(trades["realized_r"], errors="coerce").fillna(0.0)
    return trades


def _window_ohlcv(feature_frame: pd.DataFrame, timestamp: pd.Timestamp, bars: int = 96) -> pd.DataFrame:
    subset = feature_frame.loc[feature_frame.index <= timestamp].tail(max(10, int(bars)))
    if subset.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    cols = [column for column in ["open", "high", "low", "close", "volume"] if column in subset.columns]
    return subset[cols].copy()


def _evaluate_v24_3(candidates: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    router = TacticalRouter()
    rng = np.random.default_rng(20260415)
    accepted_rows: list[dict[str, Any]] = []
    for row in candidates.to_dict(orient="records"):
        timestamp = _ensure_utc(row.get("signal_time_utc"))
        market_window = _window_ohlcv(feature_frame, timestamp)
        if market_window.empty:
            continue
        strategic_signal = {
            "signal": str(row.get("action", "HOLD")).lower(),
            "confidence": float(np.clip(_safe_float(row.get("strategic_confidence"), 0.5), 0.0, 1.0)),
            "reason": "codex_v24_3_benchmark",
        }
        # Keep deterministic behavior despite random specialist internals.
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        decision = router.route_trade(market_window, strategic_signal)
        final_decision = dict(decision.get("final_decision", {}))
        if not bool(final_decision.get("should_trade", False)):
            continue
        variant_signal = str(final_decision.get("signal", "hold")).upper()
        if variant_signal not in {"BUY", "SELL"}:
            continue
        realized_return = _safe_float(row.get("future_return_15m"), 0.0) * (1.0 if variant_signal == "BUY" else -1.0)
        stop_distance = max(_safe_float(row.get("stop_pips"), 0.0) * PIP_SIZE, 1e-4)
        realized_r = realized_return / stop_distance
        enriched = dict(row)
        enriched["size_multiplier"] = 1.0
        enriched["variant_signal"] = variant_signal
        enriched["variant"] = "v24_3"
        enriched["realized_r_scaled"] = float(realized_r)
        accepted_rows.append(enriched)
    if not accepted_rows:
        return candidates.iloc[0:0].copy()
    return pd.DataFrame.from_records(accepted_rows)


def _tactical_data_from_row(row: Mapping[str, Any]) -> dict[str, float]:
    tactical_probability = float(
        np.clip(
            (0.55 * _safe_float(row.get("conformal_confidence"), 0.0))
            + (0.45 * _safe_float(row.get("strategic_confidence"), 0.0)),
            0.0,
            1.0,
        )
    )
    tradeability = float(
        np.clip(
            (0.65 * _safe_float(row.get("cabr_score"), 0.0))
            + (0.35 * min(_safe_float(row.get("rr_ratio"), 0.0) / 2.0, 1.0)),
            0.0,
            1.0,
        )
    )
    execution_quality = float(
        np.clip(
            1.0 - min(_safe_float(row.get("atr_pct"), 0.0) / 0.0020, 1.0),
            0.0,
            1.0,
        )
    )
    regime_confidence = float(np.clip(1.0 - (_safe_int(row.get("macro_vol_regime_class"), 0) / 4.0), 0.0, 1.0))
    return {
        "tactical_probability": tactical_probability,
        "tradeability": tradeability,
        "execution_quality": execution_quality,
        "regime_confidence": regime_confidence,
    }


def _evaluate_v24_4(
    candidates: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    enable_admission_layer: bool = True,
    enable_cluster_filter: bool = True,
    enable_cooldown_manager: bool = True,
    enable_dynamic_position_sizing: bool = True,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    router = V24_4TacticalRouter()
    accepted_rows: list[dict[str, Any]] = []
    for row in candidates.to_dict(orient="records"):
        timestamp = _ensure_utc(row.get("signal_time_utc"))
        market_window = _window_ohlcv(feature_frame, timestamp)
        if market_window.empty:
            continue
        strategic_signal = {
            "signal": str(row.get("action", "HOLD")).lower(),
            "confidence": float(np.clip(_safe_float(row.get("strategic_confidence"), 0.5), 0.0, 1.0)),
            "reason": "codex_v24_4_benchmark",
        }
        tactical_data = _tactical_data_from_row(row)
        decision = router.route_trade_with_admission(market_window, strategic_signal, tactical_data)
        should_trade = bool(decision.get("should_trade", False))
        if not enable_admission_layer:
            should_trade = True
        if should_trade and enable_cluster_filter:
            cluster_probe = {
                "direction": str(row.get("action", "HOLD")).upper(),
                "entry_price": _safe_float(row.get("reference_close"), 0.0),
                "timestamp": timestamp.to_pydatetime(),
            }
            existing_clusters = [
                {
                    "direction": str(item.get("variant_signal", "HOLD")).upper(),
                    "entry_price": _safe_float(item.get("reference_close"), 0.0),
                    "timestamp": _ensure_utc(item.get("signal_time_utc")).to_pydatetime(),
                }
                for item in accepted_rows
            ]
            if router.trade_cluster_filter.should_filter_trade(cluster_probe, existing_clusters):
                should_trade = False
        if not should_trade:
            continue
        size_multiplier = _safe_float(decision.get("position_size"), 1.0) if enable_dynamic_position_sizing else 1.0
        if enable_cooldown_manager:
            threshold_hint = _safe_float(decision.get("current_threshold"), 0.7)
            admission_score = _safe_float(decision.get("admission_score"), 0.0)
            if admission_score < threshold_hint:
                continue
            router.cooldown_manager.record_trade_result(_safe_float(row.get("realized_r"), 0.0) > 0.0)
        enriched = dict(row)
        enriched["size_multiplier"] = float(size_multiplier)
        enriched["variant_signal"] = str(row.get("action", "HOLD")).upper()
        enriched["variant"] = "v24_4"
        enriched["v24_4_regime"] = str(decision.get("regime", "mean_reversion"))
        enriched["admission_score"] = _safe_float(decision.get("admission_score"), 0.0)
        enriched["realized_r_scaled"] = _safe_float(row.get("realized_r"), 0.0) * float(size_multiplier)
        accepted_rows.append(enriched)
    if not accepted_rows:
        return candidates.iloc[0:0].copy()
    return pd.DataFrame.from_records(accepted_rows)


def _summarize_variant(trades: pd.DataFrame, *, candidate_count: int, label: str) -> dict[str, Any]:
    metrics = _aggregate_metrics(trades, candidate_count=candidate_count)
    return {"window": label, "candidate_count": int(candidate_count), **metrics}


def _merge_trade_tables(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    valid = [part for part in parts if part is not None and not part.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def _variant_file_payload(
    *,
    variant: str,
    assumptions: dict[str, Any],
    windows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant": variant,
        "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
        "assumptions": assumptions,
        "windows": windows,
        "aggregate_metrics": aggregate,
    }


def _regime_breakdown(v24_4_trades: pd.DataFrame) -> dict[str, Any]:
    regime_map = {
        "trend": "trend_continuation",
        "breakout": "breakout",
        "liquidity_sweep": "liquidity_sweep_reversal",
        "mean_reversion": "mean_reversion",
    }
    if v24_4_trades.empty:
        return {
            "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
            "regimes": {},
            "best_regime": None,
            "worst_regime": None,
            "recommendations": ["No V24.4 trades were produced."],
        }
    working = v24_4_trades.copy()
    working["regime_name"] = working["v24_4_regime"].astype(str).str.lower().map(regime_map).fillna("mean_reversion")
    regimes: dict[str, dict[str, Any]] = {}
    for regime_name, group in working.groupby("regime_name"):
        regime_metrics = _aggregate_metrics(group, candidate_count=len(group))
        regimes[str(regime_name)] = {
            "trade_count": int(len(group)),
            "win_rate": regime_metrics["win_rate"],
            "expectancy": regime_metrics["expectancy_R"],
            "drawdown": regime_metrics["max_drawdown"],
        }
    expectancy_items = sorted(regimes.items(), key=lambda item: item[1]["expectancy"])
    best_regime = expectancy_items[-1][0] if expectancy_items else None
    worst_regime = expectancy_items[0][0] if expectancy_items else None
    recommendations: list[str] = []
    for regime_name, payload in regimes.items():
        if _safe_float(payload.get("expectancy"), 0.0) < 0.0:
            recommendations.append(f"Reduce threshold weight for {regime_name} because expectancy < 0.")
    if not recommendations:
        recommendations.append("No negative-expectancy regime detected in V24.4 sample.")
    return {
        "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
        "regimes": regimes,
        "best_regime": best_regime,
        "worst_regime": worst_regime,
        "recommendations": recommendations,
    }


def _component_ablation(
    base_candidates: list[tuple[PeriodWindow, pd.DataFrame, pd.DataFrame]],
) -> dict[str, Any]:
    configs = [
        ("without_admission_layer", {"enable_admission_layer": False}),
        ("without_cluster_filter", {"enable_cluster_filter": False}),
        ("without_cooldown_manager", {"enable_cooldown_manager": False}),
        ("without_dynamic_position_sizing", {"enable_dynamic_position_sizing": False}),
    ]
    base_trades = _merge_trade_tables(
        [_evaluate_v24_4(candidates, frame) for _, candidates, frame in base_candidates]
    )
    base_metrics = _aggregate_metrics(base_trades, candidate_count=sum(len(candidates) for _, candidates, _ in base_candidates))
    report_rows: list[dict[str, Any]] = []
    for name, kwargs in configs:
        trades = _merge_trade_tables(
            [
                _evaluate_v24_4(
                    candidates,
                    frame,
                    enable_admission_layer=kwargs.get("enable_admission_layer", True),
                    enable_cluster_filter=kwargs.get("enable_cluster_filter", True),
                    enable_cooldown_manager=kwargs.get("enable_cooldown_manager", True),
                    enable_dynamic_position_sizing=kwargs.get("enable_dynamic_position_sizing", True),
                )
                for _, candidates, frame in base_candidates
            ]
        )
        metrics = _aggregate_metrics(trades, candidate_count=sum(len(candidates) for _, candidates, _ in base_candidates))
        report_rows.append(
            {
                "configuration": name,
                "metrics": metrics,
                "change_in_expectancy": round(_safe_float(metrics.get("expectancy_R")) - _safe_float(base_metrics.get("expectancy_R")), 6),
                "change_in_participation": round(_safe_float(metrics.get("participation_rate")) - _safe_float(base_metrics.get("participation_rate")), 6),
                "change_in_drawdown": round(_safe_float(metrics.get("max_drawdown")) - _safe_float(base_metrics.get("max_drawdown")), 6),
            }
        )
    return {
        "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
        "baseline_metrics": base_metrics,
        "ablations": report_rows,
    }


def _final_verdict(metric_comparison: dict[str, Any]) -> str:
    v24_3 = metric_comparison.get("variants", {}).get("v24_3", {})
    v24_4 = metric_comparison.get("variants", {}).get("v24_4", {})
    improved = (
        0.15 <= _safe_float(v24_4.get("participation_rate")) <= 0.30
        and (_safe_float(v24_4.get("expectancy_R")) > _safe_float(v24_3.get("expectancy_R")))
        and (_safe_float(v24_4.get("max_drawdown")) <= _safe_float(v24_3.get("max_drawdown")))
    )
    conclusion = "improved" if improved else "not improved"
    return (
        "# V24.4.1 Final Verdict\n\n"
        "V24.3:\n"
        f"    participation = {_safe_float(v24_3.get('participation_rate')):.6f}\n"
        f"    expectancy = {_safe_float(v24_3.get('expectancy_R')):.6f}\n\n"
        "V24.4:\n"
        f"    participation = {_safe_float(v24_4.get('participation_rate')):.6f}\n"
        f"    expectancy = {_safe_float(v24_4.get('expectancy_R')):.6f}\n\n"
        f"Conclusion:\n    {conclusion}\n"
    )


def run_validation() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = _default_windows()

    variant_windows: dict[str, list[dict[str, Any]]] = {"v24_1": [], "v24_3": [], "v24_4": []}
    variant_trades: dict[str, list[pd.DataFrame]] = {"v24_1": [], "v24_3": [], "v24_4": []}
    cached_inputs: list[tuple[PeriodWindow, pd.DataFrame, pd.DataFrame]] = []

    for window in windows:
        frame = _load_feature_frame(window.start, window.end, prelude_days=45)
        signals = _build_or_load_signals(window, frame)
        candidates = _enrich_signals(signals, frame)
        if candidates.empty:
            continue
        cached_inputs.append((window, candidates, frame))
        trades_v24_1 = _evaluate_v24_1(candidates)
        trades_v24_3 = _evaluate_v24_3(candidates, frame)
        trades_v24_4 = _evaluate_v24_4(candidates, frame)

        variant_windows["v24_1"].append(_summarize_variant(trades_v24_1, candidate_count=len(candidates), label=window.label))
        variant_windows["v24_3"].append(_summarize_variant(trades_v24_3, candidate_count=len(candidates), label=window.label))
        variant_windows["v24_4"].append(_summarize_variant(trades_v24_4, candidate_count=len(candidates), label=window.label))

        variant_trades["v24_1"].append(trades_v24_1)
        variant_trades["v24_3"].append(trades_v24_3)
        variant_trades["v24_4"].append(trades_v24_4)

    total_candidates = sum(len(candidates) for _, candidates, _ in cached_inputs)
    aggregates = {
        key: _aggregate_metrics(_merge_trade_tables(value), candidate_count=total_candidates)
        for key, value in variant_trades.items()
    }

    assumptions = {
        "candidate_source": "v21 fast-export signal rows joined with v21 features",
        "execution_assumptions": "15-minute signal horizon, realized R from directional future_return_15m over stop distance",
        "windows": [asdict(window) | {"start": window.start.isoformat(), "end": window.end.isoformat()} for window in windows],
        "pip_size": PIP_SIZE,
    }
    backtest_v24_1 = _variant_file_payload(
        variant="v24_1",
        assumptions=assumptions,
        windows=variant_windows["v24_1"],
        aggregate=aggregates["v24_1"],
    )
    backtest_v24_3 = _variant_file_payload(
        variant="v24_3",
        assumptions=assumptions,
        windows=variant_windows["v24_3"],
        aggregate=aggregates["v24_3"],
    )
    backtest_v24_4 = _variant_file_payload(
        variant="v24_4",
        assumptions=assumptions,
        windows=variant_windows["v24_4"],
        aggregate=aggregates["v24_4"],
    )
    (OUTPUT_DIR / "backtest_v24_1.json").write_text(json.dumps(backtest_v24_1, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "backtest_v24_3.json").write_text(json.dumps(backtest_v24_3, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "backtest_v24_4.json").write_text(json.dumps(backtest_v24_4, indent=2), encoding="utf-8")

    metric_comparison = {
        "generated_at": pd.Timestamp.now(tz=UTC).isoformat(),
        "variants": aggregates,
        "targets_for_v24_4": {
            "participation_range": [0.18, 0.25],
            "win_rate_range": [0.64, 0.70],
            "expectancy_range_R": [0.12, 0.20],
            "max_drawdown_lt": 0.18,
        },
        "window_level": {
            "v24_1": variant_windows["v24_1"],
            "v24_3": variant_windows["v24_3"],
            "v24_4": variant_windows["v24_4"],
        },
    }
    (OUTPUT_DIR / "metric_comparison.json").write_text(json.dumps(metric_comparison, indent=2), encoding="utf-8")

    v24_4_trades = _merge_trade_tables(variant_trades["v24_4"])
    regime_report = _regime_breakdown(v24_4_trades)
    (OUTPUT_DIR / "regime_breakdown.json").write_text(json.dumps(regime_report, indent=2), encoding="utf-8")

    ablation_report = _component_ablation(cached_inputs)
    (OUTPUT_DIR / "component_ablation_report.json").write_text(json.dumps(ablation_report, indent=2), encoding="utf-8")

    verdict_markdown = _final_verdict(metric_comparison)
    (OUTPUTS_DIR / "V24_4_1_final_verdict.md").write_text(verdict_markdown, encoding="utf-8")

    return {
        "backtest_v24_1": backtest_v24_1,
        "backtest_v24_3": backtest_v24_3,
        "backtest_v24_4": backtest_v24_4,
        "metric_comparison": metric_comparison,
        "regime_breakdown": regime_report,
        "component_ablation_report": ablation_report,
    }


if __name__ == "__main__":
    payload = run_validation()
    print(json.dumps({"v24_4_1_outputs": str(OUTPUT_DIR), "variants": payload["metric_comparison"]["variants"]}, indent=2))
