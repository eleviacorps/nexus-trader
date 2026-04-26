from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.project_config import V21_FEATURES_PATH
from src.v21.mt5_tester_bridge import build_v21_mt5_signal_rows
from src.v24 import V24EnsembleRiskJudge, build_world_state, load_meta_aggregator
from src.v22.circuit_breaker import CircuitBreakerConfig, DailyCircuitBreaker
from src.v22.ensemble_judge_stack import EnsembleJudgeStack, LinearMetaLabeler
from src.v22.online_hmm import OnlineHMMRegimeDetector, calibrate_confidence_threshold

_SIGNAL_CACHE_DIR = Path("outputs") / "v21" / "mt5_tester"
_MONTH_SIGNAL_STEM = "v21_mt5_tester_signals_{month}.csv"

_CONFIDENCE_RANK = {
    "very_low": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}


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


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def _profit_factor(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    gross_positive = float(values[values > 0.0].sum()) if np.any(values > 0.0) else 0.0
    gross_negative = float(abs(values[values < 0.0].sum())) if np.any(values < 0.0) else 0.0
    if gross_negative <= 0.0:
        return 0.0
    return float(gross_positive / gross_negative)


def _sharpe_like(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float((values.mean() / std) * np.sqrt(len(values)))


def _ensure_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working.index = pd.to_datetime(working.index, utc=True, errors="coerce")
    return working.loc[~working.index.isna()].sort_index()


def _month_window(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return start, end


def _signal_cache_path(month: str) -> Path:
    return _SIGNAL_CACHE_DIR / _MONTH_SIGNAL_STEM.format(month=month.replace("-", "_"))


def _load_feature_frame(month: str, *, prelude_days: int) -> pd.DataFrame:
    start, end = _month_window(month)
    columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "target_up_15m",
        "future_return_15m",
        "atr_pct",
        "macro_realized_vol_20",
        "macro_vol_regime_class",
        "macro_jump_flag",
        "return_1",
        "return_3",
        "return_12",
        "hmm_state",
        "hmm_state_name",
        "hmm_prob_0",
        "hmm_prob_1",
        "hmm_prob_2",
        "hmm_prob_3",
        "hmm_prob_4",
        "hmm_prob_5",
    ]
    frame = pd.read_parquet(V21_FEATURES_PATH, columns=columns)
    frame = _ensure_utc_index(frame)
    lead_start = start - pd.Timedelta(days=max(1, int(prelude_days)))
    return frame.loc[(frame.index >= lead_start) & (frame.index < end)].copy()


def _build_or_load_base_signals(month: str, feature_frame: pd.DataFrame) -> pd.DataFrame:
    cache_path = _signal_cache_path(month)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached["signal_time_utc"] = pd.to_datetime(cached["signal_time_utc"], utc=True, errors="coerce")
        cached["execution_time_utc"] = pd.to_datetime(cached["execution_time_utc"], utc=True, errors="coerce")
        return cached.sort_values("signal_time_utc").reset_index(drop=True)
    default_2023_path = _SIGNAL_CACHE_DIR / "v21_mt5_tester_signals.csv"
    if month == "2023-12" and default_2023_path.exists():
        cached = pd.read_csv(default_2023_path)
        cached["signal_time_utc"] = pd.to_datetime(cached["signal_time_utc"], utc=True, errors="coerce")
        cached["execution_time_utc"] = pd.to_datetime(cached["execution_time_utc"], utc=True, errors="coerce")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache_path, index=False)
        return cached.sort_values("signal_time_utc").reset_index(drop=True)

    start, end = _month_window(month)
    raw = feature_frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    rows = build_v21_mt5_signal_rows(
        raw,
        start=start,
        end=end,
        symbol="XAUUSD",
        mode="frequency",
        lookback_bars=240,
        equity=1000.0,
        pip_size=0.1,
    )
    rows["signal_time_utc"] = pd.to_datetime(rows["signal_time_utc"], utc=True, errors="coerce")
    rows["execution_time_utc"] = pd.to_datetime(rows["execution_time_utc"], utc=True, errors="coerce")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(cache_path, index=False)
    return rows.sort_values("signal_time_utc").reset_index(drop=True)


def _derive_hmm_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    close = pd.to_numeric(working["close"], errors="coerce").ffill().bfill()
    log_return = np.log(close.clip(lower=1e-9)).diff().fillna(0.0)
    volume = pd.to_numeric(working["volume"], errors="coerce").ffill().bfill().fillna(0.0)
    volume_mean = volume.rolling(96, min_periods=24).mean()
    volume_std = volume.rolling(96, min_periods=24).std(ddof=0).replace(0.0, np.nan)
    volume_zscore = ((volume - volume_mean) / volume_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame(
        {
            "log_return": log_return.astype(np.float64),
            "realized_vol_20": pd.to_numeric(working.get("macro_realized_vol_20"), errors="coerce").fillna(log_return.abs()).astype(np.float64),
            "volume_zscore": volume_zscore.astype(np.float64),
            "macro_vol_regime_class": pd.to_numeric(working.get("macro_vol_regime_class"), errors="coerce").fillna(0.0).astype(np.float64),
            "macro_jump_flag": pd.to_numeric(working.get("macro_jump_flag"), errors="coerce").fillna(0.0).astype(np.float64),
        },
        index=working.index,
    )


def _online_hmm_frame(feature_frame: pd.DataFrame, *, month_start: pd.Timestamp, config: "V22DebugConfig") -> tuple[pd.DataFrame, float]:
    detector = OnlineHMMRegimeDetector(momentum=float(config.hmm_momentum))
    inputs = _derive_hmm_inputs(feature_frame)
    prelude_confidences: list[float] = []
    snapshots: list[dict[str, Any]] = []
    for timestamp, row in inputs.iterrows():
        price_now = _safe_float(feature_frame.loc[timestamp, "close"], 0.0)
        atr_pct = _safe_float(feature_frame.loc[timestamp, "atr_pct"], 0.0)
        detector.update(row.to_dict(), price=price_now)
        if timestamp < month_start:
            prelude_confidences.append(detector.regime_confidence)
        snapshots.append(
            {
                "signal_time_utc": pd.Timestamp(timestamp),
                "online_hmm_regime_label": detector.regime_label,
                "online_hmm_regime_confidence": round(detector.regime_confidence, 6),
                "online_hmm_persistence_count": int(detector.persistence_count),
                "online_hmm_lot_size_multiplier": round(detector.runtime_flags(current_price=price_now, atr_14=max(price_now * atr_pct, 0.1)).lot_size_multiplier, 6),
            }
        )
    threshold = calibrate_confidence_threshold(
        prelude_confidences,
        quantile=float(config.hmm_confidence_quantile),
        minimum=float(config.hmm_confidence_min),
        maximum=float(config.hmm_confidence_max),
    )
    hmm_frame = pd.DataFrame.from_records(snapshots)
    return hmm_frame, float(threshold)


def _empirical_conformal_quantiles(signal_frame: pd.DataFrame) -> list[float]:
    if signal_frame.empty:
        return [0.02, 0.0, -0.03]
    accuracy_buy = float((signal_frame.loc[signal_frame["action"] == "BUY", "target_up_15m"] >= 0.5).mean()) if np.any(signal_frame["action"] == "BUY") else 0.5
    accuracy_sell = float((signal_frame.loc[signal_frame["action"] == "SELL", "target_up_15m"] < 0.5).mean()) if np.any(signal_frame["action"] == "SELL") else 0.5
    buy_adj = float(np.clip((accuracy_buy - 0.5) * 0.10, -0.03, 0.03))
    sell_adj = float(np.clip((accuracy_sell - 0.5) * 0.10, -0.03, 0.03))
    hold_adj = float(np.clip(0.01 + (0.5 * max(0.0, 0.60 - max(accuracy_buy, accuracy_sell))), 0.0, 0.04))
    return [round(buy_adj, 6), round(sell_adj, 6), round(hold_adj, 6)]


def _confidence_value(raw: Any) -> int:
    return _CONFIDENCE_RANK.get(str(raw or "very_low").strip().lower(), 0)


def _signal_dir(action: str) -> int:
    text = str(action).upper()
    if text == "BUY":
        return 1
    if text == "SELL":
        return -1
    return 0


def _current_live_performance(executed_trades: Sequence[Mapping[str, Any]], *, starting_balance: float = 1000.0, now: pd.Timestamp) -> dict[str, Any]:
    recent = list(executed_trades)[-10:]
    rolling_win_rate = float(np.mean([_safe_float(item.get("pnl_proxy"), 0.0) > 0.0 for item in recent])) if recent else 1.0
    consecutive_losses = 0
    for trade in reversed(list(executed_trades)):
        if _safe_float(trade.get("pnl_proxy"), 0.0) < 0.0:
            consecutive_losses += 1
        else:
            break
    same_day = [item for item in executed_trades if pd.Timestamp(item.get("signal_time_utc")).date() == now.date()]
    cumulative_pct = float(sum(_safe_float(item.get("pnl_pct"), 0.0) for item in executed_trades))
    balance = float(starting_balance * (1.0 + cumulative_pct))
    daily_pnl_pct = float(sum(_safe_float(item.get("pnl_pct"), 0.0) for item in same_day))
    daily_pnl = float(starting_balance * daily_pnl_pct)
    equity = float(balance)
    return {
        "balance": round(balance, 6),
        "equity": round(equity, 6),
        "rolling_win_rate_10": round(rolling_win_rate, 6),
        "consecutive_losses": int(consecutive_losses),
        "daily_pnl": round(daily_pnl, 6),
    }


def _rr_ratio(row: Mapping[str, Any]) -> float:
    stop = max(_safe_float(row.get("stop_pips"), 0.0), 1e-6)
    take = _safe_float(row.get("take_profit_pips"), 0.0)
    return float(take / stop) if stop > 0.0 else 0.0


def _circuit_profile(profile: str, *, hmm_threshold: float) -> CircuitBreakerConfig:
    key = str(profile).strip().lower()
    if key == "strict":
        return CircuitBreakerConfig(low_regime_confidence_threshold=float(hmm_threshold))
    return CircuitBreakerConfig(
        arm_after_losses=3,
        consecutive_loss_limit=4,
        consecutive_loss_pause_minutes=20,
        drawdown_limit_pct=-0.03,
        drawdown_pause_hours=12,
        min_rolling_win_rate=0.30,
        rolling_pause_minutes=30,
        low_regime_confidence_threshold=max(float(hmm_threshold) - 0.05, 0.50),
        low_regime_bars_limit=8,
        low_regime_pause_minutes=10,
        reduced_size_on_losses=0.60,
        reduced_size_on_drawdown=0.35,
        reduced_size_on_regime=0.60,
    )


class _HeuristicStudent:
    def __init__(self, *, buy_bias: float, momentum_weight: float, hold_bias: float, risk_bias: float) -> None:
        self.buy_bias = float(buy_bias)
        self.momentum_weight = float(momentum_weight)
        self.hold_bias = float(hold_bias)
        self.risk_bias = float(risk_bias)

    def __call__(self, series: Any, quant_features: Any) -> dict[str, Any]:
        q = np.asarray(quant_features, dtype=np.float32).reshape(-1)
        action_sign = _safe_float(q[0], 0.0)
        cpm = _safe_float(q[1], 0.5)
        cabr = _safe_float(q[2], 0.5)
        confidence = _safe_float(q[3], 0.0)
        rr_ratio = _safe_float(q[4], 1.0)
        atr_ratio = _safe_float(q[5], 1.0)
        short_momentum = _safe_float(q[6], 0.0)
        medium_momentum = _safe_float(q[7], 0.0)
        hmm_conf = _safe_float(q[8], 0.5)
        rolling_win_rate = _safe_float(q[9], 0.5)
        consecutive_losses = _safe_float(q[10], 0.0)
        macro_vol = _safe_float(q[11], 0.0)

        directional_edge = (0.90 * action_sign) + (0.75 * ((cpm - 0.5) * 2.0)) + (0.55 * ((cabr - 0.5) * 2.0))
        momentum_edge = self.momentum_weight * ((0.85 * short_momentum) + (0.45 * medium_momentum))
        risk_penalty = self.risk_bias * (
            max(0.0, atr_ratio - 1.0)
            + max(0.0, 0.55 - hmm_conf) * 2.5
            + max(0.0, 0.35 - rolling_win_rate) * 3.0
            + (0.45 * consecutive_losses)
            + max(0.0, macro_vol - 2.0)
        )
        buy_score = self.buy_bias + directional_edge + momentum_edge - risk_penalty
        sell_score = -self.buy_bias - directional_edge - momentum_edge - risk_penalty
        hold_score = self.hold_bias + max(0.0, 1.45 - rr_ratio) + risk_penalty + max(0.0, 2.0 - confidence) * 0.30
        risk_score = _sigmoid(risk_penalty + max(0.0, 1.20 - rr_ratio))
        disagree_score = _sigmoid(abs(short_momentum - action_sign) + max(0.0, 0.60 - confidence) + max(0.0, 0.58 - hmm_conf))
        return {
            "action_logits": np.asarray([buy_score, sell_score, hold_score], dtype=np.float32),
            "risk_pred": risk_score,
            "disagree_prob": disagree_score,
        }


@dataclass(frozen=True)
class V22DebugConfig:
    name: str
    mode: str = "ensemble"
    meta_aggregator_preference: str = "auto"
    prelude_days: int = 90
    allowed_confidence_tiers: tuple[str, ...] = ("moderate", "high", "very_high")
    min_confidence_rank: int = 2
    rr_floor: float = 1.5
    cooldown_bars: int = 8
    atr_quantile_cap: float = 0.80
    return3_quantile_floor: float = 0.20
    return12_quantile_floor: float = 0.10
    block_macro_vol_classes: tuple[int, ...] = (3,)
    hmm_momentum: float = 0.95
    hmm_confidence_quantile: float = 0.20
    hmm_confidence_min: float = 0.52
    hmm_confidence_max: float = 0.62
    circuit_profile: str = "relaxed"
    ensemble_risk_threshold: float = 0.72
    ensemble_meta_threshold: float = 0.46
    ensemble_disagreement_threshold: float = 1.05


def default_v22_experiments() -> list[V22DebugConfig]:
    return [
        V22DebugConfig(name="baseline_v21_frequency", mode="baseline", cooldown_bars=1),
        V22DebugConfig(name="v22_hybrid_relaxed", mode="hybrid_only", cooldown_bars=8),
        V22DebugConfig(
            name="v22_ensemble_relaxed",
            mode="ensemble",
            cooldown_bars=4,
            ensemble_risk_threshold=0.75,
            ensemble_disagreement_threshold=1.10,
        ),
    ]


def _signal_enrichment(signal_rows: pd.DataFrame, feature_frame: pd.DataFrame, hmm_frame: pd.DataFrame) -> pd.DataFrame:
    working = signal_rows.copy()
    working["signal_time_utc"] = pd.to_datetime(working["signal_time_utc"], utc=True, errors="coerce")
    feature_columns = [
        "target_up_15m",
        "future_return_15m",
        "atr_pct",
        "macro_realized_vol_20",
        "macro_vol_regime_class",
        "return_3",
        "return_12",
        "hmm_state_name",
    ]
    enriched = working.join(feature_frame[feature_columns], on="signal_time_utc", how="left")
    enriched = enriched.merge(hmm_frame, on="signal_time_utc", how="left")
    enriched["rr_ratio"] = enriched.apply(_rr_ratio, axis=1)
    enriched["signal_dir"] = enriched["action"].map(_signal_dir).astype(int)
    enriched["realized_dir"] = np.where(pd.to_numeric(enriched["target_up_15m"], errors="coerce").fillna(0.0) >= 0.5, 1, -1)
    enriched["pnl_proxy"] = enriched["signal_dir"] * pd.to_numeric(enriched["future_return_15m"], errors="coerce").fillna(0.0)
    enriched["pnl_pct"] = np.divide(
        enriched["pnl_proxy"],
        np.maximum(pd.to_numeric(enriched["reference_close"], errors="coerce").fillna(1.0), 1e-6),
    )
    enriched["confidence_rank"] = enriched["confidence_tier"].map(_confidence_value).astype(int)
    return enriched.sort_values("signal_time_utc").reset_index(drop=True)


def _thresholds_from_prelude(feature_frame: pd.DataFrame, *, month_start: pd.Timestamp, config: V22DebugConfig) -> dict[str, float]:
    prelude = feature_frame.loc[feature_frame.index < month_start].copy()
    if prelude.empty:
        prelude = feature_frame.copy()
    return {
        "atr_cap": float(pd.to_numeric(prelude["atr_pct"], errors="coerce").fillna(0.0).quantile(float(config.atr_quantile_cap))),
        "return3_floor": float(pd.to_numeric(prelude["return_3"], errors="coerce").fillna(0.0).quantile(float(config.return3_quantile_floor))),
        "return12_floor": float(pd.to_numeric(prelude["return_12"], errors="coerce").fillna(0.0).quantile(float(config.return12_quantile_floor))),
    }


def _hard_risk_reasons(
    row: Mapping[str, Any],
    *,
    config: V22DebugConfig,
    thresholds: Mapping[str, float],
    breaker: DailyCircuitBreaker,
    live_performance: Mapping[str, Any],
    timestamp: pd.Timestamp,
) -> tuple[list[str], Mapping[str, Any]]:
    breaker.sync_live_performance(live_performance, timestamp=timestamp)
    breaker.update_regime_confidence(_safe_float(row.get("online_hmm_regime_confidence"), 0.0), timestamp=timestamp)
    breaker_status = breaker.status(timestamp)
    reasons: list[str] = []
    action = str(row.get("action", "HOLD")).upper()
    if action not in {"BUY", "SELL"}:
        reasons.append("runtime_hold")
    if int(_safe_int(row.get("confidence_rank"), 0)) < int(config.min_confidence_rank):
        reasons.append("confidence_tier")
    if str(row.get("confidence_tier", "")).strip().lower() not in set(config.allowed_confidence_tiers):
        reasons.append("confidence_profile")
    if _safe_float(row.get("rr_ratio"), 0.0) < float(config.rr_floor):
        reasons.append("rr_ratio")
    if _safe_int(row.get("macro_vol_regime_class"), -1) in {int(item) for item in config.block_macro_vol_classes}:
        reasons.append("macro_vol_regime")
    if _safe_float(row.get("atr_pct"), 0.0) > float(thresholds.get("atr_cap", 1.0)):
        reasons.append("atr_excess")
    if action == "BUY" and _safe_float(row.get("return_3"), 0.0) < float(thresholds.get("return3_floor", 0.0)):
        reasons.append("short_momentum")
    if action == "SELL" and _safe_float(row.get("return_3"), 0.0) > -float(thresholds.get("return3_floor", 0.0)):
        reasons.append("short_momentum")
    if action == "BUY" and _safe_float(row.get("return_12"), 0.0) < float(thresholds.get("return12_floor", 0.0)):
        reasons.append("persistence_conflict")
    if action == "SELL" and _safe_float(row.get("return_12"), 0.0) > -float(thresholds.get("return12_floor", 0.0)):
        reasons.append("persistence_conflict")
    if (not breaker_status.trading_allowed) or ("daily_drawdown" in breaker_status.reasons):
        reasons.append("circuit_breaker")
    return reasons, breaker.snapshot(timestamp)


def _quant_feature_vector(row: Mapping[str, Any], *, live_performance: Mapping[str, Any], thresholds: Mapping[str, float]) -> np.ndarray:
    action_sign = float(_signal_dir(str(row.get("action", "HOLD"))))
    atr_cap = max(float(thresholds.get("atr_cap", 1.0)), 1e-6)
    confidence_norm = float(_safe_int(row.get("confidence_rank"), 0)) / 4.0
    macro_vol_norm = float(_safe_int(row.get("macro_vol_regime_class"), 0)) / 3.0
    return np.asarray(
        [
            action_sign,
            _safe_float(row.get("cpm_score"), 0.5),
            _safe_float(row.get("cabr_score"), 0.5),
            confidence_norm,
            _safe_float(row.get("rr_ratio"), 1.0),
            _safe_float(row.get("atr_pct"), 0.0) / atr_cap,
            np.tanh(_safe_float(row.get("return_3"), 0.0) * 1000.0),
            np.tanh(_safe_float(row.get("return_12"), 0.0) * 1000.0),
            _safe_float(row.get("online_hmm_regime_confidence"), 0.5),
            _safe_float(live_performance.get("rolling_win_rate_10"), 1.0),
            float(_safe_int(live_performance.get("consecutive_losses"), 0)),
            macro_vol_norm,
        ],
        dtype=np.float32,
    )


def _series_window(frame: pd.DataFrame, *, timestamp: pd.Timestamp, bars: int = 8) -> np.ndarray:
    subset = frame.loc[frame.index <= timestamp].tail(max(1, int(bars)))
    series = np.column_stack(
        [
            pd.to_numeric(subset["return_1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["return_3"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["return_12"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["atr_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["macro_realized_vol_20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["macro_vol_regime_class"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["online_hmm_regime_confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            np.asarray([_signal_dir("BUY")] * len(subset), dtype=np.float32),
        ]
    )
    if len(series) < bars:
        pad = np.zeros((bars - len(series), series.shape[1]), dtype=np.float32)
        series = np.vstack([pad, series])
    return series.astype(np.float32)


def _v24_sequence_window(frame: pd.DataFrame, *, timestamp: pd.Timestamp, direction: str, bars: int = 16) -> np.ndarray:
    subset = frame.loc[frame.index <= timestamp].tail(max(1, int(bars)))
    if subset.empty:
        return np.zeros((int(bars), 8), dtype=np.float32)
    direction_sign = float(_signal_dir(direction))
    sequence = np.column_stack(
        [
            pd.to_numeric(subset["return_1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["return_3"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["return_12"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["atr_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["macro_realized_vol_20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["macro_vol_regime_class"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["online_hmm_regime_confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            np.full(len(subset), direction_sign, dtype=np.float32),
        ]
    ).astype(np.float32)
    if len(sequence) < int(bars):
        pad = np.zeros((int(bars) - len(sequence), sequence.shape[1]), dtype=np.float32)
        sequence = np.vstack([pad, sequence])
    return sequence


def _ensemble_stack(calibrated_signal_frame: pd.DataFrame, config: V22DebugConfig) -> EnsembleJudgeStack:
    conformal_quantiles = _empirical_conformal_quantiles(calibrated_signal_frame)
    meta_model = LinearMetaLabeler(weights=(2.4, -1.6, -0.8, 1.2, -1.1, -0.9), bias=0.05)
    return EnsembleJudgeStack(
        students=[
            _HeuristicStudent(buy_bias=0.40, momentum_weight=0.55, hold_bias=0.25, risk_bias=0.65),
            _HeuristicStudent(buy_bias=0.15, momentum_weight=0.95, hold_bias=0.10, risk_bias=0.45),
            _HeuristicStudent(buy_bias=0.05, momentum_weight=0.30, hold_bias=0.55, risk_bias=0.95),
        ],
        conformal_quantiles=conformal_quantiles,
        meta_model=meta_model,
        risk_threshold=float(config.ensemble_risk_threshold),
        meta_threshold=float(config.ensemble_meta_threshold),
        disagreement_threshold=float(config.ensemble_disagreement_threshold),
    )


def _decision_payload(
    row: Mapping[str, Any],
    *,
    config: V22DebugConfig,
    thresholds: Mapping[str, float],
    breaker: DailyCircuitBreaker,
    live_performance: Mapping[str, Any],
    feature_frame: pd.DataFrame,
    stack: EnsembleJudgeStack | None,
    aggregator: Any | None,
    risk_judge: V24EnsembleRiskJudge | None,
) -> tuple[bool, dict[str, Any]]:
    timestamp = pd.Timestamp(row.get("signal_time_utc"))
    hard_reasons, breaker_snapshot = _hard_risk_reasons(
        row,
        config=config,
        thresholds=thresholds,
        breaker=breaker,
        live_performance=live_performance,
        timestamp=timestamp,
    )
    if config.mode == "baseline":
        return True, {
            "decision": str(row.get("action", "HOLD")).upper(),
            "mode": "baseline",
            "breaker": breaker_snapshot,
            "reasons": [],
        }
    if hard_reasons:
        return False, {
            "decision": "HOLD",
            "mode": config.mode,
            "breaker": breaker_snapshot,
            "reasons": hard_reasons,
        }
    if config.mode == "hybrid_only":
        return True, {
            "decision": str(row.get("action", "HOLD")).upper(),
            "mode": config.mode,
            "breaker": breaker_snapshot,
            "reasons": [],
        }
    quant_features = _quant_feature_vector(row, live_performance=live_performance, thresholds=thresholds)
    series = _series_window(feature_frame, timestamp=timestamp)
    prediction = stack.predict(series, quant_features, context_features=[quant_features[4], quant_features[8], quant_features[9]]) if stack is not None else {"action": "HOLD", "reason": "missing_stack", "confidence": 0.0}
    if config.mode == "v24_bridge":
        world_state = build_world_state(
            row,
            live_performance=live_performance,
            breaker_state=breaker_snapshot,
            ensemble_state=prediction,
        )
        sequence_features = _v24_sequence_window(
            feature_frame,
            timestamp=timestamp,
            direction=str(row.get("action", "HOLD")).upper(),
        )
        quality = (aggregator or load_meta_aggregator(preference=config.meta_aggregator_preference)).predict(
            world_state,
            sequence_features=sequence_features,
        )
        decision = (risk_judge or V24EnsembleRiskJudge()).decide(world_state, quality)
        if decision.action not in {"EXECUTE", "REDUCE_SIZE"}:
            return False, {
                "decision": "HOLD",
                "mode": config.mode,
                "breaker": breaker_snapshot,
                "reasons": [decision.reason],
                "ensemble": prediction,
                "v24": decision.to_dict(),
            }
        if decision.direction != str(row.get("action", "HOLD")).upper():
            return False, {
                "decision": "HOLD",
                "mode": config.mode,
                "breaker": breaker_snapshot,
                "reasons": ["v24_direction_mismatch"],
                "ensemble": prediction,
                "v24": decision.to_dict(),
            }
        return True, {
            "decision": decision.direction,
            "mode": config.mode,
            "breaker": breaker_snapshot,
            "reasons": [],
            "ensemble": prediction,
            "v24": decision.to_dict(),
        }
    if str(prediction.get("action", "HOLD")).upper() != str(row.get("action", "HOLD")).upper():
        return False, {
            "decision": "HOLD",
            "mode": config.mode,
            "breaker": breaker_snapshot,
            "reasons": [str(prediction.get("reason", "ensemble_reject"))],
            "ensemble": prediction,
        }
    return True, {
        "decision": str(prediction.get("action", "HOLD")).upper(),
        "mode": config.mode,
        "breaker": breaker_snapshot,
        "reasons": [],
        "ensemble": prediction,
    }


def run_v22_month_experiment(month: str, config: V22DebugConfig) -> dict[str, Any]:
    month_start, month_end = _month_window(month)
    feature_frame = _load_feature_frame(month, prelude_days=int(config.prelude_days))
    signal_rows = _build_or_load_base_signals(month, feature_frame)
    hmm_frame, hmm_threshold = _online_hmm_frame(feature_frame, month_start=month_start, config=config)
    feature_plus_hmm = feature_frame.join(hmm_frame.set_index("signal_time_utc"), how="left")
    signal_frame = _signal_enrichment(signal_rows, feature_plus_hmm, hmm_frame)
    month_signals = signal_frame.loc[(signal_frame["signal_time_utc"] >= month_start) & (signal_frame["signal_time_utc"] < month_end)].copy()
    thresholds = _thresholds_from_prelude(feature_plus_hmm, month_start=month_start, config=config)
    breaker = DailyCircuitBreaker(_circuit_profile(config.circuit_profile, hmm_threshold=hmm_threshold))
    stack = None if config.mode not in {"ensemble", "v24_bridge"} else _ensemble_stack(month_signals, config)
    aggregator = None if config.mode != "v24_bridge" else load_meta_aggregator(preference=config.meta_aggregator_preference)
    risk_judge = None if config.mode != "v24_bridge" else V24EnsembleRiskJudge()

    executed: list[dict[str, Any]] = []
    skip_breakdown: dict[str, int] = {}
    next_available = pd.Timestamp.min.tz_localize("UTC")
    for row in month_signals.to_dict(orient="records"):
        timestamp = pd.Timestamp(row.get("signal_time_utc"))
        if timestamp < next_available:
            skip_breakdown["cooldown"] = skip_breakdown.get("cooldown", 0) + 1
            continue
        live_performance = _current_live_performance(executed, now=timestamp)
        accepted, payload = _decision_payload(
            row,
            config=config,
            thresholds=thresholds,
            breaker=breaker,
            live_performance=live_performance,
            feature_frame=feature_plus_hmm,
            stack=stack,
            aggregator=aggregator,
            risk_judge=risk_judge,
        )
        if not accepted:
            for reason in payload.get("reasons", []) or ["rejected"]:
                key = str(reason)
                skip_breakdown[key] = skip_breakdown.get(key, 0) + 1
            continue
        trade = dict(row)
        trade["decision"] = payload.get("decision", trade.get("action", "HOLD"))
        trade["v22_mode"] = config.mode
        trade["live_performance"] = dict(live_performance)
        trade["v22_breaker"] = dict(payload.get("breaker", {}))
        if "ensemble" in payload:
            trade["v22_ensemble"] = dict(payload["ensemble"])
        if "v24" in payload:
            trade["v24_decision"] = dict(payload["v24"])
        executed.append(trade)
        pnl_pct = _safe_float(trade.get("pnl_pct"), 0.0)
        breaker.record_trade(
            {
                "pnl": _safe_float(trade.get("pnl_proxy"), 0.0),
                "pnl_pct": pnl_pct,
                "profitable": _safe_float(trade.get("pnl_proxy"), 0.0) > 0.0,
                "timestamp": timestamp,
            },
            timestamp=timestamp,
        )
        next_available = timestamp + pd.Timedelta(minutes=15 * max(1, int(config.cooldown_bars)))

    pnl = np.asarray([_safe_float(item.get("pnl_proxy"), 0.0) for item in executed], dtype=np.float64)
    pnl_pct = np.asarray([_safe_float(item.get("pnl_pct"), 0.0) for item in executed], dtype=np.float64)
    direction = pd.Series([str(item.get("decision", "HOLD")).upper() for item in executed], dtype="object")
    win_rate = float(np.mean(pnl > 0.0)) if pnl.size else 0.0
    metrics = {
        "experiment": config.name,
        "mode": config.mode,
        "month": month,
        "trade_count": int(len(executed)),
        "buy_trades": int((direction == "BUY").sum()) if not direction.empty else 0,
        "sell_trades": int((direction == "SELL").sum()) if not direction.empty else 0,
        "target_trade_band_met": bool(50 <= len(executed) <= 200),
        "win_rate": round(win_rate, 6),
        "accuracy": round(float(np.mean([item.get("signal_dir") == item.get("realized_dir") for item in executed])) if executed else 0.0, 6),
        "avg_return": round(float(pnl.mean()) if pnl.size else 0.0, 6),
        "cumulative_return": round(float(pnl.sum()) if pnl.size else 0.0, 6),
        "avg_return_pct": round(float(pnl_pct.mean()) if pnl_pct.size else 0.0, 6),
        "profit_factor": round(_profit_factor(pnl), 6),
        "sharpe_like": round(_sharpe_like(pnl), 6),
        "avg_rr_ratio": round(float(np.mean([_safe_float(item.get("rr_ratio"), 0.0) for item in executed])) if executed else 0.0, 6),
        "avg_hmm_confidence": round(float(np.mean([_safe_float(item.get("online_hmm_regime_confidence"), 0.0) for item in executed])) if executed else 0.0, 6),
        "avg_cabr_score": round(float(np.mean([_safe_float(item.get("cabr_score"), 0.0) for item in executed])) if executed else 0.0, 6),
        "avg_cpm_score": round(float(np.mean([_safe_float(item.get("cpm_score"), 0.0) for item in executed])) if executed else 0.0, 6),
        "skip_breakdown": skip_breakdown,
        "thresholds": {
            **thresholds,
            "hmm_confidence_floor": round(float(hmm_threshold), 6),
            "cooldown_bars": int(config.cooldown_bars),
            "rr_floor": float(config.rr_floor),
            "circuit_profile": str(config.circuit_profile),
        },
        "base_signal_count": int(len(month_signals)),
        "base_buy_signals": int((month_signals["action"].astype(str).str.upper() == "BUY").sum()),
        "base_sell_signals": int((month_signals["action"].astype(str).str.upper() == "SELL").sum()),
        "trades": executed,
    }
    if executed and any("v22_ensemble" in item for item in executed):
        metrics["ensemble_agreement_rate"] = round(float(np.mean([_safe_float((item.get("v22_ensemble") or {}).get("agreement_rate"), 0.0) for item in executed])), 6)
        metrics["meta_label_accept_rate"] = round(float(np.mean([_safe_float((item.get("v22_ensemble") or {}).get("meta_label_prob"), 0.0) for item in executed])), 6)
    if executed and any("v24_decision" in item for item in executed):
        metrics["avg_expected_value"] = round(float(np.mean([_safe_float(((item.get("v24_decision") or {}).get("quality") or {}).get("expected_value"), 0.0) for item in executed])), 6)
        metrics["avg_profit_probability"] = round(float(np.mean([_safe_float(((item.get("v24_decision") or {}).get("quality") or {}).get("profit_probability"), 0.0) for item in executed])), 6)
        metrics["avg_quality_score"] = round(float(np.mean([_safe_float(((item.get("v24_decision") or {}).get("quality") or {}).get("quality_score"), 0.0) for item in executed])), 6)
        metrics["avg_danger_score"] = round(float(np.mean([_safe_float(((item.get("v24_decision") or {}).get("quality") or {}).get("danger_score"), 0.0) for item in executed])), 6)
        metrics["avg_uncertainty_score"] = round(float(np.mean([_safe_float(((item.get("v24_decision") or {}).get("quality") or {}).get("uncertainty_score"), 0.0) for item in executed])), 6)
        metrics["v24_reduce_size_count"] = int(sum(str((item.get("v24_decision") or {}).get("action", "")).upper() == "REDUCE_SIZE" for item in executed))
    return metrics


def run_v22_month_suite(months: Sequence[str], configs: Sequence[V22DebugConfig] | None = None) -> dict[str, Any]:
    selected_configs = list(configs or default_v22_experiments())
    results: list[dict[str, Any]] = []
    for month in months:
        for config in selected_configs:
            results.append(run_v22_month_experiment(str(month), config))
    return {
        "months": [str(month) for month in months],
        "experiments": [asdict(config) for config in selected_configs],
        "results": results,
    }


__all__ = [
    "V22DebugConfig",
    "default_v22_experiments",
    "run_v22_month_experiment",
    "run_v22_month_suite",
]
