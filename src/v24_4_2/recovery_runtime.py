from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from config.project_config import OUTPUTS_DIR
from src.v24_4_2.adaptive_admission import AdaptiveAdmission, AdmissionCandidate
from src.v24_4_2.regime_threshold_router import RegimeContext, RegimeThresholdRouter
from src.v24_4_2.sell_bias_guard import MarketExecutionContext, SellBiasGuard, StreakContext
from src.v24_4_2.threshold_optimizer import ThresholdConfig

PIP_SIZE = 0.1


@dataclass(frozen=True)
class ValidationWindow:
    label: str
    start: str
    end: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RecoveryValidationResult:
    generated_at: str
    config: dict[str, Any]
    aggregate_metrics: dict[str, Any]
    windows: list[ValidationWindow]
    regime_breakdown: dict[str, Any]
    operational_safety: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "config": self.config,
            "aggregate_metrics": self.aggregate_metrics,
            "windows": [asdict(item) for item in self.windows],
            "regime_breakdown": self.regime_breakdown,
            "operational_safety": self.operational_safety,
        }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return float(number)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _aggregate_metrics(trades: pd.DataFrame, candidate_count: int) -> dict[str, Any]:
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
    return {
        "number_of_trades": trade_count,
        "participation_rate": round(participation_rate, 6),
        "win_rate": round(wins, 6),
        "expectancy_R": round(float(realized_r.mean()), 6),
        "profit_factor": round(float(profit_factor), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "total_R": round(float(realized_r.sum()), 6),
    }


def classify_regime(row: Mapping[str, Any]) -> str:
    hmm_name = str(row.get("hmm_state_name", "") or "").lower()
    macro_class = safe_int(row.get("macro_vol_regime_class"), 0)
    return_12 = safe_float(row.get("return_12"), 0.0)
    return_3 = safe_float(row.get("return_3"), 0.0)
    atr_pct = safe_float(row.get("atr_pct"), 0.0)

    if "breakout" in hmm_name or "panic" in hmm_name or macro_class >= 3:
        return "breakout"
    if atr_pct >= 0.0024 and abs(return_3) < 0.00025:
        return "chop"
    if return_12 >= 0.0008:
        return "trend_up"
    if return_12 <= -0.0008:
        return "trend_down"
    if abs(return_3) <= 0.00035 and macro_class <= 1:
        return "range"
    return "unknown"


def _window_ohlcv(feature_frame: pd.DataFrame, timestamp: pd.Timestamp, bars: int = 96) -> pd.DataFrame:
    subset = feature_frame.loc[feature_frame.index <= timestamp].tail(max(10, int(bars)))
    if subset.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    cols = [column for column in ["open", "high", "low", "close", "volume"] if column in subset.columns]
    return subset[cols].copy()


def _context_probability(row: Mapping[str, Any]) -> float:
    strategic_conf = safe_float(row.get("strategic_confidence"), 0.5)
    cpm = safe_float(row.get("cpm_score"), strategic_conf)
    conformal = safe_float(row.get("conformal_confidence"), strategic_conf)
    cabr = safe_float(row.get("cabr_score"), 0.5)
    base = (0.40 * strategic_conf) + (0.25 * cpm) + (0.20 * conformal) + (0.15 * cabr)
    quality_penalty = 0.08 if cpm < 0.52 else 0.0
    return float(np.clip(base - quality_penalty, 0.0, 1.0))


def _execution_quality(row: Mapping[str, Any], spread_estimate: float, slippage_estimate: float) -> float:
    atr_pct = safe_float(row.get("atr_pct"), 0.001)
    atr_quality = 1.0 - min(atr_pct / 0.0030, 1.0)
    spread_quality = 1.0 - min(spread_estimate / 0.40, 1.0)
    slippage_quality = 1.0 - min(slippage_estimate / 0.20, 1.0)
    return float(np.clip((0.5 * atr_quality) + (0.3 * spread_quality) + (0.2 * slippage_quality), 0.0, 1.0))


def _regime_profitability_signal(regime_expectancy: float, regime_win_rate: float) -> float:
    expectancy_component = np.clip(0.5 + (regime_expectancy * 25.0), 0.0, 1.0)
    win_component = np.clip(regime_win_rate, 0.0, 1.0)
    return float(np.clip((0.6 * expectancy_component) + (0.4 * win_component), 0.0, 1.0))


def _recent_trade_health(trade_outcomes: Iterable[float]) -> float:
    recent = list(trade_outcomes)
    if not recent:
        return 0.55
    wins = float(np.mean(np.asarray(recent, dtype=np.float64) > 0.0))
    equity = np.cumsum(np.asarray(recent, dtype=np.float64))
    running_peak = np.maximum.accumulate(equity)
    drawdown = running_peak - equity
    max_dd = float(drawdown.max()) if drawdown.size else 0.0
    drawdown_health = float(np.clip(1.0 - (max_dd / 0.18), 0.0, 1.0))
    return float(np.clip((0.65 * wins) + (0.35 * drawdown_health), 0.0, 1.0))


def evaluate_candidates_v24_4_2(
    candidates: pd.DataFrame,
    feature_frame: pd.DataFrame,
    config: ThresholdConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if candidates.empty:
        return candidates.copy(), {"sell_guard_blocks": 0, "cooldown_blocks": 0, "cluster_blocks": 0}

    router = RegimeThresholdRouter().with_threshold_overrides(config.as_threshold_map())
    admission = AdaptiveAdmission(router=router)
    sell_guard = SellBiasGuard(
        max_consecutive_sells=3,
        duplicate_cluster_radius=float(config.cluster_radius),
        duplicate_cluster_minutes=12,
    )

    accepted_rows: list[dict[str, Any]] = []
    recent_results = deque(maxlen=20)
    recent_sell_entries = deque(maxlen=10)
    regime_results: dict[str, list[float]] = {}

    cooldown_remaining = 0.0
    consecutive_sell_trades = 0
    consecutive_sell_losses = 0
    sell_guard_blocks = 0
    cooldown_blocks = 0
    cluster_blocks = 0

    for row in candidates.sort_values("signal_time_utc").to_dict(orient="records"):
        raw_timestamp = pd.Timestamp(row.get("signal_time_utc"))
        if raw_timestamp.tzinfo is None:
            timestamp = raw_timestamp.tz_localize("UTC")
        else:
            timestamp = raw_timestamp.tz_convert("UTC")
        market_window = _window_ohlcv(feature_frame, timestamp)
        if market_window.empty:
            continue

        if cooldown_remaining > 0.0:
            cooldown_blocks += 1
            cooldown_remaining = max(0.0, cooldown_remaining - float(config.cooldown_decay))
            continue

        regime_label = classify_regime(row)
        historical_regime = regime_results.get(regime_label, [])
        regime_expectancy = float(np.mean(historical_regime)) if historical_regime else 0.0
        regime_win_rate = float(np.mean(np.asarray(historical_regime) > 0.0)) if historical_regime else 0.52
        regime_confidence = float(
            np.clip(
                (0.50 * safe_float(row.get("strategic_confidence"), 0.5))
                + (0.30 * safe_float(row.get("cpm_score"), 0.5))
                + (0.20 * (1.0 - min(safe_int(row.get("macro_vol_regime_class"), 0) / 4.0, 1.0))),
                0.0,
                1.0,
            )
        )
        regime_ctx = RegimeContext(
            regime=regime_label,
            regime_confidence=regime_confidence,
            rolling_expectancy=regime_expectancy,
            rolling_win_rate=regime_win_rate,
        )

        strategic_direction = str(row.get("action", "HOLD")).upper()
        short_return = safe_float(row.get("return_3"), 0.0)
        if short_return > 0.00015:
            tactical_direction = "BUY"
        elif short_return < -0.00015:
            tactical_direction = "SELL"
        else:
            tactical_direction = strategic_direction
        spread_estimate = max(0.05, min(0.50, safe_float(row.get("atr_pct"), 0.001) * 120.0))
        slippage_estimate = max(0.01, min(0.18, safe_float(row.get("atr_pct"), 0.001) * 45.0))

        strategic_alignment = 1.0 if strategic_direction == tactical_direction and strategic_direction in {"BUY", "SELL"} else 0.25
        candidate_features = AdmissionCandidate(
            calibrated_probability=_context_probability(row),
            tactical_cabr=float(np.clip(safe_float(row.get("cabr_score"), 0.5), 0.0, 1.0)),
            regime_profitability=_regime_profitability_signal(regime_expectancy, regime_win_rate) if historical_regime else 0.45,
            execution_quality=_execution_quality(row, spread_estimate, slippage_estimate),
            strategic_alignment=strategic_alignment,
            recent_trade_health=_recent_trade_health(recent_results),
            strategic_direction=strategic_direction,
            tactical_direction=tactical_direction,
        )
        admission_decision = admission.allow(candidate_features, regime_ctx)
        if not admission_decision.allow:
            continue

        guard_decision = sell_guard.evaluate_sell(
            candidate={
                "direction": tactical_direction,
                "regime_confidence": regime_confidence,
                "tactical_cabr": candidate_features.tactical_cabr,
                "admission_score": admission_decision.admission_score,
                "timestamp": timestamp.to_pydatetime(),
                "entry_price": safe_float(row.get("reference_close"), 0.0),
            },
            market_exec_ctx=MarketExecutionContext(
                spread=spread_estimate,
                slippage_estimate=slippage_estimate,
                buy_threshold=admission_decision.threshold,
            ),
            streak_ctx=StreakContext(
                consecutive_sell_trades=consecutive_sell_trades,
                consecutive_sell_losses=consecutive_sell_losses,
                cooldown_bars=int(max(0, round(cooldown_remaining))),
                recent_sell_entries=tuple(recent_sell_entries),
            ),
        )
        if not guard_decision.allow:
            sell_guard_blocks += 1
            if guard_decision.reason == "sell_cluster_duplicate_blocked":
                cluster_blocks += 1
            cooldown_remaining = max(cooldown_remaining, float(guard_decision.adjusted_cooldown_bars))
            continue

        direction_sign = 1.0 if tactical_direction == "BUY" else -1.0
        realized_return = safe_float(row.get("future_return_15m"), 0.0) * direction_sign
        stop_distance = max(safe_float(row.get("stop_pips"), 0.0) * PIP_SIZE, 1e-4)
        realized_r = realized_return / stop_distance
        size_multiplier = float(config.size_multiplier) * (1.0 + max(0.0, admission_decision.admission_score - admission_decision.threshold))

        enriched = dict(row)
        enriched["variant"] = "v24_4_2"
        enriched["variant_signal"] = tactical_direction
        enriched["regime_label_v24_4_2"] = regime_label
        enriched["regime_confidence_v24_4_2"] = regime_confidence
        enriched["admission_score"] = admission_decision.admission_score
        enriched["admission_threshold"] = admission_decision.threshold
        enriched["admission_reason"] = admission_decision.reason
        enriched["size_multiplier"] = size_multiplier
        enriched["realized_r_scaled"] = float(realized_r * size_multiplier)
        enriched["spread_estimate"] = spread_estimate
        enriched["slippage_estimate"] = slippage_estimate
        accepted_rows.append(enriched)

        recent_results.append(float(enriched["realized_r_scaled"]))
        regime_results.setdefault(regime_label, []).append(float(enriched["realized_r_scaled"]))

        if tactical_direction == "SELL":
            recent_sell_entries.append((timestamp.to_pydatetime(), safe_float(row.get("reference_close"), 0.0)))
            consecutive_sell_trades += 1
            if enriched["realized_r_scaled"] < 0.0:
                consecutive_sell_losses += 1
                if consecutive_sell_losses >= 2:
                    cooldown_remaining = max(cooldown_remaining, 4.0)
            else:
                consecutive_sell_losses = 0
        else:
            consecutive_sell_trades = 0
            consecutive_sell_losses = 0
        cooldown_remaining = max(0.0, cooldown_remaining - float(config.cooldown_decay))

    trades = pd.DataFrame.from_records(accepted_rows) if accepted_rows else candidates.iloc[0:0].copy()
    ops = {
        "sell_guard_blocks": int(sell_guard_blocks),
        "cooldown_blocks": int(cooldown_blocks),
        "cluster_blocks": int(cluster_blocks),
    }
    return trades, ops


def build_validation_result(
    windows_data: list[tuple[Any, pd.DataFrame, pd.DataFrame]],
    config: ThresholdConfig,
) -> RecoveryValidationResult:
    window_summaries: list[ValidationWindow] = []
    all_trades: list[pd.DataFrame] = []
    total_candidates = 0
    ops_accumulator = {"sell_guard_blocks": 0, "cooldown_blocks": 0, "cluster_blocks": 0}

    for window, candidates, frame in windows_data:
        total_candidates += int(len(candidates))
        trades, ops = evaluate_candidates_v24_4_2(candidates, frame, config)
        all_trades.append(trades)
        for key in ops_accumulator:
            ops_accumulator[key] += int(ops.get(key, 0))
        metrics = _aggregate_metrics(trades, candidate_count=len(candidates))
        window_summaries.append(
            ValidationWindow(
                label=str(window.label),
                start=str(window.start.isoformat()),
                end=str(window.end.isoformat()),
                metrics=metrics,
            )
        )

    merged = pd.concat([item for item in all_trades if item is not None and not item.empty], ignore_index=True) if all_trades else pd.DataFrame()
    aggregate = _aggregate_metrics(merged, candidate_count=total_candidates)

    regimes: dict[str, Any] = {}
    if not merged.empty:
        for regime, group in merged.groupby("regime_label_v24_4_2"):
            regimes[str(regime)] = _aggregate_metrics(group, candidate_count=len(group))
    operational_safety = {
        **ops_accumulator,
        "total_candidates": int(total_candidates),
        "total_trades": int(len(merged)),
        "sell_guard_activity_rate": round(float(ops_accumulator["sell_guard_blocks"] / max(total_candidates, 1)), 6),
    }
    return RecoveryValidationResult(
        generated_at=datetime.now(tz=UTC).isoformat(),
        config=asdict(config),
        aggregate_metrics=aggregate,
        windows=window_summaries,
        regime_breakdown=regimes,
        operational_safety=operational_safety,
    )


def write_validation_outputs(result: RecoveryValidationResult, output_dir: Path | None = None) -> dict[str, Path]:
    root = output_dir or (OUTPUTS_DIR / "v24_4_2")
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "final_validation.json"
    md_path = root / "final_validation.md"
    json_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")

    aggregate = result.aggregate_metrics
    status = (
        "PASS"
        if (
            0.15 <= safe_float(aggregate.get("participation_rate")) <= 0.30
            and safe_float(aggregate.get("win_rate")) > 0.60
            and safe_float(aggregate.get("expectancy_R")) > 0.12
            and safe_float(aggregate.get("max_drawdown")) < 0.18
        )
        else "BLOCKED"
    )
    lines = [
        "# V24.4.2 Final Validation",
        "",
        f"Generated at: `{result.generated_at}`",
        f"Validation status: `{status}`",
        "",
        "## Aggregate Metrics",
        f"- Participation: `{safe_float(aggregate.get('participation_rate')):.6f}`",
        f"- Win rate: `{safe_float(aggregate.get('win_rate')):.6f}`",
        f"- Expectancy (R): `{safe_float(aggregate.get('expectancy_R')):.6f}`",
        f"- Max drawdown: `{safe_float(aggregate.get('max_drawdown')):.6f}`",
        f"- Trades: `{safe_int(aggregate.get('number_of_trades'))}`",
        "",
        "## Window Metrics",
    ]
    for window in result.windows:
        lines.append(
            f"- {window.label}: participation={safe_float(window.metrics.get('participation_rate')):.6f}, "
            f"win_rate={safe_float(window.metrics.get('win_rate')):.6f}, "
            f"expectancy={safe_float(window.metrics.get('expectancy_R')):.6f}, "
            f"drawdown={safe_float(window.metrics.get('max_drawdown')):.6f}"
        )
    lines.extend(
        [
            "",
            "## Operational Safety",
            f"- SELL guard blocks: `{safe_int(result.operational_safety.get('sell_guard_blocks'))}`",
            f"- Cooldown blocks: `{safe_int(result.operational_safety.get('cooldown_blocks'))}`",
            f"- Cluster blocks: `{safe_int(result.operational_safety.get('cluster_blocks'))}`",
            "",
            "## Config",
            f"```json\n{json.dumps(result.config, indent=2)}\n```",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "md": md_path}
