from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.v12.wfri import map_regime_class
from src.v13.mbeg import minority_guard
from src.v13.uts import derive_contradiction_type, unified_trade_score
from src.v14.rsc import RegimeStratifiedCalibrator
from src.v15.cpm import ConditionalPredictabilityMapper
from src.v15.eci import EconomicCalendarIntegration
from src.v15.pce import PredictabilityConditionedExecution


def sigmoid_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def _rolling_percentile(series: pd.Series, window: int = 512) -> pd.Series:
    def _percentile(window_values: pd.Series) -> float:
        values = np.asarray(window_values, dtype=np.float32)
        if values.size <= 1:
            return 0.5
        return float(np.sum(values <= values[-1]) / float(values.size))

    return series.rolling(window=window, min_periods=5).apply(_percentile, raw=False).fillna(0.5)


def _decision_timestamps(frame: pd.DataFrame) -> pd.Series:
    if "decision_ts" in frame.columns:
        return pd.to_datetime(frame["decision_ts"], utc=True, errors="coerce")
    base_ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return base_ts + pd.to_timedelta(frame.get("stage_bars", 0), unit="m")


def _regime_stability_bars(regimes: Sequence[str]) -> np.ndarray:
    streak = 0
    previous = None
    values: list[int] = []
    for regime in regimes:
        current = map_regime_class(regime)
        if current == previous:
            streak += 1
        else:
            streak = 1
            previous = current
        values.append(int(streak))
    return np.asarray(values, dtype=np.int32)


def enrich_v15_policy_frame(
    frame: pd.DataFrame,
    *,
    calibrator: RegimeStratifiedCalibrator,
    cabr_score_column: str = "cabr_score",
    cpm_mapper: ConditionalPredictabilityMapper | None = None,
    eci: EconomicCalendarIntegration | None = None,
) -> pd.DataFrame:
    working = frame.copy()
    mapper = cpm_mapper or ConditionalPredictabilityMapper()
    calendar = eci or EconomicCalendarIntegration.empty()

    working["regime_class"] = working.get("regime_class", working.get("dominant_regime", "ranging")).map(map_regime_class)
    raw_scores = sigmoid_scores(working[cabr_score_column].to_numpy(dtype=np.float32))
    working["cabr_raw_score"] = raw_scores
    working["calibrated_win_prob"] = [
        calibrator.calibrate(float(score), str(regime))
        for score, regime in zip(raw_scores.tolist(), working["regime_class"].tolist(), strict=False)
    ]
    working["bst_survival_score"] = np.clip(
        working.get("bst_survival_score", 1.0 - working.get("consensus_strength", 0.5)).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    working["analog_confidence"] = np.clip(
        working.get("leaf_analog_confidence", working.get("analog_similarity", 0.5)).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    working["contradiction_type"] = [
        derive_contradiction_type(branch_direction, consensus_direction, regime, consensus_strength)
        for branch_direction, consensus_direction, regime, consensus_strength in zip(
            working.get("branch_direction", 0.0).tolist(),
            working.get("consensus_direction", 0.0).tolist(),
            working["regime_class"].tolist(),
            working.get("consensus_strength", 0.5).tolist(),
            strict=False,
        )
    ]
    working["uts_score"] = [
        unified_trade_score(
            cabr_score=float(raw_score),
            calibrated_win_prob=float(prob),
            analog_confidence=float(analog),
            bst_survival_score=float(bst),
            contradiction_type=str(contradiction),
            emotional_momentum=float(momentum),
        )
        for raw_score, prob, analog, bst, contradiction, momentum in zip(
            working["cabr_raw_score"].tolist(),
            working["calibrated_win_prob"].tolist(),
            working["analog_confidence"].tolist(),
            working["bst_survival_score"].tolist(),
            working["contradiction_type"].tolist(),
            working.get("context_emotional_momentum", 0.0).tolist(),
            strict=False,
        )
    ]

    cpm_rows = [mapper.score_row(row) for row in working.to_dict(orient="records")]
    working["cpm_live_score"] = np.asarray([float(row["predictability"]) for row in cpm_rows], dtype=np.float32)
    working["cpm_agreement"] = np.asarray([float(row["agreement"]) for row in cpm_rows], dtype=np.float32)
    working["cpm_n_active"] = np.asarray([int(row["n_active"]) for row in cpm_rows], dtype=np.int16)
    working["cpm_directional_bias"] = np.asarray([float(row["directional_bias"]) for row in cpm_rows], dtype=np.float32)

    atr_series = pd.to_numeric(
        working.get("context_atr_percentile_30d", working.get("bcfe_atr_pct", pd.Series(0.0, index=working.index))),
        errors="coerce",
    ).fillna(0.5)
    if float(atr_series.max()) <= 1.5:
        atr_series = atr_series * 100.0
    else:
        atr_series = atr_series.clip(lower=0.0, upper=100.0)
    if "context_atr_percentile_30d" not in working.columns:
        atr_series = (_rolling_percentile(pd.to_numeric(working.get("bcfe_atr_pct", 0.0), errors="coerce").fillna(0.0)) * 100.0).astype(float)
    working["atr_percentile"] = atr_series.astype(float)

    decision_ts = _decision_timestamps(working)
    working["decision_ts"] = decision_ts
    if "exit_ts" not in working.columns:
        working["exit_ts"] = decision_ts + pd.Timedelta(minutes=15)

    eci_contexts = [calendar.get_context_at(ts) for ts in decision_ts.tolist()]
    working["eci_pre_release"] = np.asarray([bool(item["pre_release"]) for item in eci_contexts], dtype=bool)
    working["eci_reaction_window"] = np.asarray([bool(item["reaction_window"]) for item in eci_contexts], dtype=bool)
    working["eci_post_settling"] = np.asarray([bool(item["post_settling"]) for item in eci_contexts], dtype=bool)
    working["eci_avoid_window"] = np.asarray([bool(item["avoid_window"]) for item in eci_contexts], dtype=bool)
    working["eci_predictability_boost"] = np.asarray([float(item["eci_predictability_boost"]) for item in eci_contexts], dtype=np.float32)
    working["adjusted_cpm_score"] = np.clip(
        working["cpm_live_score"].to_numpy(dtype=np.float32) + working["eci_predictability_boost"].to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    working.loc[working["eci_avoid_window"], "adjusted_cpm_score"] = 0.0
    return working


def attach_execution_prices(frame: pd.DataFrame, raw_bars: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    decision_ts = _decision_timestamps(working)
    exit_ts = decision_ts + pd.Timedelta(minutes=15)
    lookup = raw_bars[["close"]].copy()
    aligned_entry = lookup.reindex(decision_ts, method="pad")
    aligned_exit = lookup.reindex(exit_ts, method="pad")
    working["decision_ts"] = decision_ts
    working["exit_ts"] = exit_ts
    working["entry_price"] = aligned_entry["close"].to_numpy(dtype=np.float32)
    working["exit_price"] = aligned_exit["close"].to_numpy(dtype=np.float32)
    return working


def generate_v15_decisions(
    frame: pd.DataFrame,
    *,
    pce: PredictabilityConditionedExecution,
    cabr_minimum: float = 0.52,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ranked = (
        frame.sort_values(["sample_id", "uts_score"], ascending=[True, False], kind="mergesort")
        .groupby("sample_id", sort=False)
        .head(1)
        .copy()
    )
    ranked = ranked.sort_values("decision_ts").reset_index(drop=True)
    ranked["regime_stability_bars"] = _regime_stability_bars(ranked["regime_class"].astype(str).tolist())

    keep_rows: list[int] = []
    skipped: list[dict[str, Any]] = []

    for idx, row in ranked.iterrows():
        skip_reason = None
        pce_reason = None
        size_multiplier = 1.0
        if bool(row.get("eci_avoid_window", False)):
            skip_reason = "eci_pre_release_avoid"
            pce_reason = "avoid_window"
        else:
            is_predictable, pce_reason = pce.is_predictable_window(
                cpm_score=float(row.get("adjusted_cpm_score", row.get("cpm_live_score", 0.5)) or 0.5),
                cpm_agreement=float(row.get("cpm_agreement", 0.0) or 0.0),
                regime_stability_bars=int(row.get("regime_stability_bars", 0) or 0),
                atr_percentile=float(row.get("atr_percentile", 50.0) or 50.0),
            )
            if not is_predictable:
                skip_reason = "pce_not_predictable"
            elif float(row.get("cabr_raw_score", 0.5) or 0.5) < float(cabr_minimum):
                skip_reason = "cabr_below_minimum"
            else:
                consensus_sign = int(np.sign(float(row.get("consensus_direction", row.get("branch_direction", 1.0))) or 1))
                consensus_direction = "BUY" if consensus_sign >= 0 else "SELL"
                minority_sign = int(np.sign(float(row.get("minority_rescue_branch", consensus_sign)) or consensus_sign))
                minority_direction = "BUY" if minority_sign >= 0 else "SELL"
                allow_trade, size_multiplier = minority_guard(
                    consensus_direction=consensus_direction,
                    minority_direction=minority_direction,
                    minority_score=float(row.get("leaf_minority_guardrail", 0.0) or 0.0),
                    consensus_strength=float(row.get("consensus_strength", 0.5) or 0.5),
                )
                if not allow_trade:
                    skip_reason = "minority_veto"

        ranked.at[idx, "size_multiplier"] = float(size_multiplier)
        if skip_reason is None:
            keep_rows.append(idx)
            continue

        skipped.append(
            {
                "sample_id": int(row["sample_id"]),
                "decision_ts": str(row["decision_ts"]),
                "reason": skip_reason,
                "pce_reason": pce_reason,
                "regime": str(row.get("regime_class", row.get("dominant_regime", "unknown"))),
                "uts_score": float(row.get("uts_score", 0.0) or 0.0),
                "cabr_score": float(row.get("cabr_raw_score", 0.0) or 0.0),
                "cpm_live_score": float(row.get("cpm_live_score", 0.5) or 0.5),
                "cpm_agreement": float(row.get("cpm_agreement", 0.0) or 0.0),
                "adjusted_cpm_score": float(row.get("adjusted_cpm_score", 0.5) or 0.5),
                "atr_percentile": float(row.get("atr_percentile", 50.0) or 50.0),
                "regime_stability_bars": int(row.get("regime_stability_bars", 0) or 0),
            }
        )

    executed = ranked.iloc[keep_rows].copy().reset_index(drop=True) if keep_rows else ranked.iloc[0:0].copy()
    return executed, skipped
