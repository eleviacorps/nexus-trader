from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.v12.wfri import REGIME_CLASSES, map_regime_class
from src.v13.lrtd import LiveRegimeTransitionDetector
from src.v13.mbeg import minority_guard
from src.v13.rcpc import RegimeConditionalPriorCalibrator
from src.v13.uts import UTSThresholdSelector, derive_contradiction_type, unified_trade_score


@dataclass(frozen=True)
class V13Decision:
    sample_id: int
    decision_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: str
    regime: str
    cabr_score: float
    raw_score: float
    calibrated_win_prob: float
    uts_score: float
    skip_reason: str | None
    size_multiplier: float


def sigmoid_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-values))


def enrich_v13_policy_frame(
    frame: pd.DataFrame,
    *,
    cabr_score_column: str = 'cabr_score',
    calibrator: RegimeConditionalPriorCalibrator,
) -> pd.DataFrame:
    working = frame.copy()
    working['regime_class'] = working.get('regime_class', working.get('dominant_regime', 'ranging')).map(map_regime_class)
    raw_scores = sigmoid_scores(working[cabr_score_column].to_numpy(dtype=np.float32))
    working['cabr_raw_score'] = raw_scores
    working['calibrated_win_prob'] = [
        calibrator.calibrate(float(score), str(regime))
        for score, regime in zip(raw_scores.tolist(), working['regime_class'].tolist(), strict=False)
    ]
    working['branch_diversity'] = np.clip(working.get('v10_diversity_score', 1.0 - working.get('consensus_strength', 0.5)).to_numpy(dtype=np.float32), 0.0, 1.0)
    working['analog_confidence'] = np.clip(working.get('leaf_analog_confidence', working.get('analog_similarity', 0.5)).to_numpy(dtype=np.float32), 0.0, 1.0)
    working['contradiction_type'] = [
        derive_contradiction_type(branch_direction, consensus_direction, regime, consensus_strength)
        for branch_direction, consensus_direction, regime, consensus_strength in zip(
            working.get('branch_direction', 0.0).tolist(),
            working.get('consensus_direction', 0.0).tolist(),
            working['regime_class'].tolist(),
            working.get('consensus_strength', 0.5).tolist(),
            strict=False,
        )
    ]
    working['uts_score'] = [
        unified_trade_score(
            cabr_score=float(raw_score),
            calibrated_win_prob=float(prob),
            branch_diversity=float(diversity),
            analog_confidence=float(analog),
            contradiction_type=str(contradiction),
            emotional_momentum=float(momentum),
        )
        for raw_score, prob, diversity, analog, contradiction, momentum in zip(
            working['cabr_raw_score'].tolist(),
            working['calibrated_win_prob'].tolist(),
            working['branch_diversity'].tolist(),
            working['analog_confidence'].tolist(),
            working['contradiction_type'].tolist(),
            working.get('context_emotional_momentum', 0.0).tolist(),
            strict=False,
        )
    ]
    return working


def fit_uts_selector(train_frame: pd.DataFrame, *, score_column: str = 'uts_score') -> UTSThresholdSelector:
    selector = UTSThresholdSelector(target_participation=0.35, min_participation=0.20, max_participation=0.45)
    selector.fit(
        train_frame[score_column].to_numpy(dtype=np.float32),
        train_frame.get('regime_class', train_frame.get('dominant_regime', 'ranging')).tolist(),
        train_frame['setl_target_net_unit_pnl'].to_numpy(dtype=np.float32),
    )
    return selector


def derive_deployable_regimes(train_frame: pd.DataFrame, *, score_column: str = 'uts_score') -> set[str]:
    deployable: set[str] = set()
    for regime, subset in train_frame.groupby('regime_class', sort=True):
        if len(subset) < 50:
            continue
        threshold = float(np.quantile(subset[score_column].to_numpy(dtype=np.float32), 0.65))
        active = subset[score_column].to_numpy(dtype=np.float32) >= threshold
        if not np.any(active):
            continue
        win_rate = float(np.mean(subset.loc[active, 'setl_target_net_unit_pnl'].to_numpy(dtype=np.float32) > 0.0))
        if win_rate >= 0.53:
            deployable.add(str(regime))
    if not deployable:
        deployable.update({'trending_up', 'trending_down'})
    return deployable


def attach_execution_prices(frame: pd.DataFrame, raw_bars: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    base_ts = pd.to_datetime(working['timestamp'], utc=True, errors='coerce')
    decision_ts = base_ts + pd.to_timedelta(working.get('stage_bars', 0), unit='m')
    exit_ts = decision_ts + pd.Timedelta(minutes=15)
    lookup = raw_bars[['close']].copy()
    aligned_entry = lookup.reindex(decision_ts, method='pad')
    aligned_exit = lookup.reindex(exit_ts, method='pad')
    working['decision_ts'] = decision_ts
    working['exit_ts'] = exit_ts
    working['entry_price'] = aligned_entry['close'].to_numpy(dtype=np.float32)
    working['exit_price'] = aligned_exit['close'].to_numpy(dtype=np.float32)
    return working


def generate_v13_decisions(
    frame: pd.DataFrame,
    *,
    threshold_selector: UTSThresholdSelector,
    deployable_regimes: Iterable[str],
    lrtd_threshold: float = 0.30,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ranked = frame.sort_values(['sample_id', 'uts_score'], ascending=[True, False], kind='mergesort').groupby('sample_id', sort=False).head(1).copy()
    ranked = ranked.sort_values('decision_ts').reset_index(drop=True)
    detector = LiveRegimeTransitionDetector(transition_threshold=lrtd_threshold)
    deployable = {map_regime_class(value) for value in deployable_regimes}
    keep_rows: list[int] = []
    skipped: list[dict[str, Any]] = []

    for idx, row in ranked.iterrows():
        regime = map_regime_class(row.get('regime_class', row.get('dominant_regime', 'ranging')))
        detector.update(regime, float(row.get('context_regime_confidence', row.get('hmm_regime_probability', 0.5)) or 0.5))
        skip_reason = None
        size_multiplier = 1.0
        if regime not in deployable:
            skip_reason = 'wfri_not_deployable'
        elif detector.should_suppress():
            skip_reason = 'lrtd_suppressed'
        else:
            consensus_sign = int(np.sign(float(row.get('consensus_direction', row.get('branch_direction', 1.0))) or 1))
            consensus_direction = 'BUY' if consensus_sign >= 0 else 'SELL'
            minority_sign = int(np.sign(float(row.get('minority_rescue_branch', consensus_sign)) or consensus_sign))
            minority_direction = 'BUY' if minority_sign >= 0 else 'SELL'
            allow_trade, size_multiplier = minority_guard(
                consensus_direction=consensus_direction,
                minority_direction=minority_direction,
                minority_score=float(row.get('leaf_minority_guardrail', 0.0) or 0.0),
                consensus_strength=float(row.get('consensus_strength', 0.5) or 0.5),
            )
            if not allow_trade:
                skip_reason = 'minority_veto'
            elif not threshold_selector.should_trade(float(row.get('uts_score', 0.0) or 0.0), regime):
                skip_reason = 'uts_below_threshold'
        if skip_reason is None:
            keep_rows.append(idx)
        else:
            skipped.append(
                {
                    'sample_id': int(row['sample_id']),
                    'decision_ts': str(row['decision_ts']),
                    'reason': skip_reason,
                    'regime': regime,
                    'uts_score': float(row.get('uts_score', 0.0) or 0.0),
                    'cabr_score': float(row.get('cabr_score', 0.0) or 0.0),
                }
            )
        ranked.at[idx, 'size_multiplier'] = float(size_multiplier)

    executed = ranked.iloc[keep_rows].copy().reset_index(drop=True) if keep_rows else ranked.iloc[0:0].copy()
    return executed, skipped
