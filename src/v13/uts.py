from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.v12.wfri import REGIME_CLASSES, map_regime_class


CONTRADICTION_PENALTIES = {
    'full_agreement_bull': 0.00,
    'full_agreement_bear': 0.00,
    'short_term_contrary': 0.15,
    'long_term_contrary': 0.20,
    'full_disagreement': 0.40,
    'unknown': 0.10,
}


def derive_contradiction_type(
    branch_direction: float,
    consensus_direction: float,
    regime: str,
    consensus_strength: float,
) -> str:
    branch_sign = int(np.sign(float(branch_direction)) or 0)
    consensus_sign = int(np.sign(float(consensus_direction)) or 0)
    regime_name = map_regime_class(regime)
    if branch_sign == consensus_sign and consensus_sign > 0:
        return 'full_agreement_bull'
    if branch_sign == consensus_sign and consensus_sign < 0:
        return 'full_agreement_bear'
    if consensus_strength >= 0.75:
        return 'full_disagreement'
    if regime_name in {'trending_up', 'trending_down'}:
        return 'long_term_contrary'
    return 'short_term_contrary'


def unified_trade_score(
    cabr_score: float,
    calibrated_win_prob: float,
    branch_diversity: float,
    analog_confidence: float,
    contradiction_type: str = 'unknown',
    emotional_momentum: float = 0.0,
) -> float:
    contradiction_penalty = float(CONTRADICTION_PENALTIES.get(str(contradiction_type), CONTRADICTION_PENALTIES['unknown']))
    raw_uts = (
        0.35 * float(cabr_score) +
        0.30 * float(calibrated_win_prob) +
        0.20 * float(branch_diversity) +
        0.15 * float(analog_confidence)
    ) - contradiction_penalty - max(0.0, abs(float(emotional_momentum)) - 0.5) * 0.05
    return float(np.clip(raw_uts, 0.0, 1.0))


@dataclass
class UTSThresholdSelector:
    target_participation: float = 0.30
    min_participation: float = 0.20
    max_participation: float = 0.35
    thresholds: dict[str, float] = field(default_factory=dict)

    def fit(self, scores: np.ndarray, regimes: Iterable[str], outcomes: np.ndarray | None = None) -> dict[str, float]:
        scores = np.asarray(scores, dtype=np.float32)
        outcomes = np.asarray(outcomes, dtype=np.float32) if outcomes is not None else None
        regimes = np.asarray([map_regime_class(value) for value in regimes], dtype=object)
        learned: dict[str, float] = {}
        for regime in REGIME_CLASSES:
            mask = regimes == regime
            if not np.any(mask):
                continue
            regime_scores = scores[mask]
            candidate_thresholds = np.unique(np.quantile(regime_scores, [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]).astype(np.float32))
            best_threshold = float(np.quantile(regime_scores, max(0.0, 1.0 - self.target_participation)))
            best_metric = -np.inf
            for threshold in candidate_thresholds.tolist():
                active = regime_scores >= float(threshold)
                participation = float(np.mean(active))
                if participation < self.min_participation or participation > self.max_participation:
                    continue
                if outcomes is None:
                    metric = participation
                else:
                    regime_outcomes = outcomes[mask]
                    pnl = float(np.mean(regime_outcomes[active])) if np.any(active) else -1.0
                    win_rate = float(np.mean(regime_outcomes[active] > 0.0)) if np.any(active) else 0.0
                    metric = pnl + 0.15 * win_rate
                if metric > best_metric:
                    best_metric = metric
                    best_threshold = float(threshold)
            learned[regime] = best_threshold
        self.thresholds = learned
        return learned

    def threshold_for(self, regime: str) -> float:
        regime_name = map_regime_class(regime)
        if regime_name in self.thresholds:
            return float(self.thresholds[regime_name])
        if self.thresholds:
            return float(np.median(list(self.thresholds.values())))
        return 0.5

    def should_trade(self, uts_score: float, regime: str) -> bool:
        return float(uts_score) >= self.threshold_for(regime)
