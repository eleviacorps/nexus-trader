from __future__ import annotations

from dataclasses import dataclass

from src.v24_4_2.regime_threshold_router import RegimeContext, RegimeThresholdRouter


@dataclass(frozen=True)
class AdmissionCandidate:
    calibrated_probability: float
    tactical_cabr: float
    regime_profitability: float
    execution_quality: float
    strategic_alignment: float
    recent_trade_health: float
    strategic_direction: str
    tactical_direction: str


@dataclass(frozen=True)
class AdmissionDecision:
    admission_score: float
    threshold: float
    allow: bool
    reason: str


class AdaptiveAdmission:
    def __init__(self, router: RegimeThresholdRouter | None = None):
        self.router = router or RegimeThresholdRouter()

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def score(self, candidate: AdmissionCandidate) -> float:
        return self._clamp_01(
            (0.35 * self._clamp_01(candidate.calibrated_probability))
            + (0.20 * self._clamp_01(candidate.tactical_cabr))
            + (0.15 * self._clamp_01(candidate.regime_profitability))
            + (0.10 * self._clamp_01(candidate.execution_quality))
            + (0.10 * self._clamp_01(candidate.strategic_alignment))
            + (0.10 * self._clamp_01(candidate.recent_trade_health))
        )

    def allow(self, candidate: AdmissionCandidate, regime_ctx: RegimeContext) -> AdmissionDecision:
        admission_score = self.score(candidate)
        threshold_decision = self.router.route(regime_ctx)
        if not threshold_decision.enabled:
            return AdmissionDecision(
                admission_score=admission_score,
                threshold=threshold_decision.threshold,
                allow=False,
                reason=threshold_decision.reason,
            )
        if threshold_decision.strategic_only:
            aligned = candidate.strategic_direction.upper() == candidate.tactical_direction.upper()
            allow = bool(aligned and admission_score > threshold_decision.threshold)
            return AdmissionDecision(
                admission_score=admission_score,
                threshold=threshold_decision.threshold,
                allow=allow,
                reason=f"{threshold_decision.reason}; aligned={aligned}",
            )
        allow = bool(admission_score > threshold_decision.threshold)
        return AdmissionDecision(
            admission_score=admission_score,
            threshold=threshold_decision.threshold,
            allow=allow,
            reason=threshold_decision.reason,
        )

