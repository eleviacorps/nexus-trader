from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionCandidate:
    direction: str
    confidence: float
    regime: str
    admission_score: float
    regime_threshold: float
    stop_loss: float
    take_profit: float
    reason: str
    expected_rr: float


@dataclass(frozen=True)
class RouteDecision:
    route: str
    allowed: bool
    reason: str
    payload: dict[str, Any]


class ExecutionModeRouter:
    def route(
        self,
        *,
        mode: str,
        candidate: ExecutionCandidate,
        readiness_score: float,
        judge_result: dict[str, Any],
    ) -> RouteDecision:
        normalized_mode = (mode or "manual_mode").strip().lower()
        payload = {
            "direction": candidate.direction,
            "confidence": float(candidate.confidence),
            "regime": candidate.regime,
            "admission_score": float(candidate.admission_score),
            "regime_threshold": float(candidate.regime_threshold),
            "stop_loss": float(candidate.stop_loss),
            "take_profit": float(candidate.take_profit),
            "expected_rr": float(candidate.expected_rr),
            "reason": candidate.reason,
            "judge": dict(judge_result or {}),
        }
        if normalized_mode == "manual_mode":
            return RouteDecision(
                route="manual_queue",
                allowed=True,
                reason="manual_mode_selected",
                payload=payload,
            )

        judge_approve = bool((judge_result or {}).get("approve", False))
        judge_confidence = float((judge_result or {}).get("confidence", 0.0) or 0.0)
        judge_confidence_ok = judge_confidence > 0.65
        emergency_stop = bool((judge_result or {}).get("emergency_stop", False))
        admission_ok = float(candidate.admission_score) >= (float(candidate.regime_threshold) + 0.05)
        readiness_ok = float(readiness_score) >= 90.0
        tradeability_probability_raw = (judge_result or {}).get("tradeability_probability")
        tradeability_probability = None
        if tradeability_probability_raw is not None:
            try:
                tradeability_probability = float(tradeability_probability_raw)
            except Exception:
                tradeability_probability = None
        tradeability_ok = (tradeability_probability is not None) and (tradeability_probability > 0.62)
        payload["tradeability_probability"] = tradeability_probability
        payload["judge_confidence"] = judge_confidence
        payload["emergency_stop"] = emergency_stop

        if normalized_mode == "auto_mode":
            if emergency_stop:
                return RouteDecision(
                    route="rejected",
                    allowed=False,
                    reason="judge_emergency_stop",
                    payload=payload,
                )
            if not readiness_ok:
                return RouteDecision(
                    route="rejected",
                    allowed=False,
                    reason="readiness_below_90",
                    payload=payload,
                )
            if not admission_ok:
                return RouteDecision(
                    route="rejected",
                    allowed=False,
                    reason="admission_below_threshold_plus_buffer",
                    payload=payload,
                )
            if tradeability_probability is None:
                return RouteDecision(
                    route="rejected",
                    allowed=False,
                    reason="tradeability_probability_missing",
                    payload=payload,
                )
            if not tradeability_ok:
                return RouteDecision(
                    route="rejected",
                    allowed=False,
                    reason="tradeability_probability_below_0_62",
                    payload=payload,
                )
            if not judge_approve:
                return RouteDecision(
                    route="rejected",
                    allowed=False,
                    reason="judge_rejected_trade",
                    payload=payload,
                )
            if not judge_confidence_ok:
                return RouteDecision(
                    route="rejected",
                    allowed=False,
                    reason="judge_confidence_below_0_65",
                    payload=payload,
                )
            return RouteDecision(
                route="auto_execute",
                allowed=True,
                reason="auto_mode_gates_passed",
                payload=payload,
            )

        return RouteDecision(
            route="rejected",
            allowed=False,
            reason=f"unknown_mode:{normalized_mode}",
            payload=payload,
        )
