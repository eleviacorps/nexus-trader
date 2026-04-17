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
        admission_ok = float(candidate.admission_score) >= (float(candidate.regime_threshold) + 0.05)
        readiness_ok = float(readiness_score) >= 90.0

        if normalized_mode == "auto_mode":
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
            if not judge_approve:
                return RouteDecision(
                    route="rejected",
                    allowed=False,
                    reason="judge_rejected_trade",
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

