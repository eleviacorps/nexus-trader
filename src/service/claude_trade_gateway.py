from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping

from config.project_config import OUTPUTS_DIR
from src.service.llm_sidecar import _chat_json_request, parse_json_text


MODEL_ORDER_DEFAULT = [
    "moonshotai/kimi-k2-5",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "meta/llama-3.1-70b-instruct",
]


@dataclass(frozen=True)
class ClaudeGatewayConfig:
    model_order: list[str]
    decision_log_path: Path
    fail_closed: bool = True
    cache_fallback_enabled: bool = True


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    approve = bool(payload.get("approve", False))
    confidence = max(0.0, min(1.0, _safe_float(payload.get("confidence", 0.0))))
    risk_level = str(payload.get("risk_level", "HIGH")).upper()
    if risk_level not in {"LOW", "MEDIUM", "HIGH"}:
        risk_level = "HIGH"
    size_multiplier = max(0.5, min(1.5, _safe_float(payload.get("size_multiplier", 1.0))))
    reason = str(payload.get("reason", "no_reason")).strip() or "no_reason"
    return {
        "approve": approve,
        "confidence": confidence,
        "risk_level": risk_level,
        "size_multiplier": size_multiplier,
        "reason": reason,
    }


def _parse_judge_response(text: str) -> dict[str, Any]:
    payload = parse_json_text(text)
    return _normalize_response(payload)


class ClaudeTradeGateway:
    def __init__(self, config: ClaudeGatewayConfig | None = None):
        self.config = config or ClaudeGatewayConfig(
            model_order=list(MODEL_ORDER_DEFAULT),
            decision_log_path=OUTPUTS_DIR / "live" / "claude_decision_log.jsonl",
            fail_closed=True,
            cache_fallback_enabled=True,
        )
        self.config.decision_log_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_candidate(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You are the final execution risk committee for Nexus Trader. "
            "Return strict JSON with keys: approve, confidence, risk_level, size_multiplier, reason. "
            "Do not invent a new trade direction. "
            "If uncertain, reject."
        )
        user_prompt = (
            "Candidate trade JSON:\n"
            f"{json.dumps(dict(candidate), ensure_ascii=True)}\n\n"
            "Respond with JSON only."
        )
        errors: list[str] = []
        for model in self.config.model_order:
            response = _chat_json_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                provider="nvidia_nim",
                model=model,
                request_kind="claude_execution_judge",
                symbol="XAUUSD",
                context=dict(candidate),
                response_parser=_parse_judge_response,
            )
            if response.get("available", False):
                normalized = _normalize_response(dict(response.get("content", {})))
                payload = {
                    **normalized,
                    "available": True,
                    "source": "live_model",
                    "model": model,
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
                self._append_log(candidate, payload, errors=errors)
                return payload
            errors.append(str(response.get("error", f"{model}:unknown_error")))

        if self.config.cache_fallback_enabled:
            cached = self._cached_fallback(candidate)
            if cached is not None:
                payload = {
                    **cached,
                    "available": True,
                    "source": "cache_fallback",
                    "model": "cached",
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
                self._append_log(candidate, payload, errors=errors)
                return payload

        payload = {
            "approve": False,
            "confidence": 0.0,
            "risk_level": "HIGH",
            "size_multiplier": 1.0,
            "reason": "no_live_model_and_no_cache",
            "available": False,
            "source": "fail_closed",
            "model": None,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "errors": errors,
        }
        self._append_log(candidate, payload, errors=errors)
        return payload

    def _append_log(self, candidate: Mapping[str, Any], decision: Mapping[str, Any], errors: list[str]) -> None:
        row = {
            "logged_at": datetime.now(tz=UTC).isoformat(),
            "candidate": dict(candidate),
            "decision": dict(decision),
            "attempted_models": list(self.config.model_order),
            "errors": list(errors),
            "policy": {"fail_closed": self.config.fail_closed, "cache_fallback_enabled": self.config.cache_fallback_enabled},
        }
        with self.config.decision_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _cached_fallback(self, candidate: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self.config.decision_log_path
        if not path.exists():
            return None
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            decision = row.get("decision", {})
            if not isinstance(decision, Mapping):
                continue
            if not bool(decision.get("available", False)):
                continue
            if str(decision.get("source")) == "cache_fallback":
                continue
            rows.append(row)
        if not rows:
            return None
        scored = sorted(rows, key=lambda row: self._distance(candidate, row.get("candidate", {})))
        best = scored[0].get("decision", {})
        if not isinstance(best, Mapping):
            return None
        return _normalize_response(best)

    @staticmethod
    def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        direction_penalty = 0.0 if str(left.get("strategic_direction", "")).upper() == str(right.get("strategic_direction", "")).upper() else 1.0
        regime_penalty = 0.0 if str(left.get("regime", "")).lower() == str(right.get("regime", "")).lower() else 0.5
        admission_gap = abs(_safe_float(left.get("admission_score")) - _safe_float(right.get("admission_score")))
        confidence_gap = abs(_safe_float(left.get("regime_confidence")) - _safe_float(right.get("regime_confidence")))
        return direction_penalty + regime_penalty + (3.0 * admission_gap) + confidence_gap

