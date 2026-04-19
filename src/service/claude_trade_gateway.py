from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping

from config.project_config import MODELS_DIR, OUTPUTS_DIR
from src.service.llm_sidecar import _chat_json_request, parse_json_text
from src.v25.local_judge_cache import LocalJudgeCache
from src.v25.tradeability_model import TradeabilityModel


MODEL_ORDER_DEFAULT = [
    "z-ai/glm-5.1",
]


@dataclass(frozen=True)
class ClaudeGatewayConfig:
    model_order: list[str]
    decision_log_path: Path
    fail_closed: bool = True
    cache_fallback_enabled: bool = True
    local_cache_path: Path = OUTPUTS_DIR / "v25" / "local_judge_cache.jsonl"
    local_cache_similarity_threshold: float = 0.92
    tradeability_model_path: Path = MODELS_DIR / "v25" / "tradeability_model.json"
    tradeability_threshold: float = 0.62


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    approve = bool(payload.get("approve", False))
    confidence = max(0.0, min(1.0, _safe_float(payload.get("confidence", 0.0))))
    risk_multiplier = _safe_float(payload.get("risk_multiplier", payload.get("size_multiplier", 1.0)), 1.0)
    risk_multiplier = max(0.25, min(1.5, risk_multiplier))
    emergency_stop = bool(payload.get("emergency_stop", False))
    reason = str(payload.get("reason", "no_reason")).strip() or "no_reason"
    risk_level = "HIGH" if risk_multiplier <= 0.7 else "MEDIUM" if risk_multiplier <= 0.95 else "LOW"
    return {
        "approve": approve,
        "confidence": confidence,
        "risk_multiplier": risk_multiplier,
        "risk_level": risk_level,
        "size_multiplier": risk_multiplier,
        "emergency_stop": emergency_stop,
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
        self.config.local_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_cache = LocalJudgeCache(
            path=self.config.local_cache_path,
            similarity_threshold=self.config.local_cache_similarity_threshold,
        )
        self.tradeability_model: TradeabilityModel | None = None
        if self.config.tradeability_model_path.exists():
            try:
                model = TradeabilityModel.load(self.config.tradeability_model_path)
                if model.fitted:
                    self.tradeability_model = model
            except Exception:
                self.tradeability_model = None

    def evaluate_candidate(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You are the final execution risk committee for Nexus Trader V25.1. "
            "Respond with strict JSON only, using exactly these keys: "
            "approve, confidence, risk_multiplier, reason, emergency_stop. "
            "Do not invent fields, do not change trade direction, and keep confidence between 0 and 1."
        )
        user_prompt = (
            "Candidate trade JSON:\n"
            f"{json.dumps(dict(candidate), ensure_ascii=True)}\n\n"
            "Respond with JSON only."
        )
        errors: list[str] = []
        cache_hit = self.local_cache.lookup(candidate)
        if cache_hit is not None:
            decision = dict(cache_hit.get("decision", {}))
            normalized = _normalize_response(decision)
            tradeability_probability = self._tradeability_probability(candidate, confidence=normalized["confidence"])
            gated = self._apply_tradeability_gate(normalized, tradeability_probability)
            payload = {
                **gated,
                "available": True,
                "source": "local_cache",
                "model": "local_cache_similarity",
                "similarity": _safe_float(cache_hit.get("similarity"), 0.0),
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
            self._append_log(candidate, payload, errors=errors)
            return payload

        tradeability_probability = self._tradeability_probability(candidate, confidence=0.0)
        if tradeability_probability is not None and tradeability_probability <= float(self.config.tradeability_threshold):
            payload = {
                "approve": False,
                "confidence": float(max(0.0, min(1.0, tradeability_probability))),
                "risk_multiplier": 0.5,
                "risk_level": "HIGH",
                "size_multiplier": 0.5,
                "emergency_stop": False,
                "reason": f"tradeability_probability_below_{self.config.tradeability_threshold:.2f}",
                "tradeability_probability": tradeability_probability,
                "available": True,
                "source": "local_tradeability_model",
                "model": "local_tradeability_only",
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
            self._append_log(candidate, payload, errors=errors)
            return payload

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
                tradeability_probability_live = self._tradeability_probability(candidate, confidence=normalized["confidence"])
                gated = self._apply_tradeability_gate(normalized, tradeability_probability_live)
                payload = {
                    **gated,
                    "available": True,
                    "source": "live_model",
                    "model": model,
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
                self.local_cache.add(candidate, payload)
                self.local_cache.save()
                self._append_log(candidate, payload, errors=errors)
                return payload
            errors.append(str(response.get("error", f"{model}:unknown_error")))

        if self.config.cache_fallback_enabled:
            cached = self._cached_fallback(candidate)
            if cached is not None:
                tradeability_probability_cached = self._tradeability_probability(candidate, confidence=_safe_float(cached.get("confidence"), 0.0))
                gated = self._apply_tradeability_gate(cached, tradeability_probability_cached)
                payload = {
                    **gated,
                    "available": True,
                    "source": "cache_fallback",
                    "model": "cached",
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
                self._append_log(candidate, payload, errors=errors)
                return payload

        # Never hard-stop just because live API is unavailable.
        if tradeability_probability is not None:
            fallback_approve = bool(tradeability_probability > float(self.config.tradeability_threshold))
            payload = {
                "approve": fallback_approve,
                "confidence": float(max(0.0, min(1.0, max(tradeability_probability, 0.66 if fallback_approve else tradeability_probability)))),
                "risk_multiplier": 0.5,
                "risk_level": "HIGH",
                "size_multiplier": 0.5,
                "emergency_stop": False,
                "reason": "glm_unavailable_local_tradeability_fallback",
                "tradeability_probability": float(tradeability_probability),
                "available": True,
                "source": "local_tradeability_fallback",
                "model": "local_tradeability_only",
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "errors": errors,
            }
            self._append_log(candidate, payload, errors=errors)
            return payload

        payload = {
            "approve": False,
            "confidence": 0.0,
            "risk_multiplier": 0.5,
            "risk_level": "HIGH",
            "size_multiplier": 0.5,
            "emergency_stop": False,
            "reason": "no_live_model_and_no_cache",
            "tradeability_probability": None,
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
            "policy": {
                "fail_closed": self.config.fail_closed,
                "cache_fallback_enabled": self.config.cache_fallback_enabled,
                "tradeability_threshold": self.config.tradeability_threshold,
                "cache_similarity_threshold": self.config.local_cache_similarity_threshold,
            },
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

    def _tradeability_probability(self, candidate: Mapping[str, Any], confidence: float) -> float | None:
        if self.tradeability_model is None:
            return None
        try:
            item = self._build_tradeability_features(candidate, confidence=confidence)
            probability = float(self.tradeability_model.predict_probability(item))
            return max(0.0, min(1.0, probability))
        except Exception:
            return None

    def _build_tradeability_features(self, candidate: Mapping[str, Any], confidence: float) -> dict[str, Any]:
        recent_trade_health = candidate.get("recent_trade_health", {})
        if not isinstance(recent_trade_health, Mapping):
            recent_trade_health = {}
        rolling_wr = _safe_float(recent_trade_health.get("rolling_win_rate_10"), 0.5)
        streak = (rolling_wr - 0.5) * 10.0
        return {
            "admission_score": _safe_float(candidate.get("admission_score"), 0.0),
            "regime": str(candidate.get("regime", "unknown")),
            "direction": str(candidate.get("strategic_direction", candidate.get("tactical_direction", "HOLD"))),
            "spread": _safe_float(candidate.get("spread", candidate.get("spread_estimate", 0.0)), 0.0),
            "slippage": _safe_float(candidate.get("slippage", candidate.get("slippage_estimate", 0.0)), 0.0),
            "cabr_score": _safe_float(candidate.get("tactical_cabr", candidate.get("cabr_score", 0.5)), 0.5),
            "branch_quality": _safe_float(candidate.get("branch_quality_score", candidate.get("cpm_score", 0.5)), 0.5),
            "claude_confidence": _safe_float(confidence, 0.0),
            "recent_streak": streak,
            "cluster_count": _safe_float(candidate.get("cluster_count", candidate.get("sell_cluster_count", 0.0)), 0.0),
        }

    def _apply_tradeability_gate(self, decision: Mapping[str, Any], tradeability_probability: float | None) -> dict[str, Any]:
        normalized = _normalize_response(decision)
        probability = None if tradeability_probability is None else float(max(0.0, min(1.0, tradeability_probability)))
        output = dict(normalized)
        output["tradeability_probability"] = probability
        if probability is None:
            return output
        if probability <= float(self.config.tradeability_threshold):
            output["approve"] = False
            output["risk_level"] = "HIGH"
            output["risk_multiplier"] = min(_safe_float(output.get("risk_multiplier"), 1.0), 0.5)
            output["size_multiplier"] = output["risk_multiplier"]
            output["reason"] = f"tradeability_probability_below_{self.config.tradeability_threshold:.2f}"
        return output
