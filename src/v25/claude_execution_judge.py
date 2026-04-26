from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.service.claude_trade_gateway import ClaudeTradeGateway


@dataclass
class ClaudeExecutionJudge:
    gateway: ClaudeTradeGateway
    prompt_path: Path

    def judge(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        response = self.gateway.evaluate_candidate(dict(candidate))
        return {
            "approve": bool(response.get("approve", False)),
            "confidence": float(response.get("confidence", 0.0)),
            "risk_level": str(response.get("risk_level", "HIGH")),
            "size_multiplier": float(response.get("size_multiplier", 1.0)),
            "reason": str(response.get("reason", "no_reason")),
            "available": bool(response.get("available", False)),
            "source": response.get("source"),
            "model": response.get("model"),
        }

