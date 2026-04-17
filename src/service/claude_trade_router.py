from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
from typing import Any, Mapping

from config.project_config import OUTPUTS_DIR
from src.service.claude_trade_gateway import ClaudeTradeGateway
from src.v25.auto_execution_engine import AutoExecutionEngine
from src.v25.execution_mode_router import ExecutionCandidate, ExecutionModeRouter
from src.v25.manual_execution_queue import ManualExecutionQueue


@dataclass
class ClaudeTradeRouter:
    gateway: ClaudeTradeGateway
    mode_router: ExecutionModeRouter
    manual_queue: ManualExecutionQueue
    auto_engine: AutoExecutionEngine
    live_report_path: Path = OUTPUTS_DIR / "live" / "live_paper_report.json"

    def route_candidate(
        self,
        *,
        numeric_candidate: Mapping[str, Any],
        mode: str,
        deployment_score: float,
        execution_channel: str = "paper",
    ) -> dict[str, Any]:
        candidate = ExecutionCandidate(
            direction=str(numeric_candidate.get("strategic_direction", "HOLD")).upper(),
            confidence=float(numeric_candidate.get("calibrated_probability", 0.0)),
            regime=str(numeric_candidate.get("regime", "unknown")),
            admission_score=float(numeric_candidate.get("admission_score", 0.0)),
            regime_threshold=float(numeric_candidate.get("regime_threshold", 0.0)),
            stop_loss=float(numeric_candidate.get("stop_loss", 0.0)),
            take_profit=float(numeric_candidate.get("take_profit", 0.0)),
            reason=str(numeric_candidate.get("reasons", [""])[0] if isinstance(numeric_candidate.get("reasons"), list) else numeric_candidate.get("reason", "")),
            expected_rr=float(numeric_candidate.get("expected_rr", 0.0)),
        )
        judge = self.gateway.evaluate_candidate(dict(numeric_candidate))
        route = self.mode_router.route(mode=mode, candidate=candidate, readiness_score=deployment_score, judge_result=judge)
        timestamp = datetime.now(tz=UTC).isoformat()

        if route.route == "manual_queue" and route.allowed:
            queued = self.manual_queue.enqueue(route.payload, reason=route.reason)
            result = {
                "timestamp": timestamp,
                "status": "queued_for_manual",
                "ticket_id": queued.ticket_id,
                "route_reason": route.reason,
                "judge": judge,
            }
            self._append_live_report(result)
            return result

        if route.route == "auto_execute" and route.allowed:
            execution_payload = {
                **route.payload,
                "symbol": str(numeric_candidate.get("symbol", "XAUUSD")),
                "entry_price": float(numeric_candidate.get("entry_price", numeric_candidate.get("market_price", 0.0))),
                "direction": candidate.direction,
            }
            executed = self.auto_engine.execute(execution_payload, execution_channel=execution_channel)
            result = {
                "timestamp": timestamp,
                "status": "auto_executed" if executed.executed else "auto_rejected",
                "route_reason": route.reason,
                "execution_reason": executed.reason,
                "execution_channel": executed.channel,
                "judge": judge,
                "execution_payload": executed.payload,
            }
            self._append_live_report(result)
            return result

        result = {
            "timestamp": timestamp,
            "status": "rejected",
            "route_reason": route.reason,
            "judge": judge,
        }
        self._append_live_report(result)
        return result

    def _append_live_report(self, row: Mapping[str, Any]) -> None:
        self.live_report_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict[str, Any]]
        if self.live_report_path.exists():
            try:
                payload = json.loads(self.live_report_path.read_text(encoding="utf-8"))
                existing = list(payload.get("events", [])) if isinstance(payload, Mapping) else []
            except Exception:
                existing = []
        else:
            existing = []
        existing.append(dict(row))
        output = {"updated_at": datetime.now(tz=UTC).isoformat(), "events": existing[-500:]}
        self.live_report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

