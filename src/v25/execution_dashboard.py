from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.v25.manual_execution_queue import ManualExecutionQueue
from src.v25.paper_trade_engine import PaperTradeEngine


@dataclass
class ExecutionDashboard:
    manual_queue: ManualExecutionQueue
    paper_engine: PaperTradeEngine

    def snapshot(self, *, deployment_score: float, mode: str) -> dict[str, Any]:
        return {
            "mode": mode,
            "deployment_score": float(deployment_score),
            "manual_queue": self.manual_queue.snapshot(),
            "paper_summary": self.paper_engine.summary(),
        }

