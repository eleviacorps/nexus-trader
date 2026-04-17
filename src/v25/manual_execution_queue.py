from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
import uuid


@dataclass(frozen=True)
class QueueItem:
    ticket_id: str
    created_at: str
    direction: str
    confidence: float
    regime: str
    stop_loss: float
    take_profit: float
    admission_score: float
    reason: str
    payload: dict[str, Any]


class ManualExecutionQueue:
    def __init__(self):
        self._queue: deque[QueueItem] = deque()
        self._approved: list[QueueItem] = []
        self._rejected: list[QueueItem] = []

    def enqueue(self, payload: dict[str, Any], reason: str) -> QueueItem:
        item = QueueItem(
            ticket_id=str(uuid.uuid4()),
            created_at=datetime.now(tz=UTC).isoformat(),
            direction=str(payload.get("direction", "HOLD")).upper(),
            confidence=float(payload.get("confidence", 0.0)),
            regime=str(payload.get("regime", "unknown")),
            stop_loss=float(payload.get("stop_loss", 0.0) or 0.0),
            take_profit=float(payload.get("take_profit", 0.0) or 0.0),
            admission_score=float(payload.get("admission_score", 0.0)),
            reason=str(reason),
            payload=dict(payload),
        )
        self._queue.append(item)
        return item

    def pop_next(self) -> QueueItem | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def approve(self, ticket_id: str) -> QueueItem | None:
        for idx, item in enumerate(self._queue):
            if item.ticket_id == ticket_id:
                picked = self._queue[idx]
                del self._queue[idx]
                self._approved.append(picked)
                return picked
        return None

    def reject(self, ticket_id: str) -> QueueItem | None:
        for idx, item in enumerate(self._queue):
            if item.ticket_id == ticket_id:
                picked = self._queue[idx]
                del self._queue[idx]
                self._rejected.append(picked)
                return picked
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "pending": len(self._queue),
            "approved": len(self._approved),
            "rejected": len(self._rejected),
            "latest_pending": [item.__dict__ for item in list(self._queue)[-5:]],
        }

