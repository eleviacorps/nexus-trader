from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ClusterDecision:
    allow: bool
    reason: str


class TradeClusterFilter:
    """
    Reject duplicate trades from the same regime cluster.
    """

    def __init__(self, max_age_minutes: int = 45, price_radius: float = 0.35):
        self.max_age_minutes = int(max_age_minutes)
        self.price_radius = float(price_radius)
        self._history: deque[dict[str, Any]] = deque(maxlen=2048)

    def _purge(self, now: datetime) -> None:
        if not self._history:
            return
        keep: deque[dict[str, Any]] = deque(maxlen=self._history.maxlen)
        for row in self._history:
            timestamp = row.get("timestamp")
            if not isinstance(timestamp, datetime):
                continue
            age_minutes = abs((now - timestamp).total_seconds()) / 60.0
            if age_minutes <= float(self.max_age_minutes):
                keep.append(row)
        self._history = keep

    def evaluate(
        self,
        *,
        regime: str,
        direction: str,
        timestamp: datetime,
        entry_price: float,
    ) -> ClusterDecision:
        self._purge(timestamp)
        regime_key = str(regime).lower()
        direction_key = str(direction).upper()
        for row in self._history:
            if str(row.get("regime", "")).lower() != regime_key:
                continue
            if str(row.get("direction", "")).upper() != direction_key:
                continue
            prior_price = float(row.get("entry_price", 0.0) or 0.0)
            if abs(float(entry_price) - prior_price) <= self.price_radius:
                return ClusterDecision(False, "duplicate_cluster_same_regime_direction")
        self._history.append(
            {
                "regime": regime_key,
                "direction": direction_key,
                "timestamp": timestamp,
                "entry_price": float(entry_price),
            }
        )
        return ClusterDecision(True, "cluster_ok")
