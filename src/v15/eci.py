from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


HIGH_IMPACT_EVENTS = {"fed_rate", "cpi", "nfp", "gdp", "ecb_rate", "fomc"}


@dataclass(frozen=True)
class EconomicEvent:
    timestamp: datetime
    event_type: str
    importance: int
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None

    @property
    def surprise(self) -> float | None:
        if self.actual is not None and self.forecast is not None:
            return float(self.actual - self.forecast)
        return None

    @property
    def is_high_impact(self) -> bool:
        return int(self.importance) == 3 or str(self.event_type) in HIGH_IMPACT_EVENTS


class EconomicCalendarIntegration:
    def __init__(self, events: list[EconomicEvent] | None = None):
        self.events = sorted(list(events or []), key=lambda item: item.timestamp)

    @classmethod
    def empty(cls) -> "EconomicCalendarIntegration":
        return cls(events=[])

    def get_context_at(self, current_time: datetime | pd.Timestamp) -> dict[str, Any]:
        timestamp = pd.Timestamp(current_time)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        current = timestamp.to_pydatetime()

        past_high = [event for event in self.events if event.is_high_impact and event.timestamp <= current]
        future_high = [event for event in self.events if event.is_high_impact and event.timestamp > current]

        mins_since = (
            (current - past_high[-1].timestamp).total_seconds() / 60.0
            if past_high else 9999.0
        )
        mins_to = (
            (future_high[0].timestamp - current).total_seconds() / 60.0
            if future_high else 9999.0
        )

        pre_release = 5.0 < mins_to < 45.0
        reaction_window = 0.0 <= mins_since <= 20.0
        post_settling = 30.0 <= mins_since <= 90.0
        avoid_window = 0.0 <= mins_to <= 5.0
        last_surprise = past_high[-1].surprise if past_high and past_high[-1].surprise is not None else 0.0

        return {
            "pre_release": bool(pre_release),
            "reaction_window": bool(reaction_window),
            "post_settling": bool(post_settling),
            "avoid_window": bool(avoid_window),
            "mins_to_next_high": float(mins_to),
            "mins_since_last_high": float(mins_since),
            "last_surprise_magnitude": float(abs(last_surprise)),
            "eci_predictability_boost": (
                0.15 if (reaction_window or post_settling) and not avoid_window
                else -0.10 if avoid_window
                else 0.0
            ),
        }

    @classmethod
    def from_csv(cls, path: str | Path) -> "EconomicCalendarIntegration":
        path = Path(path)
        if not path.exists():
            return cls.empty()
        frame = pd.read_csv(path, parse_dates=["datetime"])
        events = []
        for _, row in frame.iterrows():
            timestamp = pd.Timestamp(row["datetime"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(timezone.utc)
            else:
                timestamp = timestamp.tz_convert(timezone.utc)
            events.append(
                EconomicEvent(
                    timestamp=timestamp.to_pydatetime(),
                    event_type=str(row.get("event_type", "unknown")).lower().replace(" ", "_"),
                    importance=int(row.get("importance", 2)),
                    actual=_optional_float(row.get("actual")),
                    forecast=_optional_float(row.get("forecast")),
                    previous=_optional_float(row.get("previous")),
                )
            )
        return cls(events=events)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric
