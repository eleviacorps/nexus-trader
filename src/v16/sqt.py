from __future__ import annotations

from collections import deque


class SimulationQualityTracker:
    def __init__(self, window_bars: int = 48, cold_threshold: int = 18):
        self.window = int(window_bars)
        self.cold_threshold = int(cold_threshold)
        self._history: deque[dict[str, object]] = deque(maxlen=self.window)

    def record(self, predicted_direction: str, actual_direction: str, confidence_tier: str) -> None:
        self._history.append(
            {
                "correct": str(predicted_direction).upper() == str(actual_direction).upper(),
                "tier": str(confidence_tier).lower(),
            }
        )

    @property
    def rolling_accuracy(self) -> float:
        if not self._history:
            return 0.5
        return float(sum(1 for item in self._history if item["correct"]) / len(self._history))

    @property
    def high_confidence_accuracy(self) -> float:
        relevant = [item for item in self._history if item["tier"] in {"very_high", "high"}]
        if not relevant:
            return 0.5
        return float(sum(1 for item in relevant if item["correct"]) / len(relevant))

    @property
    def label(self) -> str:
        if not self._history:
            return "NEUTRAL"
        accuracy = self.rolling_accuracy
        if accuracy > 0.68:
            return "HOT"
        if accuracy > 0.58:
            return "GOOD"
        if accuracy >= 0.50:
            return "NEUTRAL"
        return "COLD"

    def should_pause(self) -> bool:
        recent = list(self._history)[-24:]
        if len(recent) < 24:
            return False
        cold_count = sum(1 for item in recent if not item["correct"])
        return bool(cold_count >= self.cold_threshold)

    def summary(self) -> dict[str, object]:
        return {
            "window_bars": self.window,
            "recorded_bars": len(self._history),
            "rolling_accuracy": round(self.rolling_accuracy, 4),
            "high_confidence_acc": round(self.high_confidence_accuracy, 4),
            "label": self.label,
            "paused": self.should_pause(),
        }
