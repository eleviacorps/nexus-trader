from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ThresholdConfig:
    trend_up: float
    trend_down: float
    breakout: float
    range_value: float
    cooldown_decay: float
    cluster_radius: float
    size_multiplier: float

    def as_threshold_map(self) -> dict[str, float]:
        return {
            "trend_up": float(self.trend_up),
            "trend_down": float(self.trend_down),
            "breakout": float(self.breakout),
            "range": float(self.range_value),
        }


@dataclass(frozen=True)
class ThresholdSearchResult:
    config: ThresholdConfig
    metrics: dict[str, Any]
    objective_score: float


class ThresholdOptimizer:
    def __init__(self, evaluator: Callable[[ThresholdConfig], dict[str, Any]]):
        self.evaluator = evaluator

    def optimize(self, base: ThresholdConfig) -> tuple[ThresholdSearchResult, list[dict[str, Any]]]:
        # Keep constrained search fast enough for repeated validation loops.
        threshold_offsets = [0.0, 0.10, 0.15, 0.20, 0.24]
        cooldown_values = [0.70, 0.85]
        cluster_values = [0.20, 0.30]
        size_values = [0.95, 1.05]
        results: list[ThresholdSearchResult] = []

        for offset, cooldown_decay, cluster_radius, size_multiplier in product(
            threshold_offsets,
            cooldown_values,
            cluster_values,
            size_values,
        ):
            config = ThresholdConfig(
                trend_up=base.trend_up + offset,
                trend_down=base.trend_down + offset,
                breakout=base.breakout + offset,
                range_value=base.range_value + offset,
                cooldown_decay=cooldown_decay,
                cluster_radius=cluster_radius,
                size_multiplier=size_multiplier,
            )
            metrics = self.evaluator(config)
            objective_score = self._objective(metrics)
            results.append(
                ThresholdSearchResult(
                    config=config,
                    metrics=metrics,
                    objective_score=objective_score,
                )
            )

        results = sorted(results, key=lambda item: item.objective_score, reverse=True)
        serializable = [
            {
                "config": asdict(item.config),
                "objective_score": round(float(item.objective_score), 6),
                "metrics": item.metrics,
            }
            for item in results
        ]
        return results[0], serializable

    @staticmethod
    def _objective(metrics: dict[str, Any]) -> float:
        participation = float(metrics.get("participation_rate", 0.0))
        expectancy = float(metrics.get("expectancy_R", 0.0))
        win_rate = float(metrics.get("win_rate", 0.0))
        drawdown = float(metrics.get("max_drawdown", 1.0))

        score = 0.0
        # Participation target: 15%-30%.
        if 0.15 <= participation <= 0.30:
            score += 35.0
        else:
            distance = min(abs(participation - 0.15), abs(participation - 0.30))
            score += max(0.0, 35.0 - (distance * 220.0))

        # Expectancy target: >0.12R.
        score += max(0.0, min(25.0, (expectancy / 0.12) * 25.0))
        # Win-rate target: >60%.
        score += max(0.0, min(20.0, (win_rate / 0.60) * 20.0))
        # Drawdown target: <18%.
        drawdown_penalty = max(0.0, (drawdown - 0.18) * 100.0)
        score += max(0.0, 20.0 - drawdown_penalty)
        return float(score)
