from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.mcts.reverse_collapse import CollapseResult


@dataclass
class ConePoint:
    horizon: int
    lower: float
    center: float
    upper: float


@dataclass
class ProbabilityCone:
    mean_probability: float
    uncertainty_width: float
    consensus_score: float
    points: List[ConePoint]


def build_probability_cone(result: CollapseResult, horizon_steps: int = 5) -> ProbabilityCone:
    points: List[ConePoint] = []
    for step in range(1, horizon_steps + 1):
        scale = step / horizon_steps
        half_width = result.uncertainty_width * scale * 0.5
        center = result.mean_probability
        lower = max(0.0, center - half_width)
        upper = min(1.0, center + half_width)
        points.append(ConePoint(horizon=step, lower=round(lower, 6), center=round(center, 6), upper=round(upper, 6)))
    return ProbabilityCone(
        mean_probability=result.mean_probability,
        uncertainty_width=result.uncertainty_width,
        consensus_score=result.consensus_score,
        points=points,
    )


def describe_cone(cone: ProbabilityCone) -> str:
    if cone.consensus_score > 0.75 and cone.mean_probability >= 0.55:
        return "narrow_upward"
    if cone.consensus_score > 0.75 and cone.mean_probability <= 0.45:
        return "narrow_downward"
    if cone.uncertainty_width >= 0.35:
        return "wide_flat"
    return "moderate"
