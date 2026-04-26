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
    quantile_lower: float = 0.0
    quantile_upper: float = 1.0
    tail_risk_score: float = 0.0


def build_probability_cone(result: CollapseResult, horizon_steps: int = 5) -> ProbabilityCone:
    points: List[ConePoint] = []
    lower_anchor = result.quantile_lower if hasattr(result, 'quantile_lower') else max(0.0, result.mean_probability - result.uncertainty_width * 0.5)
    upper_anchor = result.quantile_upper if hasattr(result, 'quantile_upper') else min(1.0, result.mean_probability + result.uncertainty_width * 0.5)
    for step in range(1, horizon_steps + 1):
        scale = step / horizon_steps
        lower = max(0.0, result.mean_probability - (result.mean_probability - lower_anchor) * scale)
        upper = min(1.0, result.mean_probability + (upper_anchor - result.mean_probability) * scale)
        points.append(ConePoint(horizon=step, lower=round(lower, 6), center=round(result.mean_probability, 6), upper=round(upper, 6)))
    return ProbabilityCone(
        mean_probability=result.mean_probability,
        uncertainty_width=result.uncertainty_width,
        consensus_score=result.consensus_score,
        points=points,
        quantile_lower=round(float(lower_anchor), 6),
        quantile_upper=round(float(upper_anchor), 6),
        tail_risk_score=round(float(getattr(result, 'tail_risk_score', 0.0)), 6),
    )


def describe_cone(cone: ProbabilityCone) -> str:
    if cone.consensus_score > 0.75 and cone.mean_probability >= 0.55:
        return "narrow_upward"
    if cone.consensus_score > 0.75 and cone.mean_probability <= 0.45:
        return "narrow_downward"
    if cone.tail_risk_score >= 0.55 or cone.uncertainty_width >= 0.35:
        return "wide_flat"
    return "moderate"
