from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev
from typing import Sequence

from src.mcts.tree import SimulationNode


@dataclass
class CollapseResult:
    mean_probability: float
    uncertainty_width: float
    consensus_score: float
    dominant_driver: str
    quantile_lower: float = 0.0
    quantile_upper: float = 1.0
    tail_risk_score: float = 0.0
    minority_share: float = 0.0


def leaf_probability(leaf: SimulationNode) -> float:
    if leaf.state is None:
        return 0.5
    return max(0.0, min(1.0, (leaf.state.directional_bias + 1.0) / 2.0))


def _weighted_quantile(values: list[float], weights: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(max(float(weight), 1e-9) for _, weight in ordered)
    target = max(0.0, min(1.0, quantile)) * total
    running = 0.0
    for value, weight in ordered:
        running += max(float(weight), 1e-9)
        if running >= target:
            return float(value)
    return float(ordered[-1][0])


def reverse_collapse(leaves: Sequence[SimulationNode]) -> CollapseResult:
    if not leaves:
        raise ValueError("reverse_collapse requires at least one leaf")

    probabilities = [leaf_probability(leaf) for leaf in leaves]
    weights = [
        max(
            leaf.probability_weight
            * (0.52 + (0.22 * float(leaf.branch_fitness)) + (0.12 * float(leaf.analog_confidence)) + (0.08 * float(leaf.minority_guardrail))),
            1e-9,
        )
        for leaf in leaves
    ]
    weight_sum = sum(weights)
    normalized = [weight / weight_sum for weight in weights]
    weighted_mean = sum(probability * weight for probability, weight in zip(probabilities, normalized))
    directional_biases = [float((probability - 0.5) * 2.0) for probability in probabilities]
    dispersion = pstdev(probabilities) if len(probabilities) > 1 else 0.0
    quantile_lower = _weighted_quantile(probabilities, normalized, 0.10)
    quantile_upper = _weighted_quantile(probabilities, normalized, 0.90)
    weighted_analog = sum(float(leaf.analog_confidence) * weight for leaf, weight in zip(leaves, normalized))
    directional_consensus = abs(sum(bias * weight for bias, weight in zip(directional_biases, normalized)))
    minority_share = sum(
        weight
        for probability, weight in zip(probabilities, normalized)
        if (probability >= 0.5) != (weighted_mean >= 0.5)
    )
    tail_risk_score = min(1.0, max(0.0, (quantile_upper - quantile_lower) / 0.35))
    consensus = max(
        0.0,
        min(
            1.0,
            (0.46 * (1.0 - min(1.0, dispersion / 0.25)))
            + (0.24 * directional_consensus)
            + (0.16 * weighted_analog)
            + (0.08 * (1.0 - tail_risk_score))
            + (0.06 * (1.0 - min(1.0, minority_share / 0.35))),
        ),
    )

    dominant_counts = {}
    for leaf in leaves:
        dominant_counts[leaf.dominant_driver] = dominant_counts.get(leaf.dominant_driver, 0.0) + leaf.probability_weight
    dominant_driver = max(dominant_counts, key=dominant_counts.get)

    return CollapseResult(
        mean_probability=round(weighted_mean, 6),
        uncertainty_width=round(min(1.0, max(dispersion * 2.5, (quantile_upper - quantile_lower))), 6),
        consensus_score=round(consensus, 6),
        dominant_driver=dominant_driver,
        quantile_lower=round(float(quantile_lower), 6),
        quantile_upper=round(float(quantile_upper), 6),
        tail_risk_score=round(float(tail_risk_score), 6),
        minority_share=round(float(minority_share), 6),
    )
